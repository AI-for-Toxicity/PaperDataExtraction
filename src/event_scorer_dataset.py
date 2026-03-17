import json
import csv
import random
import re
import sys
from rapidfuzz import fuzz
from typing import List, Dict, Any, Tuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    from rag.retriever import AOPRetriever as _AOPRetriever
except ImportError:
    _AOPRetriever = None

DATA_DIR = Path("test_data_old/processed/divided_markdown")
LABELS_DIR = Path("test_data_old/labels/aop_raw")
OUTPUT_DIR = Path("test_data_old/labels/scored")
FINAL_JSON_TRAIN = Path("train/train.jsonl")
FINAL_JSON_TEST = Path("train/test.jsonl")
EVAL_RESULTS_BASE = Path("train/results_3")
EVAL_RESULTS_JSONL = EVAL_RESULTS_BASE / "eval_preds.jsonl"
EVAL_ANALYSIS_RESULT = EVAL_RESULTS_BASE / "eval_analysis.txt"
RAG_INDEX_PATH = "new_data/aop_rag_index.json"

INSTR = "You are an assistant specialized in extracting mechanistic toxicology events (MIE, KE, AO) from scientific text.\n\nGiven the following text from a toxicology article, extract all MIE, KE and AO events with the associated chemical and a concise description.\n\nReturn one event per line in the exact format:\n\"chemical\",\"event_type\",\"description\"\n\nIf the text does not contain any MIE, KE or AO events, return an empty output.\n\nText:\n"

# Utils

def csv_quote(value: str) -> str:
    """
    Wrap a string in double quotes and escape internal quotes for CSV-like output.
    """
    if value is None:
        value = ""
    value = str(value)
    value = value.replace('"', '""')
    return f'"{value}"'

def normalize_whitespace(text: str) -> str:
    """
    Basic whitespace normalization to avoid duplicates.
    """
    return " ".join((text or "").split())

def norm(s: str) -> str:
    """
    Whitespace normalization + lowercase for simple case-insensitive matching.
    """
    return " ".join(s.lower().split())

def contains_normalized_substring(haystack: str, needle: str) -> bool:
    """
    Simple exact substring match ignoring case and extra whitespace. Useful for chemical variants and short descriptions.
    """
    if not needle:
        return False
    return norm(needle) in norm(haystack)

def contains_wordbound(haystack: str, needle: str) -> bool:
    """
    Word-boundary-ish match to avoid substring disasters.
    Works for multi-word needles too.
    """
    if not haystack or not needle:
        return False

    hs = haystack.lower()
    tokens = needle.strip().lower().split()
    if not tokens:
        return False

    # \w is unicode-aware in Python; use lookarounds to enforce boundaries
    pat = r"(?<![\w])" + r"\s+".join(re.escape(t) for t in tokens) + r"(?![\w])"
    return re.search(pat, hs, flags=re.UNICODE) is not None

def fuzzy_score(text: str, pattern: str) -> float:
    """
    Performs a fuzzy similarity check between text and pattern, returning a score in 0..1.
    """
    if not text or not pattern:
        return 0.0

    t = norm(text)
    p = norm(pattern)

    # partial_ratio is good for "pattern contained-ish in long text"
    pr = fuzz.partial_ratio(t, p) / 100.0
    ts = fuzz.token_set_ratio(t, p) / 100.0
    # blend: token_set helps paraphrase-y overlaps, partial helps near-substring
    return max(pr * 0.7 + ts * 0.3, ts * 0.6 + pr * 0.4)

def compute_score(text: str, event: dict, block_kind: str) -> float:
    """
    Returns a score in 0..100 based on description matching only.
    Perfect (100) only if short description exact match.
    Chemical presence is handled separately as a boolean flag.
    """
    desc_s = event.get("event_description_short", "")
    desc_l = event.get("event_description_long", "")

    # Perfect condition: short exact
    if contains_normalized_substring(text, desc_s):
        base = 95.0
        # small sanity bonus for short blocks where exact short appears
        if block_kind in {"line", "sentence"}:
            base += 5.0
        return min(100.0, base)

    # Graded scoring based on fuzzy similarity
    fs = fuzzy_score(text, desc_s)  # 0..1
    fl = fuzzy_score(text, desc_l)  # 0..1

    # Base anchored on short
    score = 10.0
    score += 70.0 * fs
    score += 20.0 * fl

    # Optional tiny structural bonus for short blocks when short fuzzy is high
    if block_kind in {"line", "sentence"} and fs >= 0.75:
        score += 3.0

    # Cap so that non-exact short never reaches 100
    return min(99.0, score)

# Currently unused
def proximity_bonus(text: str, chemical: str, desc_short: str) -> float:
    """
    Small bonus if both appear and the text is short-ish.
    We keep it simple: bonus only if both are present exactly.
    """
    if not chemical or not desc_short:
        return 0.0
    if contains_wordbound(text, chemical) and contains_normalized_substring(text, desc_short):
        # likely a very strong line/sentence alignment
        return 1.0
    return 0.0

# Classes

class PredEvaluator:
    """
    Helper class to analyze eval results in a structured way.
    """
    def __init__(self, eval_jsonl_path: str | Path, output_path: str | Path) -> None:
        self.eval_jsonl_path = Path(eval_jsonl_path)
        self.output_dir = Path(output_path)
        self.instruction_prefix = INSTR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        return self

    def parse_event_lines_to_list(self, s: str) -> list[dict]:
        """
        Parses a string containing lines like:
        "chemical","event_type","description"

        Returns list of dicts:
        {"chemical": ..., "event_type": ..., "description": ...}

        Robustness:
        - Ignores empty lines
        - Ignores malformed/truncated lines (len < 3)
        - Strips whitespace
        - Uppercases event_type
        """
        if not s:
            return []

        events = []
        for raw_line in s.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # csv.reader handles quotes/commas properly
            try:
                row = next(csv.reader([line]))
            except Exception:
                continue

            if len(row) < 3:
                continue

            chem = (row[0] or "").strip()
            etype = (row[1] or "").strip().upper()
            desc = (row[2] or "").strip()

            if not chem or not etype or not desc:
                continue

            events.append({"chemical": chem, "event_type": etype, "description": desc})

        # Deduplicate (case/space-insensitive on chem/desc)
        seen = set()
        deduped = []
        for ev in events:
            key = (norm(ev["chemical"]), ev["event_type"], norm(ev["description"]))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ev)

        return deduped

    def event_similarity(self, pred_ev: dict, gold_ev: dict) -> float:
        """
        Returns a similarity score 0..1.
        Hard requirement: event_type must match (MIE/KE/AO), otherwise 0.
        Then blend chemical similarity + description similarity.

        This is intentionally conservative, because your model loves hallucinating.
        """
        if (pred_ev.get("event_type") or "").upper() != (gold_ev.get("event_type") or "").upper():
            return 0.0

        chem_p = pred_ev.get("chemical", "")
        chem_g = gold_ev.get("chemical", "")
        desc_p = pred_ev.get("description", "")
        desc_g = gold_ev.get("description", "")

        chem_sim = fuzzy_score(chem_p, chem_g)  # 0..1
        desc_sim = fuzzy_score(desc_p, desc_g)  # 0..1

        # Description matters more than chemical string casing/variants
        return 0.4 * chem_sim + 0.6 * desc_sim

    def extract_chunk_text_from_prompt(self, prompt: str) -> str:
        """
        Prompts look like:
            <instructions...>
            Text:
            <chunk>

        This extracts what's after the "Text:" marker.
        If instruction_prefix is provided and prompt starts with it, it will be stripped first.
        """
        if not prompt:
            return ""

        p = prompt
        if self.instruction_prefix and p.startswith(self.instruction_prefix):
            p = p[len(self.instruction_prefix):]

        # Prefer last occurrence to be safe
        m = re.search(r"(?:^|\n)Text:\s*\n", p, flags=re.IGNORECASE)
        if not m:
            # fallback: sometimes "Text:" might be inline
            idx = p.lower().rfind("text:")
            if idx == -1:
                return p.strip()
            return p[idx + len("text:"):].strip()

        # Grab from the marker to end
        start = m.end()
        return p[start:].strip()

    def compare_gold_pred(
        self,
        gold_events: list[dict],
        pred_events: list[dict],
        *,
        sim_threshold: float = 0.85
    ) -> dict:
        """
        Computes:
        - similar_to_gold: count of pred events that match a gold event (exact or fuzzy >= threshold)
        - not_in_gold: count of pred events that couldn't be matched
        - gold_not_found: count of gold events not matched by any pred

        Uses greedy one-to-one matching on best similarity.
        """
        # Exact match shortcut
        gold_keys = {(norm(e["chemical"]), e["event_type"], norm(e["description"])) for e in gold_events}
        pred_keys = {(norm(e["chemical"]), e["event_type"], norm(e["description"])) for e in pred_events}

        exact_hits = pred_keys & gold_keys

        # Build lists excluding exact matches (so fuzzy matching doesn't double count)
        gold_remaining = [e for e in gold_events if (norm(e["chemical"]), e["event_type"], norm(e["description"])) not in exact_hits]
        pred_remaining = [e for e in pred_events if (norm(e["chemical"]), e["event_type"], norm(e["description"])) not in exact_hits]

        matched_gold = set()  # indices in gold_remaining
        matched_pred = set()  # indices in pred_remaining

        # Score all pairs (could be big, but usually gold is small)
        candidates = []
        for pi, pe in enumerate(pred_remaining):
            for gi, ge in enumerate(gold_remaining):
                sim = self.event_similarity(pe, ge)
                if sim >= sim_threshold:
                    candidates.append((sim, pi, gi))

        # Greedy match: highest similarity first
        candidates.sort(reverse=True, key=lambda x: x[0])
        fuzzy_matches = []
        for sim, pi, gi in candidates:
            if pi in matched_pred or gi in matched_gold:
                continue
            matched_pred.add(pi)
            matched_gold.add(gi)
            fuzzy_matches.append((sim, pred_remaining[pi], gold_remaining[gi]))

        similar_to_gold = len(exact_hits) + len(fuzzy_matches)
        not_in_gold = len(pred_events) - similar_to_gold
        gold_not_found = len(gold_events) - similar_to_gold  # because one-to-one matching

        return {
            "similar_to_gold": similar_to_gold,
            "not_in_gold": not_in_gold,
            "gold_not_found": gold_not_found,
            "exact_hits": len(exact_hits),
            "fuzzy_hits": len(fuzzy_matches),
            "fuzzy_matches": fuzzy_matches,
        }

    def score_pred_events_on_chunk(
        self,
        chunk_text: str,
        pred_events: list[dict],
    ) -> list[dict]:
        """
        For each pred event:
        - compute description score (0..100) using compute_score
        - compute chemical_found using your word-boundary checks + simple variant handling
        Returns a list of events with added fields:
        score, chemical_found
        """
        out = []
        for ev in pred_events:
            chem_raw = ev.get("chemical", "") or ""
            desc = ev.get("description", "") or ""
            etype = (ev.get("event_type", "") or "").upper()

            # Adapt to your compute_score API
            fake_gold = {
                "event_description_short": desc,
                "event_description_long": desc,
            }
            score = compute_score(chunk_text, fake_gold, "chunk")

            # chemical variant logic (same idea as annotate_blocks)
            chemical_norm = chem_raw.strip()
            chemical_abbr = None
            if " (" in chem_raw and chem_raw.endswith(")"):
                last_paren = chem_raw.rfind(" (")
                chemical_norm = chem_raw[:last_paren].strip()
                chemical_abbr = chem_raw[last_paren + 2 : -1].strip()
            elif " / " in chem_raw:
                parts = chem_raw.split(" / ", 1)
                chemical_norm = parts[0].strip()
                chemical_abbr = parts[1].strip() if len(parts) > 1 else None

            chemical_found = False
            if chem_raw and contains_wordbound(chunk_text, chem_raw):
                chemical_found = True
            elif chemical_norm and chemical_norm != chem_raw and contains_wordbound(chunk_text, chemical_norm):
                chemical_found = True
            elif chemical_abbr and contains_wordbound(chunk_text, chemical_abbr):
                chemical_found = True

            out.append({
                "chemical": chem_raw,
                "event_type": etype,
                "description": desc,
                "score": round(float(score), 3),
                "chemical_found": bool(chemical_found),
            })

        out.sort(key=lambda x: x["score"], reverse=True)
        return out

    def analyze_eval_jsonl(
        self,
        sim_threshold: float = 0.85,
        text_match_threshold: float = 50.0,
        show_top_scored_pred: int = 8,
        show_top_fuzzy: int = 5,
        limit: int | None = None
    ) -> None:
        """
        Reads a jsonl of records like:
        {"prompt": "...", "gold": "...", "pred": "...", ...}

        Writes per-record and aggregate analysis to output_path.
        Per-record:
        - gold/pred event counts
        - X similar to gold, Y not in gold, Z gold not found
        - how many pred events match the chunk text vs don't (score >= text_match_threshold)
        - top fuzzy matches and top scored pred events
        """
        path = self.eval_jsonl_path
        if not path.exists():
            raise FileNotFoundError(f"Eval jsonl not found: {path}")

        out_path = self.output_dir

        lines_out: list[str] = []

        def emit(s: str = "") -> None:
            lines_out.append(s)

        tot_sim = tot_fp = tot_fn = 0
        tot_gold = tot_pred = 0
        tot_text_match = tot_text_no_match = 0
        n = 0

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                prompt = rec.get("prompt", "") or ""
                gold_s = rec.get("gold", "") or ""
                pred_s = rec.get("pred", "") or ""

                gold_events = self.parse_event_lines_to_list(gold_s)
                pred_events = self.parse_event_lines_to_list(pred_s)

                chunk_text = self.extract_chunk_text_from_prompt(prompt)

                cmp = self.compare_gold_pred(gold_events, pred_events, sim_threshold=sim_threshold)
                scored_pred = self.score_pred_events_on_chunk(chunk_text, pred_events)

                # per-record report
                n += 1
                rid = rec.get("id", None)
                loss = rec.get("loss", None)

                emit()
                emit("=" * 90)
                emit(f"[RECORD {n}] id={rid} loss={loss}")
                emit(f"Gold events: {len(gold_events)} | Pred events: {len(pred_events)}")
                emit(
                    f"{cmp['similar_to_gold']} matches similar to gold ones "
                    f"(exact={cmp['exact_hits']}, fuzzy={cmp['fuzzy_hits']}), "
                    f"{cmp['not_in_gold']} matches not in gold, "
                    f"{cmp['gold_not_found']} matches from gold not found"
                )

                # text grounding line
                if scored_pred:
                    text_match_cnt = sum(1 for e in scored_pred if e["score"] >= text_match_threshold)
                    text_no_match_cnt = len(scored_pred) - text_match_cnt
                    emit(
                        f"Pred grounded in chunk text (score>={text_match_threshold:.0f}): "
                        f"{text_match_cnt} match, {text_no_match_cnt} don't match"
                    )
                    tot_text_match += text_match_cnt
                    tot_text_no_match += text_no_match_cnt
                else:
                    emit("Pred grounded in chunk text: no pred events to score")

                # show fuzzy matches (debug)
                if cmp["fuzzy_matches"]:
                    emit()
                    emit("Top fuzzy matches (pred -> gold):")
                    for sim, pe, ge in cmp["fuzzy_matches"][:show_top_fuzzy]:
                        emit(f"  sim={sim:.3f} | P: {pe}  ->  G: {ge}")

                # pred relevance on chunk
                if scored_pred:
                    scores = [e["score"] for e in scored_pred]
                    chem_found_cnt = sum(1 for e in scored_pred if e["chemical_found"])
                    emit()
                    emit("Pred relevance on chunk (description score 0..100):")
                    emit(
                        f"  score: min={min(scores):.1f} mean={sum(scores)/len(scores):.1f} max={max(scores):.1f} "
                        f"| chemical_found={chem_found_cnt}/{len(scored_pred)}"
                    )
                    emit(f"Top {min(show_top_scored_pred, len(scored_pred))} pred events by score:")
                    for e in scored_pred[:show_top_scored_pred]:
                        cf = "chem✓" if e["chemical_found"] else "chem✗"
                        emit(f"  {e['score']:6.1f} {cf} | {e['chemical']!r}, {e['event_type']}, {e['description']!r}")
                else:
                    emit()
                    emit("No parsable pred events to score (model probably output garbage/truncated lines).")

                # totals
                tot_sim += cmp["similar_to_gold"]
                tot_fp += cmp["not_in_gold"]
                tot_fn += cmp["gold_not_found"]
                tot_gold += len(gold_events)
                tot_pred += len(pred_events)

                if limit is not None and n >= limit:
                    break

        emit()
        emit("#" * 90)
        emit("[SUMMARY]")
        emit(f"Records: {n}")
        emit(f"Total gold events: {tot_gold} | Total pred events: {tot_pred}")
        emit(f"Total similar-to-gold: {tot_sim} | Total not-in-gold (FP): {tot_fp} | Total gold-not-found (FN): {tot_fn}")
        emit(
            f"Total pred grounded in text: {tot_text_match} match, {tot_text_no_match} don't match "
            f"(threshold={text_match_threshold:.0f})"
        )

        prec = (tot_sim / tot_pred) if tot_pred else 0.0
        rec = (tot_sim / tot_gold) if tot_gold else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        emit(f"Precision≈{prec:.3f} Recall≈{rec:.3f} F1≈{f1:.3f}")

        with out_path.open("w", encoding="utf-8") as fout:
            fout.write("\n".join(lines_out) + "\n")

        print(f"Analysis saved to {out_path}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class EventScorer:
    """
    Class to process the divided JSON files, load events, annotate chunks, and save results.
    """
    def __init__(self, json_files_dir: Path | str, labels_dir: Path | str, output_dir: Path | str):
        self.json_files_dir = Path(json_files_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        return self

    def load_events(self, labels_path: Path) -> list[dict]:
        """
        Returns the events contained in a labels file.
        """
        events = []
        with open(labels_path, "r", encoding="utf8") as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if len(row) < 4:
                    continue
                chemical, etype, desc_s, desc_l = row[:4]
                events.append({
                    "event_id": idx,
                    "chemical": chemical.strip(),
                    "event_type": etype.strip(),
                    "event_description_short": desc_s.strip(),
                    "event_description_long": desc_l.strip()
                })
        return events

    def annotate_blocks(
        self,
        blocks,
        events,
        key_text: str,
        block_kind: str,
        *,
        min_score: float = 50.0,
        require_chemical: bool = True,
        top_k: int | None = 2
    ):
        """
        Returns the blocks annotated with matching events, and the set of matched event_ids.
        """
        out = []
        matched_event_ids = set()

        # Prepare per-block event buckets
        block_events_map = [[] for _ in blocks]
        if not blocks:
            return [], matched_event_ids

        # For each event, find the blocks matching
        for ev in events:
            chemical_raw = ev.get("chemical", "") or ""

            # Prepare possible chemical variants (DO NOT mutate ev)
            chemical_norm = chemical_raw.strip()
            chemical_abbr = None
            if " (" in chemical_raw and chemical_raw.endswith(")"):
                last_paren = chemical_raw.rfind(" (")
                chemical_norm = chemical_raw[:last_paren].strip()
                chemical_abbr = chemical_raw[last_paren + 2 : -1].strip()
            elif " / " in chemical_raw:
                parts = chemical_raw.split(" / ", 1)
                chemical_norm = parts[0].strip()
                chemical_abbr = parts[1].strip() if len(parts) > 1 else None
            
            scored = []
            for i, block in enumerate(blocks):
                text = block.get(key_text, "") or ""

                score = compute_score(text, ev, block_kind)

                # chemical_found computed per-block
                chemical_found = False
                matched_variant = chemical_raw

                if chemical_raw:
                    if contains_wordbound(text, chemical_raw):
                        chemical_found = True
                        matched_variant = chemical_raw
                    elif chemical_norm and chemical_norm != chemical_raw and contains_wordbound(text, chemical_norm):
                        chemical_found = True
                        matched_variant = chemical_norm
                    elif chemical_abbr and chemical_abbr != chemical_raw and contains_wordbound(text, chemical_abbr):
                        chemical_found = True
                        matched_variant = chemical_abbr

                scored.append((score, i, chemical_found, matched_variant))

            # Filter by threshold
            kept = []
            for score, idx, chem_found, matched_variant in scored:
                if score < min_score:
                    continue
                if require_chemical and not chem_found:
                    continue
                kept.append((score, idx, chem_found, matched_variant))

            if not kept:
                continue

            kept.sort(key=lambda x: x[0], reverse=True)
            if top_k is not None:
                kept = kept[: max(1, top_k)]

            for score, idx, chem_found, matched_variant in kept:
                ev_with_score = dict(ev)
                ev_with_score["score"] = round(float(score), 3)
                ev_with_score["chemical_found"] = bool(chem_found)
                ev_with_score["chemical"] = matched_variant  # variant that matched THIS block

                block_events_map[idx].append(ev_with_score)

            matched_event_ids.add(ev["event_id"])

        for block, block_events in zip(blocks, block_events_map):
            block_events.sort(key=lambda x: x["score"], reverse=True)
            new_block = dict(block)
            new_block["events"] = block_events
            out.append(new_block)

        return out, matched_event_ids

    def process_file(self, json_path: Path):
        """
        Processes a single divided JSON file:
        - Loads the corresponding label file
        - Prepares block dicts for lines, sentences, paragraphs, chunks
        - Annotates chunks with events (strict matching)
        - Annotates lines, then sentences, then paragraphs, then chunks incrementally
        - Saves the annotated results and unmatched events to output_dir
        """
        base = json_path.name.replace("_divided.json", "")
        label_path = self.labels_dir / f"{base}.txt"

        if not label_path.exists():
            print(f"[WARN] No label file for {base}")
            return

        with open(json_path, "r", encoding="utf8") as f:
            data = json.load(f)

        events = self.load_events(label_path)

        # Prepare blocks
        sent_dicts = [{"text": s} for s in data.get("sentences", [])]
        line_dicts = [{"text": ln} for ln in data.get("lines", [])]
        para_dicts = [{"title": p.get("title", ""), "text": p.get("body", "")} for p in data.get("paragraphs", [])]
        chunks_dicts = [{"text": c.get("text", "")} for c in data.get("chunks", [])]

        # Annotate chunks only
        chunks_annotated, matched_chunks = self.annotate_blocks(chunks_dicts, events, "text", "chunk", top_k=1)
        unmatched_events_chunk = [ev for ev in events if ev["event_id"] not in matched_chunks]

        # Annotate lines, then try unmatched events on sentences, then paragraphs, then chunks (incremental matching)
        incr_sentences_annotated, incr_matched_sents = self.annotate_blocks(sent_dicts, events, "text", "sentence", top_k=1)
        events_after_sents = [ev for ev in events if ev["event_id"] not in incr_matched_sents]
        incr_lines_annotated, incr_matched_lines = self.annotate_blocks(line_dicts, events_after_sents, "text", "line", top_k=1)
        events_after_lines = [ev for ev in events_after_sents if ev["event_id"] not in incr_matched_lines]
        incr_paragraphs_annotated, incr_matched_paras = self.annotate_blocks(para_dicts, events_after_lines, "text", "paragraph", top_k=1)
        events_after_paras = [ev for ev in events_after_lines if ev["event_id"] not in incr_matched_paras]
        incr_chunks_annotated, incr_matched_chunks = self.annotate_blocks(chunks_dicts, events_after_paras, "text", "chunk", top_k=1)
        unmatched_events_incr = [ev for ev in events_after_paras if ev["event_id"] not in incr_matched_chunks]

        output = {
            "chunks": chunks_annotated,
            "unmatched_events_chunk": unmatched_events_chunk,
            "incr_sentences": incr_sentences_annotated,
            "incr_lines": incr_lines_annotated,
            "incr_paragraphs": incr_paragraphs_annotated,
            "incr_chunks": incr_chunks_annotated,
            "unmatched_events_incr": unmatched_events_incr,
        }

        out_path = self.output_dir / f"{base}_events.json"
        with open(out_path, "w", encoding="utf8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        lowest_matched_score = 0
        highest_matched_score = 0
        for c in chunks_annotated:
            for ev in c.get("events", []):
                if lowest_matched_score == 0 or ev["score"] < lowest_matched_score:
                    lowest_matched_score = ev["score"]
                if ev["score"] > highest_matched_score:
                    highest_matched_score = ev["score"]

        print(f"[OK] Saved {out_path} | events={len(events)} matched={len(matched_chunks)} unmatched={len(unmatched_events_chunk)} ls={lowest_matched_score} hs={highest_matched_score}")

    def run_scoring(self):
        """
        Runs process_file on all files inside json_files_dir
        """
        for json_path in self.json_files_dir.glob("paper_*_divided.json"):
            self.process_file(json_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class DatasetBuilder:
    """
    Class to build the final dataset for BioMistral fine-tuning.
    """
    def __init__(self, input_dir: Path | str, output_train_path: Path | str, output_test_path: Path | str, rag_index_path: Path | str = ""):
        self.input_dir = Path(input_dir)
        self.output_train_path = Path(output_train_path)
        self.output_test_path = Path(output_test_path)
        self.rag_index_path = Path(rag_index_path) if rag_index_path else None
        self.output_train_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_test_path.parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        return self

    def build_messages_for_chunk(self, chunk_text: str, events: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Positive example: chunk with events.
        - user: prompt + chunk text
        - assistant: list of lines "chemical","event_type","description"
        """
        user_prompt = f"{INSTR}{chunk_text}"

        lines: List[str] = []
        for ev in events:
            chem = csv_quote(ev["chemical"])
            etype = csv_quote(ev["event_type"])
            desc = csv_quote(ev["description"])
            lines.append(f"{chem},{etype},{desc}")

        assistant_output = "\n".join(lines) + "\n### END"

        return {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_output},
            ]
        }

    def build_messages_for_empty_chunk(self, chunk_text: str) -> Dict[str, Any]:
        """
        Negative example: chunk without events.
        - user: prompt + chunk text
        - assistant: empty output (just "### END" to be consistent with the positive case, but no event lines)
        """
        user_prompt = f"{INSTR}{chunk_text}"

        assistant_output = "### END"  # nessun evento → output vuoto

        return {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_output},
            ]
        }

    def extract_chunk_examples_from_file(self,path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Reads an *_events.json file and returns:
        - list of POSITIVE examples:
            [
            {
                "chunk_text": <chunk text>,
                "events": [
                {"chemical": ..., "event_type": ..., "description": ...},
                ...
                ]
            },
            ...
            ]
        - list of EMPTY chunks (NEGATIVE examples):
            [ <chunk_text_without_events>, ... ]
        - Uses ONLY events inside chunks (ignores unmatched_events).
        - Keeps only event_type in {MIE, KE, AO}.
        - Deduplicates by the triple (chemical, event_type, description) within each chunk.
        """
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        chunks: List[Dict[str, Any]] = data.get("chunks", [])
        positive_examples: List[Dict[str, Any]] = []
        empty_chunks: List[str] = []

        for ch in chunks:
            text = (ch.get("text") or "").strip()
            if not text:
                continue

            raw_events = ch.get("events") or []
            seen_triples: set[Tuple[str, str, str]] = set()
            collected_events: List[Dict[str, str]] = []

            for ev in raw_events:
                event_type = (ev.get("event_type") or "").strip().upper()
                if event_type not in {"MIE", "KE", "AO"}:
                    continue

                chemical = normalize_whitespace(ev.get("chemical", ""))
                short_desc = normalize_whitespace(ev.get("event_description_short", ""))
                # long_desc = normalize_whitespace(ev.get("event_description_long", ""))

                description = short_desc

                # se manca chemical o descrizione, per il tuo formato non ha senso tenerlo
                if not chemical or not description:
                    continue

                triple_key = (chemical.lower(), event_type, description.lower())
                if triple_key in seen_triples:
                    continue
                seen_triples.add(triple_key)

                collected_events.append(
                    {
                        "chemical": chemical,
                        "event_type": event_type,
                        "description": description,
                    }
                )

            if collected_events:
                positive_examples.append(
                    {
                        "chunk_text": text,
                        "events": collected_events,
                    }
                )
            else:
                # chunk con testo ma zero eventi validi → candidato NEGATIVO
                empty_chunks.append(text)

        return positive_examples, empty_chunks

    def build_biomistral_chunk_dataset(self, test_ratio: float = 0.05, empty_ratio: float = 1.0, seed: int = 42, use_rag: bool = False,) -> None:
        input_dir_path = Path(self.input_dir)
        files = sorted(input_dir_path.glob("*_events.json"))

        if not files:
            raise RuntimeError(f"No events .json files found in {input_dir_path}")

        rng = random.Random(seed)
        bases = [fp.stem for fp in files]  # paper id proxy
        rng.shuffle(bases)

        n_test_papers = max(1, int(len(bases) * test_ratio)) if len(bases) > 1 else 0
        test_bases = set(bases[:n_test_papers])
        train_bases = set(bases[n_test_papers:])

        # Load and build messages split-by-paper
        train_pos_msgs, test_pos_msgs = [], []
        train_empty, test_empty = [], []

        # map stem -> path
        path_by_stem = {fp.stem: fp for fp in files}

        def add_split(stem: str, pos_msgs: list, empty_list: list):
            pos, empties = self.extract_chunk_examples_from_file(path_by_stem[stem])
            for ex in pos:
                pos_msgs.append(self.build_messages_for_chunk(ex["chunk_text"], ex["events"]))
            empty_list.extend(empties)

        for stem in train_bases:
            add_split(stem, train_pos_msgs, train_empty)
        for stem in test_bases:
            add_split(stem, test_pos_msgs, test_empty)

        if not train_pos_msgs and not test_pos_msgs:
            raise RuntimeError("No positive (labeled) chunk examples found.")

        # Add negatives per split (so negatives don't leak either)
        def add_negatives(pos_msgs: list, empty_chunks: list, out_msgs: list):
            out_msgs.extend(pos_msgs)
            if empty_ratio <= 0 or not empty_chunks:
                return

            rng_local = random.Random(seed + 999)
            rng_local.shuffle(empty_chunks)

            n_pos = len(pos_msgs)
            n_neg_desired = int(n_pos * empty_ratio)
            n_neg = min(n_neg_desired, len(empty_chunks))

            for t in empty_chunks[:n_neg]:
                out_msgs.append(self.build_messages_for_empty_chunk(t))

        train_examples, test_examples = [], []
        add_negatives(train_pos_msgs, train_empty, train_examples)
        add_negatives(test_pos_msgs, test_empty, test_examples)

        # Shuffle within split
        rng2 = random.Random(seed + 1)
        rng2.shuffle(train_examples)
        rng2.shuffle(test_examples)

        # RAG augmentation: inject AOP Wiki context into each user message
        if use_rag and self.rag_index_path:
            if _AOPRetriever is None:
                print("[WARN] rag_index_path provided but AOPRetriever could not be imported; skipping RAG augmentation.")
            else:
                retriever = _AOPRetriever(self.rag_index_path)
                for ex in train_examples + test_examples:
                    for msg in ex["messages"]:
                        if msg["role"] == "user":
                            msg["content"] = retriever.augment(msg["content"])
                print(f"[RAG] Augmented {len(train_examples)} train + {len(test_examples)} test examples.")

        with Path(self.output_train_path).open("w", encoding="utf-8") as f_train:
            for ex in train_examples:
                f_train.write(json.dumps(ex, ensure_ascii=False) + "\n")

        with Path(self.output_test_path).open("w", encoding="utf-8") as f_test:
            for ex in test_examples:
                f_test.write(json.dumps(ex, ensure_ascii=False) + "\n")

    def __exit__(self, exc_type, exc, tb):
        pass
