from collections import defaultdict
import json
import csv
import random
import re
import sys
import torch
from sentence_transformers import SentenceTransformer, util
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

PROMPT_INSTRUCTIONS = "You are an assistant specialized in extracting mechanistic toxicology events (MIE, KE, AO) from scientific text.\n\nGiven the following text from a toxicology article, extract all MIE, KE and AO events with the associated chemical and a concise description.\n\nReturn one event per line in the exact format:\n\"chemical\",\"event_type\",\"description\"\n\nIf the text does not contain any MIE, KE or AO events, return an empty output.\n\nText:\n"
RESPONSE_SUFFIX = "### END"

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

def compute_score(text: str, event: dict) -> float:
    """
    Returns a score in 0..100 based on description matching only.
    Perfect (100) only if long or short description exact match.
    Chemical presence is handled separately as a boolean flag.

    desc_l is treated as a verbatim quote from the paper, so fuzz.partial_ratio
    (best-window substring alignment) is used directly for it rather than
    fuzzy_score (which blends token_set_ratio and would fire on shared domain
    vocabulary across many unrelated chunks).
    desc_s is a human-written label/paraphrase, so fuzzy_score (blend) is kept.
    """
    desc_s = event.get("event_description_short", "")
    desc_l = event.get("event_description_long", "")

    # Clean truncation artifacts from the long description
    desc_l = re.sub(r"\s*\.\.\.\s*", " ", desc_l)
    desc_l = desc_l.replace("…", " ")
    desc_l = re.sub(r"\s{2,}", " ", desc_l)

    # Perfect condition: long or short exact match
    if contains_normalized_substring(text, desc_s) or contains_normalized_substring(text, desc_l):
        return 100.0

    t = norm(text)
    p_s = norm(desc_s)
    p_l = norm(desc_l)

    # Short description: blended fuzzy (partial_ratio + token_set_ratio)
    pr_s = fuzz.partial_ratio(t, p_s) / 100.0
    ts_s = fuzz.token_set_ratio(t, p_s) / 100.0
    fs = max(pr_s * 0.7 + ts_s * 0.3, ts_s * 0.6 + pr_s * 0.4)

    # Long description: pure partial_ratio — verbatim quote should align as a
    # window inside the chunk text; token_set_ratio would produce false positives
    # from shared domain terms (firing rate, network burst, etc.)
    fl = fuzz.partial_ratio(t, p_l) / 100.0 if desc_l else fs

    score = 10.0
    score += 20.0 * fs
    score += 60.0 * fl

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
    def __init__(self, eval_jsonl_path: str | Path, output_path: str | Path, full_eval_analysis_folder_path: str | Path, split_info_path: str | Path,) -> None:
        self.eval_jsonl_path = Path(eval_jsonl_path)
        self.output_path = Path(output_path)
        self.full_eval_analysis_folder_path = Path(full_eval_analysis_folder_path)
        self.instruction_prefix = PROMPT_INSTRUCTIONS
        self.split_info_path = Path(split_info_path)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def __enter__(self):
        return self

    def parse_event_lines_to_list(self, s: str, dedup_threshold: float = 0.85) -> list[dict]:
        """
        Parses a string containing lines like:
        "chemical","event_type","description"

        Returns list of dicts:
        {"chemical": ..., "event_type": ..., "description": ...}
        
        Includes semantic deduplication to merge redundant predictions 
        (e.g., "Inhibition of X" and "Decreased X") for the same chemical and event type.
        """
        if not s:
            return []

        events = []
        for raw_line in s.splitlines():
            line = raw_line.strip()
            if not line:
                continue

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

        # --- SEMANTIC DEDUPLICATION LOGIC ---
        deduped = []
        
        # 1. Group events by (normalized chemical, event_type)
        grouped_events = defaultdict(list)
        for ev in events:
            group_key = (norm(ev["chemical"]), ev["event_type"])
            grouped_events[group_key].append(ev)
            
        # 2. Deduplicate descriptions within each group using Cosine Similarity
        for group_key, ev_list in grouped_events.items():
            unique_in_group = []
            
            for ev in ev_list:
                desc = ev["description"]
                # Encode the description
                emb = self.embedder.encode(desc, convert_to_tensor=True)
                
                is_duplicate = False
                # Compare against already accepted unique events in this group
                for unique_ev, unique_emb in unique_in_group:
                    sim = util.cos_sim(emb, unique_emb).item()
                    
                    # If the semantic similarity hits the threshold, toss it as a duplicate
                    if sim >= dedup_threshold:
                        is_duplicate = True
                        break
                        
                if not is_duplicate:
                    unique_in_group.append((ev, emb))
                    deduped.append(ev)
                    
        return deduped

    @staticmethod
    def _chemical_variants(chem: str) -> list[str]:
        """
        Return all normalized name variants for a chemical string.
        Handles parenthetical abbreviations and slash separators:
          'Diclofenac acyl glucuronide (DCF-AG)' -> ['diclofenac acyl glucuronide (dcf-ag)', 'diclofenac acyl glucuronide', 'dcf-ag']
          'Roundup / glyphosate' -> ['roundup / glyphosate', 'roundup', 'glyphosate']
        """
        chem = (chem or "").strip()
        if not chem:
            return []

        variants: set[str] = {norm(chem)}

        if " (" in chem and chem.endswith(")"):
            last_paren = chem.rfind(" (")
            base = chem[:last_paren].strip()
            abbr = chem[last_paren + 2:-1].strip()
            if base:
                variants.add(norm(base))
            if abbr:
                variants.add(norm(abbr))

        if " / " in chem:
            for part in chem.split(" / "):
                if part.strip():
                    variants.add(norm(part.strip()))

        return list(variants)

    @classmethod
    def _chemicals_match(cls, chem_a: str, chem_b: str) -> bool:
        """True if any normalized variant of chem_a matches any normalized variant of chem_b."""
        return bool(set(cls._chemical_variants(chem_a)) & set(cls._chemical_variants(chem_b)))

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

        # Exact variant match gives full chemical score; otherwise fall back to fuzzy
        chem_sim = 1.0 if self._chemicals_match(chem_p, chem_g) else fuzzy_score(chem_p, chem_g)
        desc_sim = fuzzy_score(desc_p, desc_g)  # 0..1

        # Description matters more than chemical string casing/variants
        return 0.4 * chem_sim + 0.6 * desc_sim

    def event_cosine_similarity(self, pred_ev: dict, gold_ev: dict) -> float:
        if (pred_ev.get("event_type") or "").upper() != (gold_ev.get("event_type") or "").upper():
            return 0.0

        chem_p = pred_ev.get("chemical", "")
        chem_g = gold_ev.get("chemical", "")
        desc_p = pred_ev.get("description", "")
        desc_g = gold_ev.get("description", "")

        # Chemical match
        chem_sim = 1.0 if self._chemicals_match(chem_p, chem_g) else fuzzy_score(chem_p, chem_g)
        
        # Semantic Description match
        emb_p = self.embedder.encode(desc_p, convert_to_tensor=True)
        emb_g = self.embedder.encode(desc_g, convert_to_tensor=True)
        desc_sim = util.cos_sim(emb_p, emb_g).item() # Returns a score from -1 to 1

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
        sim_threshold: float = 0.65
    ) -> dict:
        """
        Computes:
        - similar_to_gold: count of pred events that match a gold event (exact or fuzzy >= threshold)
        - not_in_gold: count of pred events that couldn't be matched
        - gold_not_found: count of gold events not matched by any pred

        Uses greedy one-to-one matching on best similarity.
        Exact match uses chemical variant normalization (_chemicals_match) so that
        e.g. 'Roundup (glyphosate)' and 'Roundup' are treated as the same chemical.
        """
        # Exact match with chemical variant awareness (greedy one-to-one)
        exact_matched_gold: set[int] = set()
        exact_matched_pred: set[int] = set()
        exact_match_pairs: list[tuple[dict, dict]] = []  # (gold_ev, pred_ev)
        for pi, pe in enumerate(pred_events):
            for gi, ge in enumerate(gold_events):
                if gi in exact_matched_gold:
                    continue
                if (
                    (pe.get("event_type") or "").upper() == (ge.get("event_type") or "").upper()
                    and norm(pe.get("description", "")) == norm(ge.get("description", ""))
                    and self._chemicals_match(pe.get("chemical", ""), ge.get("chemical", ""))
                ):
                    exact_matched_gold.add(gi)
                    exact_matched_pred.add(pi)
                    exact_match_pairs.append((ge, pe))
                    break

        # Build remaining lists for fuzzy pass
        gold_remaining = [e for i, e in enumerate(gold_events) if i not in exact_matched_gold]
        pred_remaining = [e for i, e in enumerate(pred_events) if i not in exact_matched_pred]

        matched_gold: set[int] = set()
        matched_pred: set[int] = set()

        # Score all pairs (could be big, but usually gold is small)
        candidates = []
        for pi, pe in enumerate(pred_remaining):
            for gi, ge in enumerate(gold_remaining):
                sim = self.event_cosine_similarity(pe, ge)
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

        similar_to_gold = len(exact_matched_pred) + len(fuzzy_matches)
        not_in_gold = len(pred_events) - similar_to_gold
        gold_not_found = len(gold_events) - similar_to_gold

        matched_pairs: list[tuple[dict, dict]] = (
            exact_match_pairs + [(ge, pe) for _, pe, ge in fuzzy_matches]
        )
        unmatched_gold = [e for i, e in enumerate(gold_remaining) if i not in matched_gold]
        unmatched_pred = [e for i, e in enumerate(pred_remaining) if i not in matched_pred]

        return {
            "similar_to_gold": similar_to_gold,
            "not_in_gold": not_in_gold,
            "gold_not_found": gold_not_found,
            "exact_hits": len(exact_matched_pred),
            "fuzzy_hits": len(fuzzy_matches),
            "fuzzy_matches": fuzzy_matches,
            "matched_pairs": matched_pairs,
            "unmatched_gold": unmatched_gold,
            "unmatched_pred": unmatched_pred,
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
            score = compute_score(chunk_text, fake_gold)

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

    @staticmethod
    def _sort_events(events: list[dict]) -> list[dict]:
        """Sort events by chemical (alpha), event_type (MIE→KE→AO), description (alpha)."""
        order = {"MIE": 0, "KE": 1, "AO": 2}
        return sorted(
            events,
            key=lambda e: (
                (e.get("chemical") or "").lower(),
                order.get((e.get("event_type") or "").upper(), 99),
                (e.get("description") or "").lower(),
            ),
        )

    def generate_chunk_md_files(
        self,
        folder: Path,
        chunk_text: str,
        gold_events: list[dict],
        pred_events: list[dict],
        chunk_index: int,
    ) -> None:
        """Write a markdown file for one chunk into *folder*."""
        folder.mkdir(parents=True, exist_ok=True)

        def events_md(events: list[dict]) -> str:
            if not events:
                return "_none_\n"
            lines = []
            for ev in self._sort_events(events):
                chem = ev.get("chemical", "")
                etype = (ev.get("event_type") or "").upper()
                desc = ev.get("description", "")
                lines.append(f"- **{chem}** | {etype} | {desc}")
            return "\n".join(lines) + "\n"

        md = (
            f"# Chunk {chunk_index}\n\n"
            f"## Text\n\n{chunk_text}\n\n"
            f"## Gold events\n\n{events_md(gold_events)}\n"
            f"## Pred events\n\n{events_md(pred_events)}\n"
        )

        out_file = folder / f"chunk_{chunk_index:04d}.md"
        out_file.write_text(md, encoding="utf-8")

    def analyze_eval_jsonl(
        self,
        text_match_threshold: float = 50.0,
        show_top_scored_pred: int = 8,
        show_top_fuzzy: int = 5,
        limit: int | None = None,
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

        out_path = self.output_path
        md_folder = self.full_eval_analysis_folder_path

        lines_out: list[str] = []

        def emit(s: str = "") -> None:
            lines_out.append(s)

        tot_sim = tot_fp = tot_fn = 0
        tot_gold = tot_pred = 0
        tot_text_match = tot_text_no_match = 0
        n = 0

        # Load paper→chunk mapping for deduplicated summary
        chunk_to_paper: dict[int, str] = {}
        if self.split_info_path.exists():
            with self.split_info_path.open("r", encoding="utf-8") as f:
                split_info = json.load(f)
            for paper in split_info.get("test", {}).get("papers", []):
                for cid in paper.get("chunk_ids", []):
                    chunk_to_paper[cid] = paper["id"]
        paper_chunks: dict[str, list[tuple[list, list]]] = {}

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

                cmp = self.compare_gold_pred(gold_events, pred_events)
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

                paper_id = chunk_to_paper.get(n)
                if paper_id is not None:
                    paper_chunks.setdefault(paper_id, []).append((gold_events, pred_events))

                if md_folder is not None:
                    self.generate_chunk_md_files(md_folder, chunk_text, gold_events, pred_events, n)

                if limit is not None and n >= limit:
                    break

        emit()
        emit("#" * 90)
        emit("[SUMMARY]")
        emit(f"Records: {n}")
        emit(f"Total gold events (chunk-level): {tot_gold} | Total pred events (chunk-level): {tot_pred}")
        emit(f"Total similar-to-gold: {tot_sim} | Total not-in-gold (FP): {tot_fp} | Total gold-not-found (FN): {tot_fn}")
        emit(
            f"Total pred grounded in text: {tot_text_match} match, {tot_text_no_match} don't match "
            f"(threshold={text_match_threshold:.0f})"
        )

        # Deduplicated paper-level P/R/F1
        if paper_chunks:
            ded_sim = ded_fp = ded_fn = ded_gold = ded_pred = 0
            for _, chunks in sorted(paper_chunks.items()):
                seen_gold: set[tuple] = set()
                deduped_gold: list[dict] = []
                seen_pred: set[tuple] = set()
                deduped_pred: list[dict] = []
                for gold_evs, pred_evs in chunks:
                    for ev in gold_evs:
                        key = (norm(ev.get("chemical", "")), (ev.get("event_type") or "").upper(), norm(ev.get("description", "")))
                        if key not in seen_gold:
                            seen_gold.add(key)
                            deduped_gold.append(ev)
                    for ev in pred_evs:
                        key = (norm(ev.get("chemical", "")), (ev.get("event_type") or "").upper(), norm(ev.get("description", "")))
                        if key not in seen_pred:
                            seen_pred.add(key)
                            deduped_pred.append(ev)
                cmp_ded = self.compare_gold_pred(deduped_gold, deduped_pred)
                ded_sim += cmp_ded["similar_to_gold"]
                ded_fp += cmp_ded["not_in_gold"]
                ded_fn += cmp_ded["gold_not_found"]
                ded_gold += len(deduped_gold)
                ded_pred += len(deduped_pred)
            emit(f"Paper-level dedup — Gold: {ded_gold} | Pred: {ded_pred} | TP: {ded_sim} | FP: {ded_fp} | FN: {ded_fn}")
            prec = (ded_sim / ded_pred) if ded_pred else 0.0
            rec = (ded_sim / ded_gold) if ded_gold else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        else:
            prec = (tot_sim / tot_pred) if tot_pred else 0.0
            rec = (tot_sim / tot_gold) if tot_gold else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        emit(f"Precision≈{prec:.3f} Recall≈{rec:.3f} F1≈{f1:.3f}")

        with out_path.open("w", encoding="utf-8") as fout:
            fout.write("\n".join(lines_out) + "\n")

        print(f"Analysis saved to {out_path}")
        print(f"Full per-record analysis markdown files saved to {md_folder}")

    def analyze_eval_jsonl_per_paper(
        self,
    ) -> None:
        """
        Groups eval JSONL records by paper using split_info.json, deduplicates gold and pred
        events across all chunks of each paper, then runs compare_gold_pred at paper level.

        The chunk_ids in split_info.json are 1-based line numbers in the eval JSONL file,
        so record at line N corresponds to chunk_id N.

        Outputs eval_analysis_papers.json next to eval_analysis.txt.
        """
        split_info_path = self.split_info_path
        if not split_info_path.exists():
            raise FileNotFoundError(f"split_info.json not found: {split_info_path}")

        with split_info_path.open("r", encoding="utf-8") as f:
            split_info = json.load(f)

        # Build chunk_id (1-based line number) → paper_id from the test split
        chunk_to_paper: dict[int, str] = {}
        for paper in split_info.get("test", {}).get("papers", []):
            paper_id = paper["id"]
            for cid in paper.get("chunk_ids", []):
                chunk_to_paper[cid] = paper_id

        if not chunk_to_paper:
            raise RuntimeError("No chunk→paper mapping found in split_info.json (test split).")

        path = self.eval_jsonl_path
        if not path.exists():
            raise FileNotFoundError(f"Eval jsonl not found: {path}")

        # Group (gold_events, pred_events) pairs by paper
        paper_chunks: dict[str, list[tuple[list, list]]] = {}
        with path.open("r", encoding="utf-8") as f:
            line_idx = 0
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                line_idx += 1  # 1-based chunk id
                paper_id = chunk_to_paper.get(line_idx)
                if paper_id is None:
                    continue
                gold_events = self.parse_event_lines_to_list(rec.get("gold", "") or "")
                pred_events = self.parse_event_lines_to_list(rec.get("pred", "") or "")
                paper_chunks.setdefault(paper_id, []).append((gold_events, pred_events))

        # Per-paper analysis: deduplicate events across chunks then compare
        results = []
        tot_sim = tot_fp = tot_fn = tot_gold = tot_pred = 0

        for paper_id, chunks in sorted(paper_chunks.items()):
            seen_gold: set[tuple] = set()
            deduped_gold: list[dict] = []
            seen_pred: set[tuple] = set()
            deduped_pred: list[dict] = []

            for gold_evs, pred_evs in chunks:
                for ev in gold_evs:
                    key = (norm(ev.get("chemical", "")), (ev.get("event_type") or "").upper(), norm(ev.get("description", "")))
                    if key not in seen_gold:
                        seen_gold.add(key)
                        deduped_gold.append(ev)
                for ev in pred_evs:
                    key = (norm(ev.get("chemical", "")), (ev.get("event_type") or "").upper(), norm(ev.get("description", "")))
                    if key not in seen_pred:
                        seen_pred.add(key)
                        deduped_pred.append(ev)

            cmp = self.compare_gold_pred(deduped_gold, deduped_pred)

            n_gold = len(deduped_gold)
            n_pred = len(deduped_pred)
            sim = cmp["similar_to_gold"]
            fp = cmp["not_in_gold"]
            fn = cmp["gold_not_found"]
            prec = sim / n_pred if n_pred else 0.0
            rec = sim / n_gold if n_gold else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

            def _ev_line(ev: dict) -> str:
                return f'{ev.get("chemical","")} | {(ev.get("event_type") or "").upper()} | {ev.get("description","")}'

            def _events_text(events: list[dict]) -> str:
                if not events:
                    return "(none)"
                return "\n".join(_ev_line(ev) for ev in self._sort_events(events))

            def _matched_text(pairs: list[tuple[dict, dict]]) -> str:
                if not pairs:
                    return "(none)"
                return "\n\n".join(
                    f"[GOLD] {_ev_line(ge)}\n[PRED] {_ev_line(pe)}"
                    for ge, pe in pairs
                )

            results.append({
                "paper_id": paper_id,
                "n_chunks": len(chunks),
                "n_gold_events": n_gold,
                "n_pred_events": n_pred,
                "similar_to_gold": sim,
                "not_in_gold_fp": fp,
                "gold_not_found_fn": fn,
                "exact_hits": cmp["exact_hits"],
                "fuzzy_hits": cmp["fuzzy_hits"],
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "gold_events": self._sort_events(deduped_gold),
                "pred_events": self._sort_events(deduped_pred),
                "matched_text": _matched_text(cmp["matched_pairs"]),
                "unmatched_gold_text": _events_text(cmp["unmatched_gold"]),
                "unmatched_pred_text": _events_text(cmp["unmatched_pred"]),
            })

            tot_sim += sim
            tot_fp += fp
            tot_fn += fn
            tot_gold += n_gold
            tot_pred += n_pred

        agg_prec = tot_sim / tot_pred if tot_pred else 0.0
        agg_rec = tot_sim / tot_gold if tot_gold else 0.0
        agg_f1 = (2 * agg_prec * agg_rec / (agg_prec + agg_rec)) if (agg_prec + agg_rec) else 0.0

        _text_keys = {"matched_text", "unmatched_gold_text", "unmatched_pred_text"}
        output = {
            "summary": {
                "n_papers": len(results),
                "total_gold_events": tot_gold,
                "total_pred_events": tot_pred,
                "total_similar_to_gold": tot_sim,
                "total_not_in_gold_fp": tot_fp,
                "total_gold_not_found_fn": tot_fn,
                "precision": round(agg_prec, 4),
                "recall": round(agg_rec, 4),
                "f1": round(agg_f1, 4),
            },
            "papers": [{k: v for k, v in p.items() if k not in _text_keys} for p in results],
        }

        full_analysis_papers_folder = self.full_eval_analysis_folder_path / "papers"
        full_analysis_papers_folder.mkdir(parents=True, exist_ok=True)
        for paper in results:
            pid = paper["paper_id"]
            with full_analysis_papers_folder.joinpath(f"paper_{pid}_matched.txt").open("w", encoding="utf-8") as fout:
                fout.write(paper["matched_text"])
            with full_analysis_papers_folder.joinpath(f"paper_{pid}_gold.txt").open("w", encoding="utf-8") as fout:
                fout.write(paper["unmatched_gold_text"])
            with full_analysis_papers_folder.joinpath(f"paper_{pid}_pred.txt").open("w", encoding="utf-8") as fout:
                fout.write(paper["unmatched_pred_text"])

        out_path = self.output_path.parent / "eval_analysis_papers.json"
        with out_path.open("w", encoding="utf-8") as fout:
            json.dump(output, fout, indent=2, ensure_ascii=False)

        print(f"Per-paper analysis saved to {out_path}")
        print(f"Papers: {len(results)} | Gold: {tot_gold} | Pred: {tot_pred} | Precision≈{agg_prec:.3f} Recall≈{agg_rec:.3f} F1≈{agg_f1:.3f}")

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
        *,
        min_score: float = 68.0,
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
            if " [" in chemical_raw and chemical_raw.endswith("]"):
                last_bracket = chemical_raw.rfind(" [")
                chemical_norm = chemical_raw[:last_bracket].strip()
                chemical_abbr = chemical_raw[last_bracket + 2 : -1].strip()
            elif " (" in chemical_raw and chemical_raw.endswith(")"):
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

                score = compute_score(text, ev)

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
                # Keep the original label chemical; record the variant that was
                # actually found in this block separately so callers can choose.
                ev_with_score["matched_chemical_variant"] = matched_variant

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
        chunks_dicts = [dict(c) for c in data.get("chunks", [])]

        # Annotate everything
        sentences_annotated, matched_sents = self.annotate_blocks(sent_dicts, events, "text", top_k=1)
        lines_annotated, matched_lines = self.annotate_blocks(line_dicts, events, "text", top_k=1)
        paragraphs_annotated, matched_paras = self.annotate_blocks(para_dicts, events, "text", top_k=1)
        chunks_annotated, matched_chunks = self.annotate_blocks(chunks_dicts, events, "text", top_k=None)
        unmatched_events_chunk = [ev for ev in events if ev["event_id"] not in matched_chunks]
        unmatched_events_any = [ev for ev in events if ev["event_id"] not in (matched_sents | matched_lines | matched_paras | matched_chunks)]

        # Annotate lines, then sentences, then paragraphs, then chunks
        output = {
            "sentences": sentences_annotated,
            "lines": lines_annotated,
            "paragraphs": paragraphs_annotated,
            "chunks": chunks_annotated,
            "unmatched_events_chunk": unmatched_events_chunk,
            "unmatched_events_any": unmatched_events_any
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
        user_prompt = f"{PROMPT_INSTRUCTIONS}{chunk_text}"

        lines: List[str] = []
        for ev in events:
            chem = csv_quote(ev["chemical"])
            etype = csv_quote(ev["event_type"])
            desc = csv_quote(ev["description"])
            lines.append(f"{chem},{etype},{desc}")

        assistant_output = "\n".join(lines) + "\n" + RESPONSE_SUFFIX

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
        - assistant: empty output (just RESPONSE_SUFFIX to be consistent with the positive case, but no event lines)
        """
        user_prompt = f"{PROMPT_INSTRUCTIONS}{chunk_text}"

        assistant_output = RESPONSE_SUFFIX  # nessun evento → output vuoto

        return {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_output},
            ]
        }

    def extract_chunk_examples_from_file(self, path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
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

    def build_biomistral_chunk_dataset(self, test_ratio: float = 0.05, empty_ratio: float = 1.0, seed: int = 42, use_rag: bool = False) -> None:
        input_dir_path = Path(self.input_dir)
        files = sorted(input_dir_path.glob("*_events.json"))

        if not files:
            raise RuntimeError(f"No events .json files found in {input_dir_path}")

        rng = random.Random(seed)
        path_by_stem = {fp.stem: fp for fp in files}
        paper_ids = list(path_by_stem.keys())
        rng.shuffle(paper_ids)

        # --- Paper-level split ---
        n_test_papers = max(1, int(len(paper_ids) * test_ratio)) if len(paper_ids) > 1 else 0
        test_paper_ids = paper_ids[:n_test_papers]
        train_paper_ids = paper_ids[n_test_papers:]

        # --- Extract examples per split ---
        # Each example is tagged with "_paper_stem", "_chunk_text", "_events" for stats.
        # All "_"-prefixed tags are stripped before writing to jsonl.
        def collect_examples(stems: List[str]):
            pos_msgs, tagged_empty = [], []
            for stem in stems:
                pos, empties = self.extract_chunk_examples_from_file(path_by_stem[stem])
                for ex in pos:
                    msg = self.build_messages_for_chunk(ex["chunk_text"], ex["events"])
                    msg["_paper_stem"] = stem
                    msg["_chunk_text"] = ex["chunk_text"]
                    msg["_events"] = ex["events"]
                    pos_msgs.append(msg)
                for t in empties:
                    tagged_empty.append((stem, t))
            return pos_msgs, tagged_empty

        train_pos_msgs, train_tagged_empty = collect_examples(train_paper_ids)
        test_pos_msgs, test_tagged_empty = collect_examples(test_paper_ids)

        if not train_pos_msgs and not test_pos_msgs:
            raise RuntimeError("No positive (labeled) chunk examples found.")

        # --- Add negatives per split ---
        def add_negatives(pos_msgs, tagged_empty_chunks):
            out = list(pos_msgs)
            if empty_ratio <= 0 or not tagged_empty_chunks:
                return out
            rng_local = random.Random(seed + 999)
            indices = list(range(len(tagged_empty_chunks)))
            rng_local.shuffle(indices)
            n_neg = min(int(len(pos_msgs) * empty_ratio), len(tagged_empty_chunks))
            for i in indices[:n_neg]:
                stem, t = tagged_empty_chunks[i]
                msg = self.build_messages_for_empty_chunk(t)
                msg["_paper_stem"] = stem
                msg["_chunk_text"] = t
                msg["_events"] = []
                out.append(msg)
            return out

        train_examples = add_negatives(train_pos_msgs, train_tagged_empty)
        test_examples = add_negatives(test_pos_msgs, test_tagged_empty)

        # Shuffle within split
        rng2 = random.Random(seed + 1)
        rng2.shuffle(train_examples)
        rng2.shuffle(test_examples)

        # RAG augmentation
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

        # --- Build paper→chunk IDs mapping from final jsonl order (1-based line numbers) ---
        def build_paper_to_ids(examples: List[Dict[str, Any]]) -> Dict[str, List[int]]:
            mapping: Dict[str, List[int]] = {}
            for idx, ex in enumerate(examples):
                stem = ex.get("_paper_stem")
                if stem is not None:
                    mapping.setdefault(stem, []).append(idx + 1)
            return mapping

        train_paper_to_ids = build_paper_to_ids(train_examples)
        test_paper_to_ids = build_paper_to_ids(test_examples)

        # --- Write JSONL splits (strip all internal tracking tags) ---
        _internal_keys = {"_paper_stem", "_chunk_text", "_events"}
        with self.output_train_path.open("w", encoding="utf-8") as f:
            for ex in train_examples:
                record = {k: v for k, v in ex.items() if k not in _internal_keys}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with self.output_test_path.open("w", encoding="utf-8") as f:
            for ex in test_examples:
                record = {k: v for k, v in ex.items() if k not in _internal_keys}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # --- Save split metadata ---
        def compute_split_stats(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
            total_events = 0
            event_type_distribution = {"MIE": 0, "KE": 0, "AO": 0}
            chunk_lengths = []
            for ex in examples:
                text = ex.get("_chunk_text", "")
                events = ex.get("_events") or []
                chunk_lengths.append(len(text))
                total_events += len(events)
                for ev in events:
                    et = ev.get("event_type", "")
                    if et in event_type_distribution:
                        event_type_distribution[et] += 1
            n_pos = sum(1 for ex in examples if ex.get("_events"))
            avg_len = round(sum(chunk_lengths) / len(chunk_lengths), 1) if chunk_lengths else 0.0
            return {
                "total_positive_chunks": n_pos,
                "total_negative_chunks": len(examples) - n_pos,
                "total_events": total_events,
                "average_chunk_length": avg_len,
                "event_type_distribution": event_type_distribution,
            }

        def build_split_info(stems: List[str], examples: List[Dict[str, Any]], paper_to_ids: Dict[str, List[int]]):
            stats = compute_split_stats(examples)
            return {
                "total_papers": len(stems),
                "total_chunks": len(examples),
                **stats,
                "papers": [
                    {
                        "id": stem,
                        "filename": path_by_stem[stem].name,
                        "chunk_ids": paper_to_ids.get(stem, []),
                    }
                    for stem in sorted(stems)
                ],
            }

        train_info = build_split_info(train_paper_ids, train_examples, train_paper_to_ids)
        test_info = build_split_info(test_paper_ids, test_examples, test_paper_to_ids)

        all_examples = train_examples + test_examples
        global_stats = compute_split_stats(all_examples)
        split_meta = {
            "seed": seed,
            "test_ratio": test_ratio,
            "empty_ratio": empty_ratio,
            "total_papers": len(paper_ids),
            "total_chunks": len(all_examples),
            **global_stats,
            "train": train_info,
            "test": test_info,
        }

        meta_path = self.output_train_path.parent / "split_info.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(split_meta, f, indent=2, ensure_ascii=False)

        print(f"[DONE] Train: {len(train_examples)} examples from {len(train_paper_ids)} papers")
        print(f"[DONE] Test:  {len(test_examples)} examples from {len(test_paper_ids)} papers")
        print(f"[DONE] Split metadata saved to {meta_path}")

    def __exit__(self, exc_type, exc, tb):
        pass
