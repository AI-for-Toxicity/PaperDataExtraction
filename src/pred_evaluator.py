import re
import csv
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from pathlib import Path

from common import PROMPT_INSTRUCTIONS, norm, contains_normalized_substring, contains_wordbound, fuzzy_score, compute_score


# Minimum fuzzy chemical similarity required to allow a fuzzy event match
MIN_CHEM_SIM = 0.4


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
        Event type is ignored — a correct chemical+description pair counts regardless
        of whether MIE/KE/AO is predicted correctly.
        Chemical similarity is used as a hard gate (MIN_CHEM_SIM) to prevent
        unrelated chemicals from matching via description alone.
        """
        chem_p = pred_ev.get("chemical", "")
        chem_g = gold_ev.get("chemical", "")
        desc_p = pred_ev.get("description", "")
        desc_g = gold_ev.get("description", "")

        # Exact variant match gives full chemical score; otherwise fall back to fuzzy
        if self._chemicals_match(chem_p, chem_g):
            chem_sim = 1.0
        else:
            chem_sim = fuzzy_score(chem_p, chem_g)
            if chem_sim < MIN_CHEM_SIM:
                return 0.0

        desc_sim = fuzzy_score(desc_p, desc_g)  # 0..1

        # Description matters more than chemical string casing/variants
        return 0.4 * chem_sim + 0.6 * desc_sim

    def event_cosine_similarity(self, pred_ev: dict, gold_ev: dict) -> float:
        """
        Event type is ignored — a correct chemical+description pair counts regardless
        of whether MIE/KE/AO is predicted correctly.
        Chemical similarity is used as a hard gate (MIN_CHEM_SIM) to prevent
        unrelated chemicals from matching via description alone.
        """
        chem_p = pred_ev.get("chemical", "")
        chem_g = gold_ev.get("chemical", "")
        desc_p = pred_ev.get("description", "")
        desc_g = gold_ev.get("description", "")

        # Chemical match — gate on minimum string similarity to block unrelated chemicals
        if self._chemicals_match(chem_p, chem_g):
            chem_sim = 1.0
        else:
            chem_sim = fuzzy_score(chem_p, chem_g)
            if chem_sim < MIN_CHEM_SIM:
                return 0.0

        # Semantic description match
        emb_p = self.embedder.encode(desc_p, convert_to_tensor=True)
        emb_g = self.embedder.encode(desc_g, convert_to_tensor=True)
        desc_sim = util.cos_sim(emb_p, emb_g).item()

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
                    norm(pe.get("description", "")) == norm(ge.get("description", ""))
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

                # threshold=1.1 disables semantic dedup for gold — deduplicating gold events
                # can collapse legitimately distinct annotations, deflating recall unfairly.
                gold_events = self.parse_event_lines_to_list(gold_s, dedup_threshold=1.1)
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
                gold_events = self.parse_event_lines_to_list(rec.get("gold", "") or "", dedup_threshold=1.1)
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
