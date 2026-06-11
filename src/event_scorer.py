import re
import csv
import json
from pathlib import Path

from typing import Callable, List

from model.common import contains_wordbound, compute_score, compute_score_short_only

_ScoreFn = Callable[[str, dict], float]


class EventScorer:
    """
    Class to process the divided JSON files, load events, annotate chunks, and save results.
    """
    def __init__(
        self,
        divided_md_files: List[Path],
        output_dir: Path | str,
        label_files: List[Path] | None = None,
    ):
        self.divided_md_files = divided_md_files
        self._divided_by_stem = {f.stem: f for f in divided_md_files}
        self.label_files = label_files if label_files is not None else []
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
        score_fn: _ScoreFn = compute_score,
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
                clean_text = text

                score = score_fn(text, ev)

                # chemical_found computed per-block
                chemical_found = False
                matched_variant = chemical_raw

                if chemical_raw:
                    # Prepare the normalized chemical strings
                    norm_raw = chemical_raw
                    norm_norm = chemical_norm if chemical_norm else ""
                    norm_abbr = chemical_abbr if chemical_abbr else ""

                    # 1. Word-boundary matches on artifact-normalized strings
                    if norm_raw and contains_wordbound(clean_text, norm_raw):
                        chemical_found = True
                        matched_variant = chemical_raw
                    elif norm_norm and norm_norm != norm_raw and contains_wordbound(clean_text, norm_norm):
                        chemical_found = True
                        matched_variant = chemical_norm
                    elif norm_abbr and norm_abbr != norm_raw and contains_wordbound(clean_text, norm_abbr):
                        chemical_found = True
                        matched_variant = chemical_abbr
                    
                    # 2. Compound plurals / "No. X" aliases
                    if not chemical_found and chemical_norm:
                        # Extract the number from your label (e.g., "compound 5" -> "5")
                        match = re.search(r'(?:compound|no[.:]?)\s*(\d+)', chemical_norm, re.IGNORECASE)
                        if match:
                            num = match.group(1)
                            # Look for "compound(s)" OR "No." OR "No:" followed by the number in the ORIGINAL text
                            pattern = r'(?:compound[s]?|no[.:]?)\s*[\d,and\s]*\b' + num + r'\b'
                            if re.search(pattern, text, re.IGNORECASE):
                                chemical_found = True
                                matched_variant = chemical_norm

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
        out_path = self.output_dir / f"{base}_events.json"
        if out_path.exists():
            print(f"[SKIP] {out_path} already exists")
            return

        if self.label_files is None:
            raise ValueError("label_files is required for run_scoring / process_file")

        # labels file with matching base name
        label_path = None
        for lf in self.label_files:
            if lf.stem.startswith(base):
                label_path = lf
                break
        
        if not label_path:
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
        for json_path in self.divided_md_files:
            self.process_file(json_path)

    # -------------------------------------------------------------------------
    # Mode 2: score events predicted by EventExtractor
    # -------------------------------------------------------------------------

    def _events_from_extracted(
        self, extracted_data: dict
    ) -> tuple[list[dict], dict[int, int]]:
        """
        Convert EventExtractor output chunks into the event list format that
        annotate_blocks expects.  Also returns a map of event_id → source chunk_id
        so that unmatched events can be placed back in their original chunk.
        """
        events: list[dict] = []
        event_id_to_chunk_id: dict[int, int] = {}
        event_id = 0
        for chunk in extracted_data.get("chunks", []):
            chunk_id = chunk.get("chunk_id", -1)
            for ev in chunk.get("predicted_events", []):
                desc = ev.get("description", "")
                events.append({
                    "event_id": event_id,
                    "chemical": ev.get("chemical", ""),
                    "event_type": ev.get("event_type", ""),
                    "event_description_short": desc,
                    "event_description_long": "",
                })
                event_id_to_chunk_id[event_id] = chunk_id
                event_id += 1
        return events, event_id_to_chunk_id

    def process_extracted_file(
        self, extracted_json_path: Path, divided_json_path: Path, output_subdir: Path
    ):
        """
        Score events from one EventExtractor output file.
        Events that don't reach the scoring threshold on any textual unit are
        kept in their original extraction chunk with score=0.
        """
        stem = extracted_json_path.stem.removesuffix("_extracted")
        out_path = output_subdir / f"{stem}_events.json"
        if out_path.exists():
            print(f"[SKIP] {out_path} already exists")
            return

        with open(extracted_json_path, "r", encoding="utf-8") as f:
            extracted_data = json.load(f)
        with open(divided_json_path, "r", encoding="utf-8") as f:
            divided_data = json.load(f)

        events, event_id_to_chunk_id = self._events_from_extracted(extracted_data)
        if not events:
            print(f"[SKIP] No predicted events in {extracted_json_path.name}")
            return

        sent_dicts  = [{"text": s} for s in divided_data.get("sentences", [])]
        line_dicts  = [{"text": ln} for ln in divided_data.get("lines", [])]
        para_dicts  = [{"title": p.get("title", ""), "text": p.get("body", "")} for p in divided_data.get("paragraphs", [])]
        chunks_dicts = [dict(c) for c in divided_data.get("chunks", [])]

        sentences_annotated, matched_sents   = self.annotate_blocks(sent_dicts,   events, "text", score_fn=compute_score_short_only, top_k=1)
        lines_annotated,     matched_lines   = self.annotate_blocks(line_dicts,   events, "text", score_fn=compute_score_short_only, top_k=1)
        paragraphs_annotated, matched_paras  = self.annotate_blocks(para_dicts,   events, "text", score_fn=compute_score_short_only, top_k=1)
        chunks_annotated,    matched_chunks  = self.annotate_blocks(chunks_dicts, events, "text", score_fn=compute_score_short_only, top_k=None)

        all_matched = matched_sents | matched_lines | matched_paras | matched_chunks
        unmatched   = [ev for ev in events if ev["event_id"] not in all_matched]

        # Place unmatched events back in their original extraction chunk (score=0)
        chunk_id_to_idx = {c.get("chunk_id", i): i for i, c in enumerate(chunks_annotated)}
        for ev in unmatched:
            idx = chunk_id_to_idx.get(event_id_to_chunk_id.get(ev["event_id"], -1))
            if idx is not None:
                ev_entry = dict(ev)
                ev_entry["score"] = 0.0
                ev_entry["chemical_found"] = False
                ev_entry["matched_chemical_variant"] = ev["chemical"]
                chunks_annotated[idx]["events"].append(ev_entry)

        output = {
            "sentences": sentences_annotated,
            "lines": lines_annotated,
            "paragraphs": paragraphs_annotated,
            "chunks": chunks_annotated,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(
            f"[OK] Saved {out_path} | "
            f"events={len(events)} matched={len(all_matched)} fallback={len(unmatched)}"
        )

    def run_scoring_from_extracted(self, extracted_files: List[Path], folder: str):
        output_subdir = self.output_dir / folder
        output_subdir.mkdir(parents=True, exist_ok=True)

        for extracted_path in extracted_files:
            divided_stem = extracted_path.stem.removesuffix("_extracted")
            divided_path = self._divided_by_stem.get(divided_stem)
            if not divided_path:
                print(f"[WARN] No divided JSON for {extracted_path.name}")
                continue
            self.process_extracted_file(extracted_path, divided_path, output_subdir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
