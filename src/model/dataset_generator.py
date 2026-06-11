import json
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path

from model.common import PROMPT_INSTRUCTIONS, csv_quote, normalize_whitespace


RESPONSE_SUFFIX = "### END"


class DatasetBuilder:
    """
    Class to build the final dataset for model fine-tuning.
    """
    def __init__(self, input_dir: Path | str, output_train_path: Path | str, output_test_path: Path | str):
        self.input_dir = Path(input_dir)
        self.output_train_path = Path(output_train_path)
        self.output_test_path = Path(output_test_path)
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

                chemical = normalize_whitespace(ev.get("matched_chemical_variant") or ev.get("chemical", ""))
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

    def _write_split(
        self,
        train_paper_ids: List[str],
        test_paper_ids: List[str],
        path_by_stem: Dict[str, Path],
        train_path: Path,
        test_path: Path,
        meta_path: Path,
        empty_ratio: float,
        seed: int,
        extra_meta: Dict[str, Any] | None = None,
    ) -> None:
        """
        Given explicit train/test paper ID lists, collects examples, writes JSONL splits
        and a split_info JSON. Called once for a standard split or k times for k-fold.
        """
        # --- Extract examples per split ---
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
        train_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)

        with train_path.open("w", encoding="utf-8") as f:
            for ex in train_examples:
                record = {k: v for k, v in ex.items() if k not in _internal_keys}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with test_path.open("w", encoding="utf-8") as f:
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
        split_meta: Dict[str, Any] = {
            "seed": seed,
            "empty_ratio": empty_ratio,
            "total_papers": len(train_paper_ids) + len(test_paper_ids),
            "total_chunks": len(all_examples),
            **global_stats,
            **(extra_meta or {}),
            "train": train_info,
            "test": test_info,
        }

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(split_meta, f, indent=2, ensure_ascii=False)

        print(f"[DONE] Train: {len(train_examples)} examples from {len(train_paper_ids)} papers")
        print(f"[DONE] Test:  {len(test_examples)} examples from {len(test_paper_ids)} papers")
        print(f"[DONE] Split metadata saved to {meta_path}")

    def build_biomistral_chunk_dataset(
        self,
        test_ratio: float = 0.05,
        empty_ratio: float = 1.0,
        seed: int = 42,
        k_folds: int | None = None,
    ) -> None:
        """
        Builds JSONL train/test splits for fine-tuning.

        When k_folds is None (default): a single stratified split using test_ratio.
        When k_folds=k: produces k separate folds; test_ratio is ignored.
          Each paper appears in exactly one test fold.
          Output files are named with a _fold_{i} suffix:
            train_fold_0.jsonl, test_fold_0.jsonl, split_info_fold_0.json, ...
        """
        input_dir_path = Path(self.input_dir)
        files = sorted(input_dir_path.glob("*_events.json"))

        if not files:
            raise RuntimeError(f"No events .json files found in {input_dir_path}")

        rng = random.Random(seed)
        path_by_stem = {fp.stem: fp for fp in files}
        paper_ids = list(path_by_stem.keys())
        rng.shuffle(paper_ids)

        if k_folds is not None:
            if k_folds < 2:
                raise ValueError("k_folds must be >= 2")
            if k_folds > len(paper_ids):
                raise ValueError(f"k_folds ({k_folds}) > number of papers ({len(paper_ids)})")

            # Distribute papers round-robin into k folds so sizes differ by at most 1
            folds: List[List[str]] = [paper_ids[i::k_folds] for i in range(k_folds)]

            for fold_idx, test_ids in enumerate(folds):
                train_ids = [p for i, f in enumerate(folds) for p in f if i != fold_idx]

                train_path = self.output_train_path.with_stem(
                    f"{self.output_train_path.stem}_fold_{fold_idx}"
                )
                test_path = self.output_test_path.with_stem(
                    f"{self.output_test_path.stem}_fold_{fold_idx}"
                )
                meta_path = self.output_train_path.parent / f"split_info_fold_{fold_idx}.json"

                print(f"\n[K-FOLD {fold_idx + 1}/{k_folds}] test papers: {len(test_ids)}, train papers: {len(train_ids)}")
                self._write_split(
                    train_ids,
                    test_ids,
                    path_by_stem,
                    train_path,
                    test_path,
                    meta_path,
                    empty_ratio,
                    seed + fold_idx,
                    extra_meta={"k_folds": k_folds, "fold_index": fold_idx},
                )
        else:
            n_test_papers = max(1, int(len(paper_ids) * test_ratio)) if len(paper_ids) > 1 else 0
            test_paper_ids = paper_ids[:n_test_papers]
            train_paper_ids = paper_ids[n_test_papers:]
            meta_path = self.output_train_path.parent / "split_info.json"

            self._write_split(
                train_paper_ids,
                test_paper_ids,
                path_by_stem,
                self.output_train_path,
                self.output_test_path,
                meta_path,
                empty_ratio,
                seed,
                extra_meta={"test_ratio": test_ratio},
            )

    def __exit__(self, exc_type, exc, tb):
        pass
