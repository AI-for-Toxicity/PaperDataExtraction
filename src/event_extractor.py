import csv
import io
import json
import torch
from pathlib import Path
from typing import List, Dict

from common import PROMPT_INSTRUCTIONS


class EventExtractor:
    def __init__(
        self,
        divided_md_files: List[Path],
        output_dir: Path,
        skip_existing: bool = False,
        model: str | None = None,
        model_weights: str | None = None,
    ):
        print("### EventExtractor - init ###")
        self.divided_md_files = divided_md_files
        self.output_dir = Path(output_dir)
        self.skip_existing = skip_existing

        from transformers import AutoTokenizer, AutoModelForCausalLM

        if not model:
            raise ValueError("model must be provided")

        print(f"Loading tokenizer from: {model}")
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"Loading model from: {model} (device={self.device}, dtype={dtype})")
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )

        adapter_path = Path(model_weights) if model_weights else None
        if adapter_path and (adapter_path / "adapter_config.json").exists():
            from peft import PeftModel
            print(f"Loading LoRA adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))

        self.model.eval()
        print(f"Model loaded.")

    def __enter__(self):
        return self

    def _run_inference(self, chunk_text: str) -> str:
        messages = [{"role": "user", "content": PROMPT_INSTRUCTIONS + chunk_text}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=400,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _parse_events(self, output_text: str) -> List[Dict[str, str]]:
        events = []
        for line in output_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = next(csv.reader(io.StringIO(line)))
                if len(row) >= 3:
                    events.append({
                        "chemical": row[0].strip(),
                        "event_type": row[1].strip(),
                        "description": row[2].strip(),
                    })
            except Exception:
                continue
        return events

    def extract_events(self):
        total = len(self.divided_md_files)
        print(f"Found {total} divided markdown files to process for event extraction.")

        for i, json_path in enumerate(self.divided_md_files, 1):
            print(f"[{i}/{total}] Processing {json_path.name}...")

            if not json_path.is_file():
                continue

            stem = json_path.stem
            out_path = self.output_dir / f"{stem}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if self.skip_existing and out_path.exists():
                print(f"\tSkipping {stem} because {out_path} exists.")
                continue

            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            chunks = data.get("chunks", [])
            result_chunks = []

            for chunk in chunks:
                raw_output = self._run_inference(chunk.get("text", ""))
                result_chunk = dict(chunk)
                result_chunk["predicted_events"] = self._parse_events(raw_output)
                result_chunk["raw_model_output"] = raw_output
                result_chunks.append(result_chunk)

            with out_path.open("w", encoding="utf-8") as f:
                json.dump({"chunks": result_chunks}, f, indent=2, ensure_ascii=False)

            n_events = sum(len(c["predicted_events"]) for c in result_chunks)
            print(f"\tSaved {out_path} | chunks={len(result_chunks)} events={n_events}")

        print("Event extraction complete.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("### EventExtractor - exit ###")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
