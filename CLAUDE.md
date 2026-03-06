# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Thesis project for extracting mechanistic toxicology events — **MIE** (Molecular Initiating Events), **KE** (Key Events), **AO** (Adverse Outcomes) — from scientific papers, in the context of Adverse Outcome Pathways (AOPs) as defined in the AOP Wiki.

## Environment Setup

```bash
# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies (inference + pipeline)
pip install -r requirements.txt

# Install dependencies (includes training extras)
pip install -r requirements_train.txt
```

Always run Python from the **project root**, not from `src/`, because modules import each other without a package structure.

## Common Commands

```bash
# Run the main extraction pipeline
python src/main.py

# Train BioMistral-7B (QLoRA, detached from SSH)
./train.sh

# Evaluate fine-tuned model (detached from SSH)
./eval.sh

# Direct training invocation
python src/biomistral/BioMistralTrain.py \
  --base_model BioMistral/BioMistral-7B \
  --train_file train/train.jsonl \
  --eval_file train/test.jsonl \
  --output_dir outputs/biomistral_mie_ke_ao_qlora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --max_length 2048

# Direct evaluation invocation
python src/biomistral/BioMistralEval.py \
  --base_model BioMistral/BioMistral-7B \
  --adapter_dir outputs/biomistral_mie_ke_ao_qlora \
  --eval_file train/test.jsonl \
  --load_in_4bit --bf16 \
  --max_new_tokens 256 \
  --save_preds eval_preds.jsonl

# Monitor training/eval logs
tail -f logs/latest.log.path
```

## Architecture

### Main Pipeline (`src/main.py`)

Sequential processing pipeline:

1. **`PDFExtractor`** — Converts PDFs to Markdown using [Docling](https://github.com/DS4SD/docling); uses PyMuPDF (`fitz`) for image extraction and span-level analysis (font sizes, header/footer detection). Optionally splits multi-paper PDFs.

2. **`MarkdownCleaner`** — Cleans raw extracted Markdown: fixes OCR artifacts, repairs hyphenated broken words using Levenshtein distance + `pyspellchecker`/`wordfreq`.

3. **`MarkdownDivider`** — Chunks cleaned Markdown into semantic units of 60–250 tokens using spaCy (`en_core_web_sm`). Skips blacklisted sections (references, acknowledgements, etc.). Outputs one JSON file per paper containing a list of text chunks.

4. **`BioNERExtractor`** — Runs NER over chunks using two spaCy models (`en_ner_bionlp13cg_md`, `en_ner_bc5cdr_md`) for biomedical entities, plus `ChemDataExtractor` for chemicals. Applies species and chemical gazetteers with fuzzy matching (`rapidfuzz`).

5. **`DatasetGenerator`** (`src/DatasetGenerator.py`) — Reads labeled chunks from `test_data/labels/` and generates training/test JSONL files at `train/train.jsonl` and `train/test.jsonl` for BioMistral fine-tuning.

### BioMistral Fine-Tuning (`src/biomistral/`)

- **`BioMistralTrain.py`** — QLoRA fine-tuning of `BioMistral/BioMistral-7B` via HuggingFace `transformers` + `peft` + `bitsandbytes`.
- **`BioMistralEval.py`** — Evaluation with token-level loss, exact match, ordered line match, and set-based micro P/R/F1.
- **`BioMistralTry.py`** — Ad-hoc inference for manual testing.

The model is prompted to extract events in CSV format, one per line:
```
"chemical","event_type","description"
```

### Data Layout

```
data/resources/        # Gazetteers (species, chemicals)
test_data/raw/         # Input PDFs
test_data/processed/   # Pipeline outputs (markdown → cleaned → divided)
test_data/labels/      # Human-annotated AOP event labels
train/                 # train.jsonl, test.jsonl, eval results
outputs/               # Saved LoRA adapters
logs/                  # Training/eval logs (train.sh / eval.sh)
utils/                 # Standalone helper scripts (AOP Wiki parser, PubMed downloader, etc.)
```

### Key Data Resources

- `data/resources/species_gazetteer_ncbi.json` — NCBI taxonomy name→taxid mappings used by `BioNERExtractor`
- `train/train.jsonl` / `train/test.jsonl` — Chat-format JSONL with user+assistant messages
