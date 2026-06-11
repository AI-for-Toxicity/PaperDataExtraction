# PaperDataExtraction

Thesis project for extracting mechanistic toxicology events — **MIE** (Molecular Initiating Events), **KE** (Key Events), **AO** (Adverse Outcomes) — from scientific papers, in the context of Adverse Outcome Pathways (AOPs).

## Setup

Requires **Python 3.10+**.

```bash
# Activate the virtual environment
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS

# Install dependencies
pip install -r requirements.txt        # inference + pipeline
pip install -r requirements_train.txt  # training extras
```

Always run Python from the **project root**, not from `src/`.

---

## Configuration — `config.ini`

All runtime parameters are read from `config.ini` in the project root. There is no need to pass paths or model names on the command line for the main scripts.

### `[MODEL]` — Model & chunking

| Key | Description |
|-----|-------------|
| `model` | HuggingFace model ID used for inference and tokenizer init |
| `model_weights` | Local path to trained LoRA adapter weights (output of `train.py`) |
| `model_context_tokens` | Model's context window size limit |
| `min_chunk_tokens` | Minimum tokens per markdown chunk |
| `target_chunk_tokens` | Target tokens per chunk |
| `max_chunk_tokens` | Hard maximum tokens per chunk |
| `reserved_output_tokens` | Tokens reserved for model output |
| `safety_margin_tokens` | Extra margin subtracted from usable context |

### `[DATASET]` — Dataset generation (`dev.py` only)

| Key | Description |
|-----|-------------|
| `test_ratio` | Fraction of data held out as test set |
| `empty_ratio` | Fraction of empty (negative) chunks to keep |
| `k_folds` | Number of cross-validation folds |
| `seed` | Random seed |

### `[DIRECTORIES]` — Data paths

Used by `main.py`:

| Key | Default | Description |
|-----|---------|-------------|
| `input_pdf_dir` | `data/raw` | Input PDF files |
| `raw_markdown_dir` | `data/processed/raw_markdown` | Raw extracted markdown |
| `clean_markdown_dir` | `data/processed/clean_markdown` | Cleaned markdown |
| `divided_markdown_dir` | `data/processed/divided_markdown` | Chunked JSON files |
| `extracted_events_dir` | `data/results/extracted_events` | Model extraction output |
| `scored_events_dir` | `data/results/scored_events` | Scored/annotated events |

Used by `dev.py`:

| Key | Default | Description |
|-----|---------|-------------|
| `raw_labels_dir` | `data/labels/raw` | Human-annotated label files |
| `scored_labels_dir` | `data/labels/scored` | Scored label files |
| `dataset_dir` | `data/dataset` | Output directory for generated dataset |
| `train_file_path` | `data/dataset/train.jsonl` | Training JSONL (for token check) |
| `test_file_path` | `data/dataset/test.jsonl` | Test JSONL (for token check) |
| `split_info_path` | `data/eval/split_info.json` | Fold split metadata |
| `eval_preds_path` | `data/eval/eval_preds.jsonl` | Model evaluation predictions |
| `eval_analysis_path` | `data/eval/eval_analysis.txt` | Evaluation analysis output |
| `full_eval_analysis_dir` | `data/eval/full_analysis` | Per-paper evaluation details |

---

## `src/main.py` — Extraction Pipeline

Runs the full end-to-end pipeline for extracting AOP events from PDF papers.

**Steps:**

1. **PDF → Markdown** — converts PDFs using Docling; extracts tables and body text
2. **Clean Markdown** — fixes OCR artifacts, repairs hyphenated words
3. **Divide Markdown** — chunks text into semantic units of 60–250 tokens
4. **Extract Events** — runs the fine-tuned LLM over each chunk to extract MIE/KE/AO events
5. **Score Events** — matches extracted events back to the source chunks that support them, computing a relevance score based on keyword and entity overlap
6. **Display Results** — launches an interactive results viewer

### Usage

```bash
python src/main.py [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--skip-existing` | `True` | In each step, skip files whose output already exists |
| `--run-only N` | — | Run only step N (1–6) |
| `--display-results` | — | Alias for `--run-only 6` |
| `--verbose` | — | Enable verbose logging from Docling/PyMuPDF |

### Examples

```bash
# Run the full pipeline
python src/main.py

# Re-run only event extraction (step 4)
python src/main.py --run-only 4

# Open the results viewer
python src/main.py --display-results
```

---

## `src/dev.py` — Development Tools

Utility script for dataset preparation and evaluation analysis. Does not run the main pipeline.

**Functions:**

- `--do-scoring` — scores human-annotated label files against divided markdown chunks
- `--do-dataset` — builds train/test JSONL files from scored labels, with k-fold splits
- `--do-eval-analysis` — analyses model prediction output, per-paper and aggregate
- `--do-token-check` — reports token length statistics for train/test JSONL files

### Usage

```bash
python src/dev.py [options]
```

Multiple flags can be combined in a single call.

### Examples

```bash
# Score raw labels and build the dataset
python src/dev.py --do-scoring --do-dataset

# Analyse evaluation results
python src/dev.py --do-eval-analysis

# Check token lengths in the generated dataset
python src/dev.py --do-token-check
```

---

## `src/model/train.py` — Fine-tuning

QLoRA/LoRA fine-tuning of a causal LLM (example: Llama-3.1-8B-Instruct) on the generated dataset. Checkpoints are saved every epoch; the last 3 are kept.

### Usage

```bash
python src/model/train.py \
  --base_model <model_id> \
  --train_file <path> \
  --eval_file <path> \
  --output_dir <path> \
  [options]
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--base_model` | yes | — | HuggingFace model ID |
| `--train_file` | yes | — | Path to training JSONL |
| `--eval_file` | yes | — | Path to evaluation JSONL |
| `--output_dir` | yes | — | Directory for saved checkpoints |
| `--lora_r` | no | `32` | LoRA rank (`lora_alpha` is set to `2 * lora_r`) |
| `--load_in_4bit` | no | off | Use QLoRA (4-bit NF4); omit for bf16 LoRA |
| `--max_length` | no | `2048` | Maximum sequence length |
| `--num_train_epochs` | no | `3` | Number of training epochs |
| `--per_device_train_batch_size` | no | `1` | Per-device training batch size |
| `--per_device_eval_batch_size` | no | `1` | Per-device evaluation batch size |
| `--gradient_accumulation_steps` | no | `4` | Gradient accumulation steps |
| `--learning_rate` | no | `2e-4` | Learning rate |
| `--warmup_ratio` | no | `0.03` | Warmup ratio |
| `--weight_decay` | no | `0.0` | Weight decay |
| `--logging_steps` | no | `10` | Log every N steps |
| `--seed` | no | `42` | Random seed |

### Example

```bash
python src/model/train.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --train_file dataset/train_fold_0.jsonl \
  --eval_file dataset/test_fold_0.jsonl \
  --output_dir fold_0/outputs/ \
  --num_train_epochs 8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-6 \
  --lora_r 32
```

## `src/model/eval.py` — Evaluation

Runs a trained model (base or with LoRA adapter) over an evaluation JSONL file and reports exact match, ordered line accuracy, and set-based micro P/R/F1 (for more accurate evaluation metrics, run `--eval-analysis` with the [Development Tools script](#srcdevpy--development-tools)).

### Usage

```bash
python src/model/eval.py \
  --base_model <model_id> \
  --eval_file <path> \
  [options]
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--base_model` | yes | — | HuggingFace model ID |
| `--eval_file` | yes | — | Path to evaluation JSONL |
| `--adapter_dir` | no | `""` | Path to LoRA adapter checkpoint; omit for zero-shot |
| `--max_length` | no | `2048` | Maximum input sequence length |
| `--max_new_tokens` | no | `256` | Maximum tokens to generate |
| `--temperature` | no | `0.0` | Sampling temperature (`0` = greedy) |
| `--top_p` | no | `0.95` | Top-p sampling |
| `--load_in_4bit` | no | off | Load model in 4-bit quantization |
| `--bf16` | no | off | Use bfloat16 precision |
| `--limit` | no | `0` | Evaluate only the first N examples (`0` = all) |
| `--save_preds` | no | `""` | Path to write per-example JSONL predictions |

### Example

```bash
python src/model/eval.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --adapter_dir fold_0/outputs/checkpoint-264/ \
  --eval_file dataset/test_fold_0.jsonl \
  --max_length 2048 \
  --max_new_tokens 512 \
  --save_preds fold_0/eval_preds.jsonl
```
