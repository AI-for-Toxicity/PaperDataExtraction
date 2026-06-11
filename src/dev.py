import json
from pathlib import Path
from transformers import AutoTokenizer


def check_token_lengths(paths, max_tokens, tokenizer):
  """
  Reads train/test JSONL files, tokenizes each example using the chat template,
  and reports any entries exceeding max_tokens.
  """
  for split_name, path in [("train", paths["train_file"]), ("test", paths["test_file"])]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if line:
          data.append(json.loads(line))

    lengths = []
    over_limit = []

    for i, example in enumerate(data):
      messages = example["messages"]
      full_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
      )
      n_tokens = len(tokenizer.encode(full_str))
      lengths.append(n_tokens)

      if n_tokens > max_tokens:
        over_limit.append((i, n_tokens))

    print(f"\n--- {split_name} ({len(data)} examples) ---")
    print(f"  Token range: {min(lengths)}-{max(lengths)}, median: {sorted(lengths)[len(lengths)//2]}")
    print(f"  Exceeding {max_tokens}: {len(over_limit)}/{len(data)}")
    for idx, n in over_limit:
      print(f"    example {idx}: {n} tokens (+{n - max_tokens} over)")

def score_events(dirs):
  from event_scorer import EventScorer

  divided_md_files = list(dirs["divided_markdown"].glob("*.json"))
  raw_labels_files = list(dirs["raw_labels"].glob("*.txt"))
  
  with EventScorer(
    divided_md_files=divided_md_files,
    output_dir=dirs["scored_labels"],
    label_files=raw_labels_files,
  ) as scorer:
    scorer.run_scoring()

def build_dataset(dirs, test_ratio=0.15, empty_ratio=1.0, k_folds=5, seed=42):
  from dataset_builder import DatasetBuilder
  
  with DatasetBuilder(
    input_dir=dirs["scored_labels"],
    output_dir=dirs["dataset"]
  ) as builder:
    builder.build_biomistral_chunk_dataset(
      test_ratio=test_ratio,
      empty_ratio=empty_ratio,
      k_folds=k_folds,
      seed=seed
    )

def analyze_eval_results(paths):
  from pred_evaluator import PredEvaluator

  with PredEvaluator(
    paths["eval_preds"], 
    output_path=paths["eval_analysis"], 
    full_eval_analysis_folder_path=paths["full_eval_analysis"], 
    split_info_path=paths["split_info"]
  ) as evaluator:
    evaluator.analyze_eval_jsonl()
    evaluator.analyze_eval_jsonl_per_paper()


if __name__ == "__main__":
  # Parse command line arguments
  import argparse
  parser = argparse.ArgumentParser(description="AOP Events Extractor (Dev Tools)")
  parser.add_argument("--do-scoring", action="store_true",
                      help="Run event scoring")
  parser.add_argument("--do-dataset", action="store_true",
                      help="Build dataset for model training")
  parser.add_argument("--do-eval-analysis", action="store_true",
                      help="Analyze evaluation results")
  parser.add_argument("--do-token-check", action="store_true",
                      help="Analyze token lengths in train/test datasets")
  
  args = parser.parse_args()

  # Get config values from config.ini
  import configparser
  from common import PROMPT_INSTRUCTIONS
  config = configparser.ConfigParser()
  config.read("config.ini")
  model = config.get("MODEL", "model")
  model_weights = config.get("MODEL", "model_weights")
  model_context_tokens = config.getint("MODEL", "model_context_tokens")
  min_chunk_tokens = config.getint("MODEL", "min_chunk_tokens")
  target_chunk_tokens = config.getint("MODEL", "target_chunk_tokens")
  max_chunk_tokens = config.getint("MODEL", "max_chunk_tokens")
  reserved_output_tokens = config.getint("MODEL", "reserved_output_tokens")
  safety_margin_tokens = config.getint("MODEL", "safety_margin_tokens")
  dataset_test_ratio = config.getfloat("DATASET", "test_ratio")
  dataset_empty_ratio = config.getfloat("DATASET", "empty_ratio")
  dataset_k_folds = config.getint("DATASET", "k_folds")
  dataset_seed = config.getint("DATASET", "seed")


  paths = {
    "input_pdf": Path(config.get("DIRECTORIES", "input_pdf_dir")),
    "raw_markdown": Path(config.get("DIRECTORIES", "raw_markdown_dir")),
    "clean_markdown": Path(config.get("DIRECTORIES", "clean_markdown_dir")),
    "divided_markdown": Path(config.get("DIRECTORIES", "divided_markdown_dir")),
    "extracted_events": Path(config.get("DIRECTORIES", "extracted_events_dir")),
    "scored_events": Path(config.get("DIRECTORIES", "scored_events_dir")),
    "raw_labels": Path(config.get("DIRECTORIES", "raw_labels_dir")),
    "scored_labels": Path(config.get("DIRECTORIES", "scored_labels_dir")),
    "dataset": Path(config.get("DIRECTORIES", "dataset_dir")),
    "split_info": Path(config.get("DIRECTORIES", "split_info_path")),
    "eval_preds": Path(config.get("DIRECTORIES", "eval_preds_path")),
    "eval_analysis": Path(config.get("DIRECTORIES", "eval_analysis_path")),
    "full_eval_analysis": Path(config.get("DIRECTORIES", "full_eval_analysis_dir")),
    "train_file": Path(config.get("DIRECTORIES", "train_file_path")),
    "test_file": Path(config.get("DIRECTORIES", "test_file_path"))
  }

  if not (args.do_scoring or args.do_dataset or args.do_eval_analysis or args.do_token_check):
    print("No action specified. Use --do-scoring, --do-dataset, --do-eval-analysis, and/or --do-token-check.")

  if args.do_scoring:
    score_events(paths)

  if args.do_dataset:
    build_dataset(
      paths, 
      test_ratio=dataset_test_ratio, 
      empty_ratio=dataset_empty_ratio, 
      k_folds=dataset_k_folds, 
      seed=dataset_seed
    )

  if args.do_eval_analysis:
    analyze_eval_results(paths)
  
  if args.do_token_check:
    check_token_lengths(
      paths,
      max_tokens=model_context_tokens - reserved_output_tokens - safety_margin_tokens,
      tokenizer=AutoTokenizer.from_pretrained(model_weights)
    )
