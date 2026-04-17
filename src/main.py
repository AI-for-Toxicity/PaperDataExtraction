'''
Usage: python main.py input_pdf_folder
The p
'''

import json
import logging
from pathlib import Path
from transformers import AutoTokenizer

from display_results import ResultsApp

VERBOSE = False
DATA_BASE_DIR = Path("test_data") # command line arg
PDF_DIR = DATA_BASE_DIR / "raw"
PREPROCESSING_OUTPUT_DIR = DATA_BASE_DIR / "processed"
LABELS_DIR = DATA_BASE_DIR / "labels/aop_raw" # maybe restructure to fit rest of pipeline
SCORED_EVENTS_DIR = DATA_BASE_DIR / "labels/scored" # maybe restructure to fit rest of pipeline
DIVIDED_MD_DIR = PREPROCESSING_OUTPUT_DIR / "divided_markdown" # to remove, get from PREPROCESSING_OUTPUT_DIR / "divided_markdown"
TRAIN_TEST_BASE_DIR = Path("train") # command line arg
FINAL_JSON_TRAIN = TRAIN_TEST_BASE_DIR / "train.jsonl"
FINAL_JSON_TEST = TRAIN_TEST_BASE_DIR / "test.jsonl"
EVAL_RESULTS_BASE_DIR = Path("train/results_5/mistral_7b_instruct_epoch_5") # command line arg
EVAL_RESULTS_JSONL = EVAL_RESULTS_BASE_DIR / "eval_preds.jsonl"
EVAL_ANALYSIS_RESULT = EVAL_RESULTS_BASE_DIR / "eval_analysis.txt"
RAG_INDEX_PATH = "rag/aop_rag_index.json"

TOKENIZER = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B", use_fast=True)
MIN_CHUNK_TOKENS = 700
TARGET_CHUNK_TOKENS = 1100
MAX__CHUNK_TOKENS = 1400
BIOMISTRAL_CONTEXT_TOKENS = 2048
from event_scorer_dataset import PROMPT_INSTRUCTIONS
RESERVED_PROMPT_TOKENS = len(TOKENIZER(PROMPT_INSTRUCTIONS, add_special_tokens=False)["input_ids"]) # 113
RESERVED_OUTPUT_TOKENS = 400
SAFETY_MARGIN_TOKENS = 96


def check_token_lengths(
    train_path: str,
    test_path: str,
    max_tokens: int = BIOMISTRAL_CONTEXT_TOKENS,
    tokenizer=TOKENIZER,
) -> None:
    """
    Reads train/test JSONL files, tokenizes each example using the chat template,
    and reports any entries exceeding max_tokens.
    """
    for split_name, path in [("train", train_path), ("test", test_path)]:
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

def test(do_scoring=False, do_dataset=False, do_eval_analysis=True):
  from event_scorer_dataset import PredEvaluator, EventScorer, DatasetBuilder
  if do_scoring:
        with EventScorer(json_files_dir=DIVIDED_MD_DIR, labels_dir=LABELS_DIR, output_dir=SCORED_EVENTS_DIR) as scorer:
            scorer.run_scoring()

  if do_dataset:
      with DatasetBuilder(input_dir=SCORED_EVENTS_DIR, output_train_path=FINAL_JSON_TRAIN, output_test_path=FINAL_JSON_TEST, rag_index_path=RAG_INDEX_PATH) as builder:
          builder.build_biomistral_chunk_dataset(
              test_ratio=0.15,
              empty_ratio=1.0,   # Keep all empty chunks as negatives
              seed=42,
              use_rag=False
          )
      
  if do_eval_analysis:
      with PredEvaluator(EVAL_RESULTS_JSONL, output_path=EVAL_ANALYSIS_RESULT) as evaluator:
            evaluator.analyze_eval_jsonl()


if __name__ == "__main__":
  #check_token_lengths(str(FINAL_JSON_TRAIN), str(FINAL_JSON_TEST), max_tokens=BIOMISTRAL_CONTEXT_TOKENS)
  test()
  exit(0)

  
  print("#####################")
  print("#   AOP Extractor   #")
  print("#####################")

  if not VERBOSE:
    logging.getLogger("docling").setLevel(logging.ERROR)
    logging.getLogger("docling_core").setLevel(logging.ERROR)
    logging.getLogger("fitz").setLevel(logging.ERROR)
    logging.getLogger("pymupdf").setLevel(logging.ERROR)

  # Extract text from pdfs and save as markdown files
  print("\nLoading PDF Extractor module...")
  from pdf_extractor import PDFExtractor
  pdf_files = list(PDF_DIR.glob("*.pdf"))
  md_folder = "markdown"
  images_folder = "images"
  with PDFExtractor(pdf_files, PREPROCESSING_OUTPUT_DIR, skip_existing=True, keep_divided_pdfs=False) as extractor:
    extractor.run_text_extraction(folder=md_folder)

  # Clean markdown files and save cleaned versions
  print("\nLoading Markdown Cleaner module...")
  from md_cleaner import MarkdownCleaner
  md_files = list((PREPROCESSING_OUTPUT_DIR / md_folder).glob("*.md"))
  md_cleaned_folder = "cleaned_markdown"
  with MarkdownCleaner(md_files, PREPROCESSING_OUTPUT_DIR, skip_existing=True) as cleaner:
    cleaner.clean_markdowns(folder=md_cleaned_folder)

  # Divide cleaned versions of markdown files and save as json files
  print("\nLoading Markdown Divider module...")
  from md_divider import MarkdownDivider
  md_cleaned_files = list((PREPROCESSING_OUTPUT_DIR / md_cleaned_folder).glob("*.md"))
  divided_md_folder = "divided_markdown"
  with MarkdownDivider(
    md_cleaned_files,
    PREPROCESSING_OUTPUT_DIR,
    skip_existing=True,
    min_chunk_tokens=MIN_CHUNK_TOKENS,
    target_chunk_tokens=TARGET_CHUNK_TOKENS,
    max_chunk_tokens=MAX__CHUNK_TOKENS,
    model_context_tokens=BIOMISTRAL_CONTEXT_TOKENS,
    reserved_prompt_tokens=RESERVED_PROMPT_TOKENS,
    reserved_output_tokens=RESERVED_OUTPUT_TOKENS,
    safety_margin_tokens=SAFETY_MARGIN_TOKENS,
    tokenizer=TOKENIZER
  ) as divider:
    divided_markdowns = divider.divide_files(folder=divided_md_folder)
  
  # Processa markdown diviso con BioMistral
  print("\nLoading Event Extractor module...")
  pass

  # Score events
  print("\nLoading Event Scorer module...")
  from event_scorer_dataset import EventScorer
  with EventScorer(json_files_dir=DIVIDED_MD_DIR, labels_dir=LABELS_DIR, output_dir=SCORED_EVENTS_DIR) as scorer:
    scorer.run_scoring()

  exit(0)
  
  # Display results
  print("\nLoading Display Results module...")
  from display_results import ResultsApp
  clean_md_folder = PREPROCESSING_OUTPUT_DIR / md_cleaned_folder
  scored_events_folder = SCORED_EVENTS_DIR
  with ResultsApp(md_folder=clean_md_folder, events_folder=scored_events_folder) as app:
    app.run()
  
