'''
Usage: python main.py input_pdf_folder
The p
'''


import json
import logging
from pathlib import Path

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
EVAL_RESULTS_BASE_DIR = Path("train/results_3") # command line arg
EVAL_RESULTS_JSONL = EVAL_RESULTS_BASE_DIR / "eval_preds.jsonl"
EVAL_ANALYSIS_RESULT = EVAL_RESULTS_BASE_DIR / "eval_analysis.txt"
RAG_INDEX_PATH = "rag/aop_rag_index.json"

def test(do_scoring=True, do_dataset=False, do_eval_analysis=False):
  from event_scorer_dataset import PredEvaluator, EventScorer, DatasetBuilder
  if do_scoring:
        with EventScorer(json_files_dir=DIVIDED_MD_DIR, labels_dir=LABELS_DIR, output_dir=SCORED_EVENTS_DIR) as scorer:
            scorer.run_scoring()

  if do_dataset:
      with DatasetBuilder(input_dir=SCORED_EVENTS_DIR, output_train_path=FINAL_JSON_TRAIN, output_test_path=FINAL_JSON_TEST, rag_index_path=RAG_INDEX_PATH) as builder:
          builder.build_biomistral_chunk_dataset(
              test_ratio=0.1,
              empty_ratio=1.0,   # 50% negatives (results_3)
              seed=42,
              use_rag=False
          )
      
  if do_eval_analysis:
      with PredEvaluator(EVAL_RESULTS_JSONL, output_path=EVAL_ANALYSIS_RESULT) as evaluator:
            evaluator.analyze_eval_jsonl()


if __name__ == "__main__":
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
  print("\nLoading PDF Extractor module...\n\n")
  from pdf_extractor import PDFExtractor
  pdf_files = list(PDF_DIR.glob("*.pdf"))
  md_folder = "markdown"
  images_folder = "images"
  with PDFExtractor(pdf_files, PREPROCESSING_OUTPUT_DIR, skip_existing=True, keep_divided_pdfs=False) as extractor:
    extractor.run_text_extraction(folder=md_folder)
  
  # Clean markdown files and save cleaned versions
  print("\nLoading Markdown Cleaner module...\n\n")
  from md_cleaner import MarkdownCleaner
  md_files = list((PREPROCESSING_OUTPUT_DIR / md_folder).glob("*.md"))
  md_cleaned_folder = "cleaned_markdown"
  with MarkdownCleaner(md_files, PREPROCESSING_OUTPUT_DIR, skip_existing=True) as cleaner:
    cleaner.clean_markdowns(folder=md_cleaned_folder)

  # Divide cleaned versions of markdown files and save as json files
  print("\nLoading Markdown Divider module...\n\n")
  from md_divider import MarkdownDivider
  md_cleaned_files = list((PREPROCESSING_OUTPUT_DIR / md_cleaned_folder).glob("*.md"))
  divided_md_folder = "divided_markdown"
  with MarkdownDivider(md_cleaned_files, PREPROCESSING_OUTPUT_DIR, skip_existing=True) as divider:
    divided_markdowns = divider.divide_files(folder=divided_md_folder)
  
  # Processa markdown diviso con NER e BioMistral
  json_divided_files = list((PREPROCESSING_OUTPUT_DIR / divided_md_folder).glob("*.json"))
  
  print("Pipeline completed.")
  print("#####################")
