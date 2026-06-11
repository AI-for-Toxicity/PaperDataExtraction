'''
Usage: python main.py input_pdf_folder
The p
'''

import logging
from pathlib import Path

def pipeline(input_dir, output_base_dir, skip_existing):
  preprocessing_output_dir = Path(output_base_dir) / "processed"
  preprocessing_output_dir.mkdir(parents=True, exist_ok=True)

  results_output_dir = Path(output_base_dir) / "results"
  results_output_dir.mkdir(parents=True, exist_ok=True)

  # Extract text from pdfs and save as markdown files
  print("\nLoading PDF Extractor module...")
  from pdf_extractor import PDFExtractor
  pdf_files = list(input_dir.glob("*.pdf"))
  md_folder = "markdown"
  with PDFExtractor(
    pdf_files, 
    preprocessing_output_dir, 
    skip_existing=skip_existing, 
    keep_divided_pdfs=False
  ) as extractor:
    extractor.run_text_extraction(folder=md_folder, only_tables=True)

  # Clean markdown files and save cleaned versions
  print("\nLoading Markdown Cleaner module...")
  from md_cleaner import MarkdownCleaner
  md_files = list((preprocessing_output_dir / md_folder).glob("*.md"))
  clean_md_folder = "cleaned_markdown"
  with MarkdownCleaner(
    md_files,
    preprocessing_output_dir, 
    skip_existing=skip_existing
  ) as cleaner:
    cleaner.clean_markdowns(folder=clean_md_folder)

  # Divide cleaned versions of markdown files and save as json files
  print("\nLoading Markdown Divider module...")
  from md_divider import MarkdownDivider
  md_cleaned_files = list((preprocessing_output_dir / clean_md_folder).glob("*.md"))
  divided_md_folder = "divided_markdown"
  with MarkdownDivider(
    md_cleaned_files,
    preprocessing_output_dir,
    skip_existing=skip_existing,
    min_chunk_tokens=min_chunk_tokens,
    target_chunk_tokens=target_chunk_tokens,
    max_chunk_tokens=max_chunk_tokens,
    model_context_tokens=model_context_tokens,
    reserved_prompt_tokens=reserved_prompt_tokens,
    reserved_output_tokens=reserved_output_tokens,
    safety_margin_tokens=safety_margin_tokens,
    tokenizer=tokenizer
  ) as divider:
    divider.divide_files(folder=divided_md_folder)
  
  # Process divided markdown with the model
  print("\nLoading Event Extractor module...")
  from event_extractor import EventExtractor
  divided_md_files = list((preprocessing_output_dir / divided_md_folder).glob("*.json"))
  extracted_events_folder = "extracted_events"
  with EventExtractor(
    divided_md_files,
    results_output_dir,
    skip_existing=skip_existing,
    model=model,
    model_weights=model_weights,
  ) as extractor:
    extractor.extract_events(folder=extracted_events_folder)
  
  # Score events predicted by the model
  print("\nLoading Event Scorer module...")
  from event_scorer import EventScorer
  scored_events_folder = "scored_events"
  with EventScorer(
    json_files_dir=preprocessing_output_dir / divided_md_folder,
    output_dir=results_output_dir / scored_events_folder,
  ) as scorer:
    scorer.run_scoring_from_extracted(
      extracted_dir=results_output_dir / extracted_events_folder,
      divided_md_dir=preprocessing_output_dir / divided_md_folder,
    )

  exit(0)
  
  # Display results
  print("\nLoading Display Results module...")
  from display_results import ResultsApp
  clean_md_folder = PREPROCESSING_OUTPUT_DIR / md_cleaned_folder
  scored_events_folder = SCORED_EVENTS_DIR
  with ResultsApp(md_folder=clean_md_folder, events_folder=scored_events_folder) as app:
    app.run()


if __name__ == "__main__":
  # Get command line args and set up logging
  import argparse
  parser = argparse.ArgumentParser(description="AOP Events Extractor")
  parser.add_argument("--input_dir", type=str, default=str("data/raw"), help="Directory containing input PDF files")
  parser.add_argument("--output_base_dir", type=str, default=str("data"), help="Base directory for output files")
  parser.add_argument("--skip_existing", type=bool, default=True, help="Skip processing steps that have already been completed")
  parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

  # Get config values from config.ini
  import configparser
  from model.common import PROMPT_INSTRUCTIONS
  config = configparser.ConfigParser()
  config.read("config.ini")
  model = config.get("DEFAULT", "model")
  model_weights = config.get("DEFAULT", "model_weights")
  model_context_tokens = config.getint("DEFAULT", "model_context_tokens")
  min_chunk_tokens = config.getint("DEFAULT", "min_chunk_tokens")
  target_chunk_tokens = config.getint("DEFAULT", "target_chunk_tokens")
  max_chunk_tokens = config.getint("DEFAULT", "max_chunk_tokens")
  reserved_output_tokens = config.getint("DEFAULT", "reserved_output_tokens")
  safety_margin_tokens = config.getint("DEFAULT", "safety_margin_tokens")

  # Initialize tokenizer and calculate reserved prompt tokens
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
  reserved_prompt_tokens = len(tokenizer(PROMPT_INSTRUCTIONS, add_special_tokens=False)["input_ids"])

  # Setup logging level based on verbose flag
  args = parser.parse_args()
  if not args.verbose:
    logging.getLogger("docling").setLevel(logging.ERROR)
    logging.getLogger("docling_core").setLevel(logging.ERROR)
    logging.getLogger("fitz").setLevel(logging.ERROR)
    logging.getLogger("pymupdf").setLevel(logging.ERROR)

  pipeline(
    input_dir=Path(args.input_dir),
    output_base_dir=Path(args.output_base_dir),
    skip_existing=args.skip_existing
  )
