import logging
from pathlib import Path


def pipeline(dirs, skip_existing, model, model_weights, tokenizer, run_only=None):
  # Subdirectory names
  pdf_folder              = dirs["input_pdf"]
  md_folder               = dirs["raw_markdown"]
  clean_md_folder         = dirs["clean_markdown"]
  divided_md_folder       = dirs["divided_markdown"]
  extracted_events_folder = dirs["extracted_events"]
  scored_events_folder    = dirs["scored_events"]

  # 1. Extract text from pdfs and save as markdown files
  pdf_files = list(pdf_folder.glob("*.pdf"))
  if run_only is None or run_only == 1:
    print("\nLoading PDF Extractor module...")
    from pdf_extractor import PDFExtractor
    with PDFExtractor(
      pdf_files,
      md_folder,
      skip_existing=skip_existing,
      keep_divided_pdfs=False
    ) as extractor:
      extractor.run_text_extraction(only_tables=True)

  # 2. Clean markdown files and save cleaned versions
  md_files = list(md_folder.glob("*.md"))
  if run_only is None or run_only == 2:
    print("\nLoading Markdown Cleaner module...")
    from md_cleaner import MarkdownCleaner
    with MarkdownCleaner(
      md_files,
      clean_md_folder,
      skip_existing=skip_existing
    ) as cleaner:
      cleaner.clean_markdowns()

  # 3. Divide cleaned versions of markdown files and save as json files
  clean_md_files = list(clean_md_folder.glob("*.md"))
  if run_only is None or run_only == 3:
    print("\nLoading Markdown Divider module...")
    from md_divider import MarkdownDivider
    with MarkdownDivider(
      clean_md_files,
      divided_md_folder,
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
      divider.divide_files()

  # 4. Process divided markdown with the model
  divided_md_files = list(divided_md_folder.glob("*.json"))
  if run_only is None or run_only == 4:
    print("\nLoading Event Extractor module...")
    from event_extractor import EventExtractor
    with EventExtractor(
      divided_md_files,
      extracted_events_folder,
      skip_existing=skip_existing,
      model=model,
      model_weights=model_weights,
    ) as extractor:
      extractor.extract_events()

  # 5. Score events predicted by the model
  extracted_files = list(extracted_events_folder.glob("*_extracted.json"))
  if run_only is None or run_only == 5:
    print("\nLoading Event Scorer module...")
    from event_scorer import EventScorer
    with EventScorer(
      divided_md_files,
      scored_events_folder,
    ) as scorer:
      scorer.run_scoring_from_extracted(extracted_files=extracted_files)

  # 6. Display results
  scored_events_files = list(scored_events_folder.glob("*_events.json"))
  if run_only is None or run_only == 6:
    print("\nLoading Display Results module...")
    from display_results import ResultsApp
    with ResultsApp(md_files=clean_md_files, events_files=scored_events_files) as app:
      app.run()


if __name__ == "__main__":
  # Parse command line arguments
  import argparse
  parser = argparse.ArgumentParser(description="AOP Events Extractor")
  parser.add_argument("--skip_existing", type=bool, default=True,
                      help="Skip processing steps that have already been completed")
  parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
  parser.add_argument("--run-only", type=int, choices=range(1, 7), metavar="N",
                      dest="run_only", default=None,
                      help="Run only pipeline step N (1-6)")
  parser.add_argument("--display-results", action="store_true", dest="display_results",
                      help="Alias for --run-only=6")

  args = parser.parse_args()

  run_only = args.run_only
  if args.display_results:
    run_only = 6

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

  directories = {
    "input_pdf": Path(config.get("DIRECTORIES", "input_pdf_dir")),
    "raw_markdown": Path(config.get("DIRECTORIES", "raw_markdown_dir")),
    "clean_markdown": Path(config.get("DIRECTORIES", "clean_markdown_dir")),
    "divided_markdown": Path(config.get("DIRECTORIES", "divided_markdown_dir")),
    "extracted_events": Path(config.get("DIRECTORIES", "extracted_events_dir")),
    "scored_events": Path(config.get("DIRECTORIES", "scored_events_dir")),
  }

  # Initialize tokenizer and calculate reserved prompt tokens
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
  reserved_prompt_tokens = len(tokenizer(PROMPT_INSTRUCTIONS, add_special_tokens=False)["input_ids"])

  if not args.verbose:
    logging.getLogger("docling").setLevel(logging.ERROR)
    logging.getLogger("docling_core").setLevel(logging.ERROR)
    logging.getLogger("fitz").setLevel(logging.ERROR)
    logging.getLogger("pymupdf").setLevel(logging.ERROR)

  pipeline(
    dirs=directories,
    skip_existing=args.skip_existing,
    run_only=run_only,
    model=model,
    model_weights=model_weights,
    tokenizer=tokenizer
  )
