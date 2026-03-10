'''
Usage: python main.py input_pdf_folder
The p
'''


import json
import logging
from pathlib import Path
from MarkdownCleaner import MarkdownCleaner
from MarkdownDivider import MarkdownDivider
from BioNERExtractor import BioNERExtractor

VERBOSE = False
PDF_DIR = Path("test_data/raw/")
LABELS_DIR = Path("test_data/labels/")
OUTPUT_DIR = Path("test_data/processed/")

TEST = True

def test():
  pdf_files = list(PDF_DIR.glob("*.pdf"))
  md_folder = "markdown"
  with PDFExtractor(pdf_files, OUTPUT_DIR, skip_existing=True, keep_divided_pdfs=False) as extractor:
    extractor.run_text_extraction(folder=md_folder)


if __name__ == "__main__":
  print("#####################")
  print("#   AOP Extractor   #")
  print("#####################")

  print("\nStarting...\n\n")

  if not VERBOSE:
    logging.getLogger("docling").setLevel(logging.ERROR)
    logging.getLogger("docling_core").setLevel(logging.ERROR)
    logging.getLogger("fitz").setLevel(logging.ERROR)
    logging.getLogger("pymupdf").setLevel(logging.ERROR)

  from pdf_extractor import PDFExtractor

  if TEST:
    test()
    exit(0)

  # Estrai testo e immagini dai PDF
  pdf_files = list(PDF_DIR.glob("*.pdf"))
  md_folder = "markdown"
  images_folder = "images"
  with PDFExtractor(pdf_files, OUTPUT_DIR, skip_existing=True, keep_divided_pdfs=True, divided_folder=PDF_DIR) as extractor:
    extractor.run_text_extraction(folder=md_folder)
    # Extract tables
  
  # Pulisci markdown estratto
  md_files = list((OUTPUT_DIR / md_folder).glob("*.md"))
  md_cleaned_folder = "cleaned_markdown"
  with MarkdownCleaner(md_files, OUTPUT_DIR, skip_existing=True) as cleaner:
    cleaner.clean_markdowns(folder=md_cleaned_folder)

  # Dividi markdown
  md_cleaned_files = list((OUTPUT_DIR / md_cleaned_folder).glob("*.md"))
  divided_md_folder = "divided_markdown"
  with MarkdownDivider(md_cleaned_files, OUTPUT_DIR, skip_existing=True) as divider:
    divided_markdowns = divider.divide_files(folder=divided_md_folder)
  
  # Processa markdown diviso con NER e BioMistral
  json_divided_files = list((OUTPUT_DIR / divided_md_folder).glob("*.json"))
  

  pass

