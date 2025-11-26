from pathlib import Path
from MarkdownCleaner import MarkdownCleaner
from MarkdownDivider import MarkdownDivider
from PDFExtractor import PDFExtractor

PDF_DIR = Path("test_data/raw/")
LABELS_DIR = Path("test_data/labels/")
OUTPUT_DIR = Path("test_data/processed/")

if __name__ == "__main__":
  # Estrai testo e immagini dai PDF
  paper_files = list(PDF_DIR.glob("*.pdf"))
  md_folder = "markdown"
  with PDFExtractor(paper_files, OUTPUT_DIR, skip_existing=True) as extractor:
    extractor.run_text_extraction(folder=md_folder)
    extractor.run_image_extraction()
  
  # Pulisci markdown estratto
  md_files = list((OUTPUT_DIR / md_folder).glob("*.md"))
  md_cleaned_folder = "cleaned_markdown"
  with MarkdownCleaner(md_files, OUTPUT_DIR, skip_existing=True) as cleaner:
    cleaner.clean_markdowns(folder=md_cleaned_folder)

  # Dividi markdown
  md_cleaned_files = list((OUTPUT_DIR / md_cleaned_folder).glob("*.md"))
  with MarkdownDivider(md_cleaned_files) as divider:
    divided_markdowns = divider.divide_files()
  
  # Save divided markdowns to JSON output
  import json
  output_json_path = OUTPUT_DIR / "divided_markdowns.json"
  with output_json_path.open("w", encoding="utf-8") as json_file:
    json.dump(divided_markdowns, json_file, indent=2, ensure_ascii=False)

  # Processa markdown diviso con NER e BioMistral

  pass

