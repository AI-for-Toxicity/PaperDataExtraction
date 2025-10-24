from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from collections import Counter
from pathlib import Path
import fitz, re

PDF_DIR = Path("data/raw/pdfs/")
OUTPUT_DIR = Path("data/raw/txts/")

def pages_to_clean_text(pages):
    # Remove repetitive headers/footers
    first_lines = [p[0].strip() for p in pages if len(p) > 0]
    last_lines = [p[-1].strip() for p in pages if len(p) > 0]
    common_heads = {line for line, c in Counter(first_lines).items() if c > len(pages)//2}
    common_foots = {line for line, c in Counter(last_lines).items() if c > len(pages)//2}

    clean_text = []
    for lines in pages:
        lines = [l for l in lines if l not in common_heads and l not in common_foots]
        clean_text.append("\n".join(lines))

    out = "\n\n".join(clean_text)

    # Remove page numbers and section numbers
    out = re.sub(r'\n?\s*\d+\s*\n', '\n\n', out)

    # Normalize whitespace and broken paragraphs
    #out = re.sub(r'(?<!\n)\n(?!\n)', ' ', out)  # join single newlines into spaces
    #out = re.sub(r'\n{2,}', '\n\n', out)        # keep paragraph breaks

    # Remove "Page X of Y" patterns
    out = re.sub(r'Page\s*\d+\s*of\s*\d+', '', out, flags=re.I)

    return out

def extract_text_fitz(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    text = re.sub(r'\s+\n', '\n', text)
    
    return text.strip()

def extract_text_pdfminer(pdf_path):
    pages = []
    for page_layout in extract_pages(pdf_path):
        lines = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text().strip()
                lines.extend(text.splitlines())
        pages.append(lines)
    
    return pages_to_clean_text(pages)

paper_files = list(PDF_DIR.glob("*.pdf"))
i = 0
for pdf in paper_files:
    i += 1
    print(f"[{i}/{len(paper_files)}] Processing {pdf.name}...")
    
    name = pdf.stem.split("_")
    name = name[0] + "_" + name[1]
    
    output_path = OUTPUT_DIR / f"{name}/"
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract text using pdfminer and save to file
    pdfminer_text = extract_text_pdfminer(pdf)
    pdfminer_text_path = output_path / f"{name}_pdfminer.txt"
    with pdfminer_text_path.open("w", encoding="utf-8") as out:
        out.write(pdfminer_text)
    
    # Extract text using fitz (PyMuPDF) and save to file
    fitz_text = extract_text_fitz(pdf)
    fitz_text_path = output_path / f"{name}_fitz.txt"
    with fitz_text_path.open("w", encoding="utf-8") as out:
        out.write(fitz_text)

    # Check if the extracted text is likely from an image-based PDF
    if len(pdfminer_text.strip()) < 100:
        # Run OCR extraction (placeholder for actual OCR code)
        continue

print("Text extraction completed.")
