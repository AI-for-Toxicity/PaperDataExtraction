#!/usr/bin/env python3
"""
extract_paper_images.py

For each PDF in INPUT matching "paper_XXXX*.pdf" or "paper_XXXX.pdf":
 - create OUTPUT/paper_XXXX/
 - delete any .png files in that folder whose filename contains "_picture_"
 - extract every image from the pdf with dimensions >= min_size x min_size
 - save them as OUTPUT/paper_XXXX/paper_XXXX_picture_{Y}.png (Y starts at 1)
"""

import re
import argparse
from pathlib import Path
import sys
import fitz  # PyMuPDF
from PIL import Image
import io

PDF_ID_RE = re.compile(r"paper_(\d+)", re.IGNORECASE)

def delete_old_pictures(out_dir: Path):
    if not out_dir.exists():
        return
    for p in out_dir.glob("*_picture_*.png"):
        try:
            p.unlink()
        except Exception as e:
            print(f"  [!] Failed to delete {p}: {e}")

def save_pixmap_as_png(pix, out_path: Path):
    """
    Save a fitz.Pixmap to out_path as PNG. Handles CMYK and alpha.
    """
    if pix.n < 5:
        # n < 5: grayscale (1), rgb (3), or with alpha (4)
        pix.save(str(out_path))
    else:
        # CMYK or other: convert to RGB first
        pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
        pix_rgb.save(str(out_path))
        pix_rgb = None
    pix = None

def extract_images_from_pdf(pdf_path: Path, out_dir: Path, paper_id: str, min_size: int):
    doc = fitz.open(pdf_path)
    counter = 1
    seen_xrefs = set()  # avoid saving the same XObject twice on different pages
    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)
        if not images:
            continue
        for img_info in images:
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                pix = fitz.Pixmap(doc, xref)
            except Exception as e:
                print(f"  [!] Could not make pixmap for xref {xref} on {pdf_path.name}: {e}")
                continue

            width = pix.width
            height = pix.height

            if width < min_size or height < min_size:
                # discard small images
                pix = None
                continue

            out_name = f"{paper_id}_picture_{counter}.png"
            out_path = out_dir / out_name

            try:
                # Use PIL if conversion to PNG via bytes is preferred,
                # but fitz.Pixmap.save also writes PNG reliably.
                save_pixmap_as_png(pix, out_path)
            except Exception as e:
                # fallback via PIL (less direct, but sometimes helps)
                try:
                    im_bytes = pix.tobytes(output="png")
                    im = Image.open(io.BytesIO(im_bytes))
                    im.save(out_path, format="PNG")
                except Exception as e2:
                    print(f"  [!] Failed to save image xref {xref}: {e} / {e2}")
                    continue
            finally:
                pix = None

            print(f"    saved {out_name} ({width}x{height})")
            counter += 1

    doc.close()
    return counter - 1  # number of saved images

def main(args):
    input_dir = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    min_size = args.min_size

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] input folder not found: {input_dir}")
        sys.exit(1)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print("[INFO] No PDF files found in input folder.")
        return

    for pdf in pdf_files:
        m = PDF_ID_RE.search(pdf.stem)
        if not m:
            print(f"[skip] {pdf.name} does not match 'paper_XXXX' pattern.")
            continue
        paper_id = f"paper_{m.group(1)}"
        out_dir = output_root / paper_id
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {pdf.name} -> {out_dir}/")
        delete_old_pictures(out_dir)
        saved = extract_images_from_pdf(pdf, out_dir, paper_id, min_size)
        if saved == 0:
            print(f"  (no images >= {min_size}x{min_size} found)")
        else:
            print(f"  done: {saved} images saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract large images from PDFs into per-paper folders.")
    parser.add_argument("--input", "-i", default="data/raw/pdfs", help="INPUT folder with PDFs")
    parser.add_argument("--output", "-o", default="data/raw/markdown", help="OUTPUT root folder")
    parser.add_argument("--min-size", "-m", type=int, default=100, help="Minimum width/height in pixels (default 100)")
    args = parser.parse_args()
    main(args)
