'''
This module implements a class that takes a list of PDF files and an output directory, and extracts text from the PDFs into markdown files.
The extraction is ran with the run_text_extraction method.
'''

import re
import collections, io, tempfile, fitz
from pathlib import Path
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
from PIL import Image, ImageOps
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

class PDFExtractor:
  # INITIALIZATION

  def __init__(self, paper_files: list, output_dir: Path, skip_existing: bool = False, keep_divided_pdfs: bool = False, divided_folder: Path | None = None) -> None:
    print("### PDFExtractor - init ###")
    self.pipeline_options = PdfPipelineOptions(
      do_table_structure=False,
      do_ocr=False,
      generate_table_images=False,
      generate_picture_images=False,
    )
    self.docling_converter = DocumentConverter(
      format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
      }
    )
    self.keep_divided_pdfs = keep_divided_pdfs
    self.divided_folder = divided_folder
    self.tmpdir = tempfile.TemporaryDirectory()
    if self.keep_divided_pdfs and self.divided_folder is not None:
      self.tmpdir = tempfile.TemporaryDirectory(delete=False, dir=self.divided_folder)
    self.paper_files = paper_files
    self.output_dir = output_dir
    self.skip_existing = skip_existing
    return
  
  def __enter__(self):
    return self

  # MAIN TEXT EXTRACTION

  '''
  Yield (page_idx, span_dict, page_height)
  '''
  def extract_spans(self, doc):
    for p_idx, page in enumerate(doc):
      page_dict = page.get_text("dict")
      ph = page.rect.height
      for block in page_dict.get("blocks", []):
        if block["type"] != 0:
          continue
        for line in block.get("lines", []):
          for span in line.get("spans", []):
            text = span.get("text", "").strip()
            if not text:
              continue
            yield (p_idx, span, line["bbox"], ph)

  '''
  Identify if a line (given by its bbox) is likely in the header or footer region of the page, based on a margin ratio.
  '''
  def is_in_header_or_footer_region(self, line_bbox, page_height, margin_ratio=0.10):
    y0, y1 = line_bbox[1], line_bbox[3]
    return (y0 <= page_height * margin_ratio) or (y1 >= page_height * (1 - margin_ratio))

  '''
  Compute dominant font size weighted by number of characters.
  Supports both span shapes:
  1) (page_idx, span_dict, line_bbox, page_height)
  2) (page_idx, text, size, y0, y1, page_height)
  '''
  def dominant_font_size_by_chars(self, spans, debug=False):
    char_counter = collections.Counter()

    for item in spans:
      # try to detect shape
      if len(item) == 4:
        # shape 1
        _, span, _, _ = item
        text = span.get("text", "")
        size = span.get("size", 0)
      elif len(item) == 6:
        # shape 2
        _, text, size, _, _, _ = item
      else:
        # unexpected, skip
        continue

      if not text:
        continue
      text = text.strip()
      if not text:
        continue

      try:
        size = float(size)
      except Exception:
        continue

      if size <= 0:
        continue

      # round lightly to group close sizes
      size_key = round(size, 1)
      char_counter[size_key] += len(text)

    if not char_counter:
      if debug:
        print("dominant_font_size_by_chars: no chars counted. spans sample:", spans[:5])
      return 0.0

    # pick size with the highest total char count
    dom_size, char_count = max(char_counter.items(), key=lambda x: x[1])

    if debug:
      print("font char counts:", char_counter)
      print("dominant:", dom_size, "chars:", char_count)

    return dom_size

  '''
  Choose the body font size based on:
  - how many pages a font size appears on (coverage)
  - how many characters it has in total (tie-breaker)

  spans must be like: (page_idx, span_dict, line_bbox, page_height)
  headers/footers are sets of strings to ignore.
  '''
  def pick_body_font_size(self, spans, headers=None, footers=None, debug=False):
    if headers is None:
      headers = set()
    if footers is None:
      footers = set()

    # per-size total chars
    size_total_chars = collections.Counter()
    # per-size page coverage
    size_pages = collections.defaultdict(set)

    for item in spans:
      if len(item) != 4:
        # if your extract_spans has a different shape, fix that upstream
        continue
      page_idx, span, line_bbox, ph = item
      text = span.get("text", "")
      if not text:
        continue
      text = text.strip()
      if not text:
        continue
      if text in headers or text in footers:
        continue

      size = span.get("size", 0)
      try:
        size = float(size)
      except Exception:
        continue
      if size <= 0:
        continue

      size = round(size, 1)

      chars = len(text)
      size_total_chars[size] += chars
      size_pages[size].add(page_idx)

    if not size_total_chars:
      return 0.0

    # score sizes: prefer ones that appear on more pages
    candidates = []
    for sz, total_chars in size_total_chars.items():
      coverage = len(size_pages[sz])
      candidates.append((coverage, total_chars, sz))

    # sort by coverage desc, then chars desc
    candidates.sort(reverse=True)
    coverage, total_chars, chosen_size = candidates[0]

    if debug:
      print("sizes by page coverage:")
      for cov, chars, sz in candidates:
        print(f"size {sz}: pages={cov}, chars={chars}")
      print("chosen:", chosen_size)

    return chosen_size

  '''
  Collect rectangles representing the body text for each page.
  '''
  def _collect_body_rects(self, spans, threshold, eps, padding_factor=1.5):
    body_rects_by_page = collections.defaultdict(list)
    lower = threshold - eps

    for (p_idx, span, line_bbox, ph) in spans:
        text = span.get("text", "").strip()
        if not text:
            continue

        try:
            size = float(span.get("size", 0.0))
        except Exception:
            continue

        if size < lower:
            continue

        bbox = fitz.Rect(span["bbox"])

        # allarga il rect un po' in verticale per beccare i superscript
        h = bbox.height
        pad = h * (padding_factor - 1.0) / 2.0
        bbox.y0 -= pad
        bbox.y1 += pad

        body_rects_by_page[p_idx].append(bbox)

    return body_rects_by_page

  '''
  Check if a bbox overlaps significantly with any of the body rects on the page.
  '''
  def _overlaps_body(self, bbox, body_rects, min_overlap_ratio=0.2):
    if not body_rects:
        return False

    area = bbox.get_area()
    if area <= 0:
        return False

    for br in body_rects:
        inter = bbox & br
        if inter.is_empty:
            continue
        # quanto del SMALL rect è coperto da un body rect
        if inter.get_area() / area >= min_overlap_ratio:
            return True

    return False
  
  '''
  Open original PDF, redact spans we DON'T want, keep the rest.
  This preserves layout and fonts.
  keep_big=True  -> keep size >= threshold, redact smaller
  keep_big=False -> keep size < threshold, redact bigger
  '''
  def build_filtered_pdf_redaction(self, src_doc, spans, threshold, keep_big=True, eps=0.5):
    doc = src_doc

    # 1) rettangoli di testo "body" per pagina
    body_rects_by_page = self._collect_body_rects(spans, threshold, eps)

    # 2) group spans by page
    spans_by_page = collections.defaultdict(list)
    for (p_idx, span, line_bbox, ph) in spans:
        spans_by_page[p_idx].append((span, line_bbox, ph))

    lower = threshold - eps

    # 3) Redaction
    for p_idx in range(len(doc)):
      page = doc[p_idx]
      page_spans = spans_by_page.get(p_idx, [])

      body_rects = body_rects_by_page.get(p_idx, [])

      for span, line_bbox, ph in page_spans:
        text = span["text"].strip()
        if not text:
          continue

        #if is_in_header_or_footer_region(line_bbox, ph):
        #    page.add_redact_annot(fitz.Rect(span["bbox"]), fill=(1, 1, 1))
        #    continue

        try:
          size = float(span.get("size", 0.0))
        except Exception:
          size = 0.0
        # if size is unknown, don't risk removing it
        if size <= 0:
          continue

        bbox = fitz.Rect(span["bbox"])

        # gray zone: sizes in [body_size - eps, body_size + eps] are treated as body
        is_big = size >= lower
        if keep_big:
          if is_big:
            continue
          # inline piccolo (¹, H₂O, stuff vicino alle parole): lascialo
          if self._overlaps_body(bbox, body_rects):
            continue
          page.add_redact_annot(bbox, fill=(1, 1, 1))
        else:
          if is_big or self._overlaps_body(bbox, body_rects):
            page.add_redact_annot(bbox, fill=(1, 1, 1))
          else:
            continue
                
      page.apply_redactions()

    return doc

  '''
  Run Docling conversion on a PDF and return the markdown string.
  '''
  def run_docling_on_pdf(self, path_str: str) -> str:
      res = self.docling_converter.convert(path_str)
      return res.document.export_to_markdown()

  # TABLE EXTRACTION

  '''
  Clean a cell value by normalizing whitespace and stripping.
  '''
  def _clean_cell(self, value) -> str:
    if value is None:
      return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text

  '''
  Heuristic:
    - header row should be mostly non-empty
    - usually short textual labels, not mostly numeric
  '''
  def _looks_like_header_row(self, row) -> bool:
    cleaned = [self._clean_cell(x) for x in row]
    non_empty = [x for x in cleaned if x]
    if len(non_empty) < max(1, len(cleaned) // 2):
      return False

    numeric_like = 0
    for cell in non_empty:
      if re.fullmatch(r"[%\d\.\,\-\+\(\)\/]+", cell):
        numeric_like += 1

    # if most cells are numeric, it's probably not a header
    return numeric_like < len(non_empty) / 2

  '''
  Normalize a header cell, using "column N" as fallback if it looks empty after cleaning.
  '''
  def _normalize_header(self, header: str, fallback_idx: int) -> str:
    header = self._clean_cell(header).lower()
    if not header:
      return f"column {fallback_idx}"
    return header

  '''
  Convert a table (list of rows) into a serialized string:
  Table N: header1 value1, header2 value2. header1 value1, header2 value2.
  '''
  def _serialize_table_rows(self, rows, table_label: str) -> str | None:
    if not rows:
      return None

    # clean all rows
    cleaned_rows = [
      [self._clean_cell(cell) for cell in row]
      for row in rows
    ]

    # remove completely empty rows
    cleaned_rows = [
      row for row in cleaned_rows
      if any(cell for cell in row)
    ]

    if not cleaned_rows:
      return None

    # choose header
    if len(cleaned_rows) >= 2 and self._looks_like_header_row(cleaned_rows[0]):
      headers = [
        self._normalize_header(h, idx + 1)
        for idx, h in enumerate(cleaned_rows[0])
      ]
      data_rows = cleaned_rows[1:]
    else:
      max_cols = max(len(r) for r in cleaned_rows)
      headers = [f"column {i+1}" for i in range(max_cols)]
      data_rows = cleaned_rows

    row_texts = []
    for row in data_rows:
      pieces = []
      for idx, cell in enumerate(row):
        cell = self._clean_cell(cell)
        if not cell:
          continue
        header = headers[idx] if idx < len(headers) else f"column {idx+1}"
        pieces.append(f"{header} {cell}")

      if pieces:
        row_texts.append(", ".join(pieces))

    if not row_texts:
      return None

    return f"{table_label}: " + ". ".join(row_texts) + "."

  '''
  Return a list of serialized table strings extracted from the PDF.
  Uses PyMuPDF's built-in table detection on each page.
  '''
  def extract_tables_from_pdf(self, doc, paper_id: str):
    extracted_tables = []
    global_table_idx = 1

    for page_index in range(len(doc)):
      page = doc[page_index]

      try:
        tabs = page.find_tables()
      except Exception as e:
        print(f"  [!] Could not detect tables on page {page_index+1} of {paper_id}: {e}")
        continue

      page_tables = getattr(tabs, "tables", [])
      if not page_tables:
        continue

      for table in page_tables:
        try:
          rows = table.extract()
        except Exception as e:
          print(f"  [!] Could not extract table {global_table_idx} on page {page_index+1} of {paper_id}: {e}")
          global_table_idx += 1
          continue

        label = f"Table {global_table_idx}"
        serialized = self._serialize_table_rows(rows, label)
        if serialized:
          extracted_tables.append(serialized)

        global_table_idx += 1

    return extracted_tables

  # IMAGE EXTRACTION

  def _normalize_ocr_text(self, text: str) -> str:
    """
    Normalize OCR text by fixing form feeds, collapsing multiple newlines, and normalizing spaces.
    """
    text = text.replace("\x0c", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

  def _prepare_image_for_ocr(self, img: Image.Image) -> Image.Image:
    """
    Preprocess image for OCR: grayscale, upscale small images, 
    adaptive contrast enhancement, and binarization.
    """
    img = ImageOps.exif_transpose(img)

    if img.mode not in ("L", "RGB", "RGBA"):
        img = img.convert("RGB")
    if img.mode == "RGBA":
        # Flatten alpha onto white background
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    if img.mode != "L":
        img = ImageOps.grayscale(img)

    # Upscale small images — Tesseract performs poorly below ~200 DPI equivalent
    w, h = img.size
    min_dim = 1000
    if w < min_dim or h < min_dim:
        scale = max(min_dim / w, min_dim / h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Adaptive histogram equalization for better contrast
    img_np = np.array(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_np = clahe.apply(img_np)

    # Otsu binarization — separates text from background cleanly
    _, img_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(img_np)

  def _ocr_image_preserve_blocks(self, img: Image.Image, lang: str = "eng") -> str:
    """
    OCR one figure image using paragraph-aware block grouping.
    Uses PSM 3 (auto page segmentation) instead of PSM 11 (sparse chars).

    Output format:
        block 1 paragraph text...

        block 2 paragraph text...
    """
    img = self._prepare_image_for_ocr(img)

    data = pytesseract.image_to_data(
        img,
        lang=lang,
        output_type=Output.DICT,
        config="--psm 3 --oem 1",  # PSM 3 = auto layout, OEM 1 = LSTM only
    )

    n = len(data["text"])
    # Group words by (block, paragraph, line) while filtering low-confidence noise
    grouped = collections.OrderedDict()

    for i in range(n):
        word = (data["text"][i] or "").strip()
        if not word:
            continue

        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1

        # Raise threshold significantly — real text in figures is usually high-confidence
        # -1 means Tesseract didn't score it (e.g. line-level entries), keep those
        if conf != -1 and conf < 40:
            continue

        # Skip tokens that are pure punctuation/symbols with no alphanumeric content
        if not re.search(r"[a-zA-Z0-9]", word):
            continue

        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        grouped.setdefault(key, []).append(word)

    if not grouped:
        return ""

    # Reconstruct: block -> paragraphs -> lines -> joined words
    # Structure: blocks[block_num][par_num] = [line_text, ...]
    blocks: dict = collections.OrderedDict()
    for (block_num, par_num, line_num), words in grouped.items():
        line_text = " ".join(words)
        blocks.setdefault(block_num, collections.OrderedDict()) \
              .setdefault(par_num, []) \
              .append(line_text)

    block_texts = []
    for block_num, paragraphs in blocks.items():
        para_texts = []
        for par_num, lines in paragraphs.items():
            # Join lines in a paragraph into a single flowing sentence
            para_text = " ".join(line.strip() for line in lines if line.strip())
            if para_text:
                para_texts.append(para_text)
        block_body = "\n".join(para_texts)
        block_body = self._normalize_ocr_text(block_body)
        if block_body:
            block_texts.append(block_body)

    return "\n\n".join(block_texts).strip()

  '''
  Extract embedded images from the PDF, OCR them, and return a list like:
  ["Figure 1: ...", "Figure 2: ..."]
  '''
  def extract_figure_texts_from_pdf(self, doc, paper_id: str, min_size: int = 80, ocr_lang: str = "eng"):
    figure_texts = []
    seen_xrefs = set()
    counter = 1

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
          base_image = doc.extract_image(xref)
        except Exception as e:
          print(f"  [!] Could not extract image xref {xref} from {paper_id}: {e}")
          continue

        img_bytes = base_image.get("image")
        if not img_bytes:
          continue

        try:
          img = Image.open(io.BytesIO(img_bytes))
        except Exception as e:
          print(f"  [!] Could not open image xref {xref} from {paper_id}: {e}")
          continue

        width, height = img.size
        if width < min_size or height < min_size:
          continue

        try:
          ocr_text = self._ocr_image_preserve_blocks(img, lang=ocr_lang)
        except Exception as e:
          print(f"  [!] OCR failed for image xref {xref} in {paper_id}: {e}")
          continue

        if not ocr_text:
          continue

        label = f"Figure {counter}"
        figure_texts.append(f"{label}:\n{ocr_text}")
        counter += 1

    return figure_texts

  # FULL EXTRACTION PIPELINE

  '''
  Iterates paper_files and extracts text into markdown files in output_dir/folder.
  For each PDF it does the following:
  1. Divide body text (font size >= threshold) from small text
  2. Extract text from tables, trasforming each row into a sentence
  3. Extract text from figures
  4. Finally combines body text, small text, table text, and figure text in an output markdown with sections.
  '''
  def run_text_extraction(self, folder="markdown", ocr_figure_lang: str = "eng"):
    total = len(self.paper_files)
    print(f"Found {total} PDFs to process for text extraction.\n")

    for i, pdf in enumerate(self.paper_files, 1):
      print(f"[{i}/{total}] Processing {pdf.name}...")
      parts = pdf.stem.split("_")
      name = "_".join(parts[:2]) if len(parts) > 1 else parts[0]
      out_dir = self.output_dir / folder
      out_dir.mkdir(parents=True, exist_ok=True)
      out_path = out_dir / f"{name}.md"

      if self.skip_existing and out_path.exists():
        print(f"Skipping {name} because {out_path} exists.")
        continue

      tmpdir_path = Path(self.tmpdir.name)
      big_pdf_path = tmpdir_path / f"{name}_big.pdf"
      small_pdf_path = tmpdir_path / f"{name}_small.pdf"
      
      doc = fitz.open(str(pdf))

      # first pass spans
      spans = list(self.extract_spans(doc))
      dom_size = self.dominant_font_size_by_chars(spans)
      #dom_size = pick_body_font_size(spans, headers, footers, debug=False)

      # build two filtered PDFs in temp files
      big_doc = fitz.open()  # new empty
      big_doc.insert_pdf(doc) # clone
      big_doc = self.build_filtered_pdf_redaction(big_doc, spans, dom_size, keep_big=True)
      big_doc.save(str(big_pdf_path))
      big_doc.close()

      small_doc = fitz.open()  # new empty
      small_doc.insert_pdf(doc) # clone
      small_doc = self.build_filtered_pdf_redaction(small_doc, spans, dom_size, keep_big=False)
      small_doc.save(str(small_pdf_path))
      small_doc.close()

      # docling on both
      big_md = self.run_docling_on_pdf(str(big_pdf_path)).strip()
      small_md = self.run_docling_on_pdf(str(small_pdf_path)).strip()

      # tables from original PDF
      table_texts = self.extract_tables_from_pdf(doc, name)

      # images OCR from original PDF
      figure_texts = self.extract_figure_texts_from_pdf(doc, name, ocr_lang=ocr_figure_lang)

      # combine
      combined = []

      if big_md:
        combined.append(big_md)

      if small_md:
        combined.append("\n# Small-font content\n")
        combined.append(small_md)

      if table_texts:
        combined.append("\n# Tables\n")
        combined.extend(table_texts)

      if figure_texts:
          combined.append("\n# Figures\n")
          combined.extend(figure_texts)


      final_md = "\n".join(combined).strip() + "\n"

      with out_path.open("w", encoding="utf-8") as out:
        out.write(final_md)

    print("Text extraction complete")

  # CLEANUP

  '''
  Clean up temp dir if needed
  '''
  def __exit__(self, exc_type, exc_value, traceback):
    if not self.keep_divided_pdfs:
      self.tmpdir.cleanup()
    print("### PDFExtractor - close ###")
    pass
