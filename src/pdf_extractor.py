"""
This module implements a class that takes a list of PDF files and an output directory, and extracts all text from the PDFs into markdown files.
The extraction is ran with the run_text_extraction method and includes body text, text from tables and text from images.
"""

import re
import easyocr
import numpy as np
import collections, io, tempfile, fitz
from pathlib import Path
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

  def _extract_spans(self, doc):
    """
    Yield (page_idx, span_dict, page_height)
    """
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

  def _is_in_header_or_footer_region(self, line_bbox, page_height, margin_ratio=0.10):
    """
    Identify if a line (given by its bbox) is likely in the header or footer region of the page, based on a margin ratio.
    """
    y0, y1 = line_bbox[1], line_bbox[3]
    return (y0 <= page_height * margin_ratio) or (y1 >= page_height * (1 - margin_ratio))

  def _dominant_font_size_by_chars(self, spans, debug=False):
    """
    Compute dominant font size weighted by number of characters.
    Supports both span shapes:
    1) (page_idx, span_dict, line_bbox, page_height)
    2) (page_idx, text, size, y0, y1, page_height)
    """
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
        print("_dominant_font_size_by_chars: no chars counted. spans sample:", spans[:5])
      return 0.0

    # pick size with the highest total char count
    dom_size, char_count = max(char_counter.items(), key=lambda x: x[1])

    if debug:
      print("font char counts:", char_counter)
      print("dominant:", dom_size, "chars:", char_count)

    return dom_size

  def _pick_body_font_size(self, spans, headers=None, footers=None, debug=False):
    """
    Choose the body font size based on:
    - how many pages a font size appears on (coverage)
    - how many characters it has in total (tie-breaker)

    spans must be like: (page_idx, span_dict, line_bbox, page_height)
    headers/footers are sets of strings to ignore.
    """
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
        # if your _extract_spans has a different shape, fix that upstream
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

  def _collect_body_rects(self, spans, threshold, eps, padding_factor=1.5):
    """
    Collect rectangles representing the body text for each page.
    """
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

  def _overlaps_body(self, bbox, body_rects, min_overlap_ratio=0.2):
    """
    Check if a bbox overlaps significantly with any of the body rects on the page.
    """
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
  
  def _build_filtered_pdf_redaction(self, src_doc, spans, threshold, keep_big=True, eps=0.5):
    """
    Open original PDF, redact spans we DON'T want, keep the rest.
    This preserves layout and fonts.
    keep_big=True  -> keep size >= threshold, redact smaller
    keep_big=False -> keep size < threshold, redact bigger
    """
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

        #if _is_in_header_or_footer_region(line_bbox, ph):
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

  def run_docling_on_pdf(self, path_str: str) -> str:
    """
    Run Docling conversion on a PDF and return the markdown string.
    """
    res = self.docling_converter.convert(path_str)
    return res.document.export_to_markdown()

  # TABLE EXTRACTION


  def _clean_cell(self, value) -> str:
    """
    Clean a cell value by normalizing whitespace and stripping.
    """
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text

  def _looks_like_header_row(self, row) -> bool:
    """
    Heuristic:
    - header row should be mostly non-empty
    - usually short textual labels, not mostly numeric
    """
    cleaned = [self._clean_cell(x) for x in row]
    non_empty = [x for x in cleaned if x]
    if len(non_empty) < max(1, len(cleaned) // 2):
        return False

    numeric_like = 0
    for cell in non_empty:
        if re.fullmatch(r"[%\d\.\,\-\+\(\)\/]+", cell):
            numeric_like += 1

    return numeric_like < len(non_empty) / 2

  def _normalize_header(self, header: str, fallback_idx: int) -> str:
    """
    Normalize a header cell.
    """
    header = self._clean_cell(header).lower()
    if not header:
        return f""
    return header

  def _pad_rows(self, rows):
    """
    Pad all rows to the same number of columns.
    """
    if not rows:
        return rows
    max_cols = max(len(r) for r in rows)
    return [list(r) + [""] * (max_cols - len(r)) for r in rows]

  def _forward_fill_rowspans(self, rows):
    """
    Forward-fill empty cells vertically to approximate rowspan behavior.

    Example:
    Female | mouse | x
           | rat   | y
           | human | z

    becomes:
    Female | mouse | x
    Female | rat   | y
    Female | human | z
    """
    if not rows:
        return rows

    rows = self._pad_rows(rows)
    filled = [rows[0][:]]

    for row_idx in range(1, len(rows)):
        prev = filled[row_idx - 1]
        curr = rows[row_idx][:]
        for col_idx, cell in enumerate(curr):
            if (
                not self._clean_cell(cell)
                and any(self._clean_cell(x) for j, x in enumerate(curr) if j != col_idx)
            ):
                curr[col_idx] = prev[col_idx]
        filled.append(curr)

    return filled

  def _serialize_table_rows(self, rows, table_label: str) -> str | None:
    """
    Convert a table (list of rows) into a serialized string:
    Table N: header1 value1, header2 value2. header1 value1, header2 value2.
    """
    if not rows:
        return None

    cleaned_rows = [
        [self._clean_cell(cell) for cell in row]
        for row in rows
    ]

    cleaned_rows = [
        row for row in cleaned_rows
        if any(cell for cell in row)
    ]

    if not cleaned_rows:
        return None

    cleaned_rows = self._pad_rows(cleaned_rows)

    # choose header
    if len(cleaned_rows) >= 2 and self._looks_like_header_row(cleaned_rows[0]):
        headers = [
            self._normalize_header(h, idx + 1)
            for idx, h in enumerate(cleaned_rows[0])
        ]
        data_rows = cleaned_rows[1:]
    else:
        max_cols = max(len(r) for r in cleaned_rows)
        headers = ["" for i in range(max_cols)]
        data_rows = cleaned_rows

    if not data_rows:
        return None

    # fill vertically so merged cells spanning multiple rows are repeated
    data_rows = self._forward_fill_rowspans(data_rows)

    row_texts = []
    for row in data_rows:
        pieces = []
        for idx, cell in enumerate(row):
            cell = self._clean_cell(cell)
            if not cell:
                continue
            header = headers[idx] if idx < len(headers) else ""
            pieces.append(f"{header}{' ' if len(header) > 0 else ''}{cell}")

        if pieces:
            row_texts.append(", ".join(pieces))

    if not row_texts:
        return None

    return f"{table_label}:\n" + ". ".join(row_texts) + "."

  def extract_tables_from_pdf(self, doc, paper_id: str):
    """
    Return a list of serialized table strings extracted from the PDF.
    Uses PyMuPDF's built-in table detection on each page.
    """
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

        label = f"## Table {global_table_idx}"
        serialized = self._serialize_table_rows(rows, label)
        if serialized:
          extracted_tables.append(serialized + "\n")

        global_table_idx += 1

    return extracted_tables

  # IMAGE EXTRACTION

  def _normalize_ocr_text(self, text: str) -> str:
    text = text.replace("\x0c", " ")
    text = text.replace("\r", "\n")

    # fix hyphenation across line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # preserve paragraph breaks
    text = re.sub(r"\n{2,}", "<<<PARA>>>", text)

    # single line breaks -> space
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)

    # restore paragraphs
    text = text.replace("<<<PARA>>>", "\n\n")

    # normalize spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n\n *", "\n\n", text)

    return text.strip()
  
  def _prepare_image_for_ocr(self, img: Image.Image) -> np.ndarray:
      img = ImageOps.exif_transpose(img)

      if img.mode == "RGBA":
          bg = Image.new("RGB", img.size, (255, 255, 255))
          bg.paste(img, mask=img.split()[3])
          img = bg
      elif img.mode != "RGB":
          img = img.convert("RGB")

      # Mild upscale for small images
      w, h = img.size
      min_dim = 1600
      if min(w, h) < min_dim:
          scale = min_dim / min(w, h)
          img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

      return np.array(img)

  def _get_easy_ocr(self, lang: list | None = None):
    """
    Lazy init so you don't reload models for every figure.
    """
    if lang is None:
        lang = ["en"]
    lang_key = tuple(sorted(lang))

    if not hasattr(self, "_easy_ocr_cache"):
        self._easy_ocr_cache = {}

    if lang_key not in self._easy_ocr_cache:
        self._easy_ocr_cache[lang_key] = easyocr.Reader(list(lang_key), gpu=False)

    return self._easy_ocr_cache[lang_key]

  @staticmethod
  def _polygon_to_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return {
        "x0": float(x0),
        "y0": float(y0),
        "x1": float(x1),
        "y1": float(y1),
        "w": float(x1 - x0),
        "h": float(y1 - y0),
        "cx": float((x0 + x1) / 2.0),
        "cy": float((y0 + y1) / 2.0),
    }

  @staticmethod
  def _alnum_ratio(text: str) -> float:
    if not text:
        return 0.0
    alnum = sum(ch.isalnum() for ch in text)
    return alnum / max(len(text), 1)

  def _group_detected_text_boxes(self, items):
    """
    Group OCR detections into text blocks using pairwise adjacency
    instead of greedy sequential merging.

    Works much better for multiline text inside boxes.
    """
    if not items:
        return []

    items = sorted(items, key=lambda it: (it["bbox"]["y0"], it["bbox"]["x0"]))

    heights = [it["bbox"]["h"] for it in items if it["bbox"]["h"] > 0]
    avg_h = float(np.mean(heights)) if heights else 20.0

    same_line_y_tol = max(8.0, avg_h * 0.55)
    multiline_gap_tol = max(10.0, avg_h * 1.0)
    horiz_gap_tol = max(18.0, avg_h * 1.5)
    left_align_tol = max(12.0, avg_h * 0.9)

    n = len(items)
    adj = [[] for _ in range(n)]

    def x_overlap(a, b):
        return max(0.0, min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]))

    def y_overlap(a, b):
        return max(0.0, min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]))

    def horiz_gap(a, b):
        if a["x1"] < b["x0"]:
            return b["x0"] - a["x1"]
        if b["x1"] < a["x0"]:
            return a["x0"] - b["x1"]
        return 0.0

    def vert_gap(a, b):
        if a["y1"] < b["y0"]:
            return b["y0"] - a["y1"]
        if b["y1"] < a["y0"]:
            return a["y0"] - b["y1"]
        return 0.0

    def should_link(box1, box2):
        h1, h2 = box1["h"], box2["h"]
        w1, w2 = box1["w"], box2["w"]

        xo = x_overlap(box1, box2)
        yo = y_overlap(box1, box2)
        hg = horiz_gap(box1, box2)
        vg = vert_gap(box1, box2)

        # normalized overlaps
        xo_ratio = xo / max(1.0, min(w1, w2))
        yo_ratio = yo / max(1.0, min(h1, h2))

        left_aligned = abs(box1["x0"] - box2["x0"]) <= left_align_tol
        right_aligned = abs(box1["x1"] - box2["x1"]) <= left_align_tol

        # same OCR line split into multiple chunks
        same_line = (
            abs(box1["cy"] - box2["cy"]) <= same_line_y_tol
            and hg <= horiz_gap_tol
        )

        # two stacked lines of same multiline block
        stacked_multiline = (
            vg <= multiline_gap_tol
            and (
                xo_ratio >= 0.45
                or left_aligned
                or right_aligned
            )
        )

        # Prevent absurd merges between far-apart columns
        far_apart_columns = (
            hg > max(40.0, avg_h * 3.0)
            and xo_ratio < 0.15
        )

        if far_apart_columns:
            return False

        return same_line or stacked_multiline

    # build adjacency graph
    for i in range(n):
        bi = items[i]["bbox"]
        for j in range(i + 1, n):
            bj = items[j]["bbox"]

            # cheap early exit on vertical distance
            if abs(bi["cy"] - bj["cy"]) > avg_h * 4.0 and vert_gap(bi, bj) > avg_h * 2.0:
                continue

            if should_link(bi, bj):
                adj[i].append(j)
                adj[j].append(i)

    # connected components
    visited = [False] * n
    groups_idx = []

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []

        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

        groups_idx.append(comp)

    # reconstruct text inside each component
    block_texts = []

    for comp in groups_idx:
        group = [items[i] for i in comp]
        group.sort(key=lambda it: (it["bbox"]["y0"], it["bbox"]["x0"]))

        # local line clustering
        local_lines = []

        for it in group:
            placed = False
            for line in local_lines:
                line_y = np.mean([x["bbox"]["cy"] for x in line])
                if abs(it["bbox"]["cy"] - line_y) <= same_line_y_tol:
                    line.append(it)
                    placed = True
                    break
            if not placed:
                local_lines.append([it])

        local_lines.sort(key=lambda line: min(x["bbox"]["y0"] for x in line))

        line_texts = []
        for line in local_lines:
            line.sort(key=lambda x: x["bbox"]["x0"])
            txt = " ".join(x["text"] for x in line).strip()
            txt = self._normalize_ocr_text(txt)
            if txt:
                line_texts.append(txt)

        block = "\n".join(line_texts).strip()
        if block:
            block_texts.append({
                "text": block,
                "y": min(it["bbox"]["y0"] for it in group),
                "x": min(it["bbox"]["x0"] for it in group),
            })

    block_texts.sort(key=lambda b: (b["y"], b["x"]))
    return [b["text"] for b in block_texts]
  
  def _ocr_image_detect_text_blocks(self, img: Image.Image, lang: str = "en") -> str:
    """
    Detect text regions with EasyOCR, then group nearby detections into blocks.
    Returns one block separated by blank lines.
    """
    arr = self._prepare_image_for_ocr(img)
    reader = self._get_easy_ocr(lang=[lang])

    # EasyOCR returns list of (bbox_points, text, confidence)
    # bbox_points: [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
    result = reader.readtext(arr)

    items = []

    for (poly, text, score) in result:
        text = self._normalize_ocr_text(str(text or ""))
        score = float(score) if score is not None else 0.0

        if not text:
            continue
        if score < 0.35:
            continue
        if self._alnum_ratio(text) < 0.25:
            continue

        bbox = self._polygon_to_bbox(poly)
        if bbox["w"] < 6 or bbox["h"] < 6:
            continue

        items.append({
            "text": text,
            "score": score,
            "bbox": bbox,
        })

    if not items:
        return ""

    blocks = self._group_detected_text_boxes(items)

    # final cleanup + dedup
    final_blocks = []
    seen = set()
    for blk in blocks:
        blk = self._normalize_ocr_text(blk)
        key = re.sub(r"\s+", " ", blk).strip().lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        final_blocks.append(blk)

    return ". ".join(final_blocks).strip() + '.'

  def extract_figure_texts_from_pdf(self, doc, paper_id: str, min_size: int = 200, ocr_lang: str = "en"):
    """
    Extract embedded images from the PDF, OCR them, and return a list like:
    ["Figure 1: ...", "Figure 2: ..."]
    """
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
          ocr_text = self._ocr_image_detect_text_blocks(img, lang=ocr_lang)
        except Exception as e:
          print(f"  [!] OCR failed for image xref {xref} in {paper_id}: {e}")
          continue

        if not ocr_text:
          continue

        label = f"## Figure {counter}"
        figure_texts.append(f"{label}:\n{ocr_text}\n")
        counter += 1

    return figure_texts

  # FULL EXTRACTION PIPELINE

  def run_text_extraction(self, folder="markdown", ocr_figure_lang: str = "en", only_tables=False, only_figures=False):
    """
    Iterates paper_files and extracts text into markdown files in output_dir/folder.
    For each PDF it does the following:
    1. Divide body text (font size >= threshold) from small text
    2. Extract text from tables, trasforming each row into a sentence
    3. Extract text from figures
    4. Finally combines body text, small text, table text, and figure text in an output markdown with sections.
    """
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
      spans = list(self._extract_spans(doc))
      dom_size = self._dominant_font_size_by_chars(spans)
      #dom_size = _pick_body_font_size(spans, headers, footers, debug=False)

      # build two filtered PDFs in temp files
      big_doc = fitz.open()  # new empty
      big_doc.insert_pdf(doc) # clone
      big_doc = self._build_filtered_pdf_redaction(big_doc, spans, dom_size, keep_big=True)
      big_doc.save(str(big_pdf_path))
      big_doc.close()

      small_doc = fitz.open()  # new empty
      small_doc.insert_pdf(doc) # clone
      small_doc = self._build_filtered_pdf_redaction(small_doc, spans, dom_size, keep_big=False)
      small_doc.save(str(small_pdf_path))
      small_doc.close()

      # docling on both
      big_md, small_md = None, None
      if not only_tables and not only_figures:
        big_md = self.run_docling_on_pdf(str(big_pdf_path)).strip()
        small_md = self.run_docling_on_pdf(str(small_pdf_path)).strip()

      # tables from original PDF
      table_texts = None
      if not only_figures:
        table_texts = self.extract_tables_from_pdf(doc, name)

      # images OCR from original PDF
      figure_texts = None
      if not only_tables:
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

  def __exit__(self, exc_type, exc_value, traceback):
    """
    Clean up temp dir if needed
    """
    if not self.keep_divided_pdfs:
      self.tmpdir.cleanup()
    print("### PDFExtractor - close ###")
    pass
