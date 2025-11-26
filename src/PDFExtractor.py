import collections, io, tempfile, fitz
from pathlib import Path
from PIL import Image
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

class PDFExtractor:
  def __init__(self, paper_files: list, output_dir: Path, skip_existing: bool = False) -> None:
    print("### PDFExtractor - init ###")
    self.pipeline_options = PdfPipelineOptions(
      do_table_structure=True,
      do_ocr=True,
      generate_table_images=True,
      generate_picture_images=True,
    )
    self.docling_converter = DocumentConverter(
      format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
      }
    )
    self.tmpdir = tempfile.TemporaryDirectory()
    self.paper_files = paper_files
    self.output_dir = output_dir
    self.skip_existing = skip_existing
    return
  
  def __enter__(self):
    return self

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
  Given spans as (page_idx, span_dict, line_bbox, page_height),
  return the font size that covers the largest number of characters.
  '''
  def dominant_font_size_by_chars_old(self, spans):
    char_counter = collections.Counter()
    for _, span, _, _ in spans:
      text = span.get("text", "")
      if not text:
          continue
      size = round(span.get("size", 0), 1)
      char_counter[size] += len(text)
    if not char_counter:
      return 0
    # pick the size that accumulated the most characters
    return max(char_counter.items(), key=lambda x: x[1])[0]

  def dominant_font_size_by_spans(self, spans):
    counter = collections.Counter()
    for _, span, _, _ in spans:
      size = span.get("size", 0)
      counter[round(size, 1)] += 1
    if not counter:
      return 0
    return max(counter.items(), key=lambda x: x[1])[0]

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
  Create a new PDF containing only spans that match the condition:
  keep_big=True  -> keep spans with size >= threshold
  keep_big=False -> keep spans with size < threshold
  Also skip repeated headers/footers.
  We preserve page sizes and place text at original coordinates.
  '''
  def build_filtered_pdf(self, src_doc, spans, headers, footers, threshold, keep_big=True):
    out_doc = fitz.open()
    page_count = len(src_doc)

    # we need spans per page to decide placement
    spans_by_page = collections.defaultdict(list)
    for (p_idx, span, line_bbox, ph) in spans:
      spans_by_page[p_idx].append((span, line_bbox, ph))

    for p_idx in range(page_count):
      src_page = src_doc.load_page(p_idx)
      rect = src_page.rect
      new_page = out_doc.new_page(width=rect.width, height=rect.height)

      page_spans = spans_by_page.get(p_idx, [])
      for span, line_bbox, ph in page_spans:
        text = span["text"].strip()
        size = span.get("size", 0)
        fontname = span.get("font", "helv")
        x0, y0, x1, y1 = span["bbox"]

        if text in headers or text in footers:
          continue

        if keep_big and size >= threshold:
          new_page.insert_text(
            (x0, y0),
            text,
            fontsize=size,
            fontname="helv",
          )
        elif not keep_big and size < threshold:
          new_page.insert_text(
            (x0, y0),
            text,
            fontsize=size,
            fontname="helv",
          )

    return out_doc

  '''
  Open original PDF, redact spans we DON'T want, keep the rest.
  This preserves layout and fonts.
  keep_big=True  -> keep size >= threshold, redact smaller
  keep_big=False -> keep size < threshold, redact bigger
  '''
  def build_filtered_pdf_redaction(self, src_path, spans, headers, footers, threshold, keep_big=True, out_path=None, eps=0.5):
    doc = fitz.open(src_path)

    # group spans by page
    spans_by_page = collections.defaultdict(list)
    for (p_idx, span, line_bbox, ph) in spans:
        spans_by_page[p_idx].append((span, line_bbox, ph))

    for p_idx in range(len(doc)):
      page = doc[p_idx]
      page_spans = spans_by_page.get(p_idx, [])
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

        if size <= 0:
          # if size is unknown, don't risk removing it
          continue

        # gray zone: sizes in [body_size - eps, body_size + eps] are treated as body
        lower = threshold - eps
        if keep_big:
          keep = (size >= lower)  # allow slightly smaller glyphs in body lines
        else:
          keep = (size < lower)
            
        if not keep:
          bbox = fitz.Rect(span["bbox"])
          # white fill so background looks normal
          page.add_redact_annot(bbox, fill=(1, 1, 1))
          
      page.apply_redactions()

    if out_path is not None:
      doc.save(out_path)
      doc.close()
      return out_path

    # if caller wants the document object
    return doc

  def run_docling_on_pdf(self, path_str: str) -> str:
      res = self.docling_converter.convert(path_str)
      return res.document.export_to_markdown()

  '''
  Save a fitz.Pixmap to out_path as PNG. Handles CMYK and alpha.
  '''
  def save_pixmap_as_png(self, pix, out_path: Path):
    if pix.n < 5:
      # n < 5: grayscale (1), rgb (3), or with alpha (4)
      pix.save(str(out_path))
    else:
      # CMYK or other: convert to RGB first
      pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
      pix_rgb.save(str(out_path))
      pix_rgb = None
    pix = None

  def extract_images_from_pdf(self, doc, out_dir: Path, paper_id: str, min_size: int):
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
          print(f"  [!] Could not make pixmap for xref {xref} on {paper_id}: {e}")
          continue

        width = pix.width
        height = pix.height

        if width < min_size or height < min_size:
          # discard small images
          pix = None
          continue

        out_name = f"{paper_id}_image_{counter}.png"
        out_path = out_dir / out_name

        try:
          # Use PIL if conversion to PNG via bytes is preferred,
          # but fitz.Pixmap.save also writes PNG reliably.
          self.save_pixmap_as_png(pix, out_path)
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

    return counter - 1

  def extract_tables_from_pdf(self, doc, out_dir: Path, paper_id: str):
    # Not implemented yet
    pass

  def debug_spans_for_word(self, spans, needle="variability"):
    # group spans by (page, line bbox)
    lines = collections.defaultdict(list)
    for p_idx, span, line_bbox, ph in spans:
      lines[(p_idx, tuple(line_bbox))].append(span)

    for (p_idx, bbox), line_spans in lines.items():
      # reconstruct the line text as PyMuPDF created it
      line_text = "".join(s["text"] for s in line_spans)

      # be generous with the match in case PDF splits weirdly
      if needle in line_text or "variabi" in line_text:
        print(f"\n=== Page {p_idx+1}, line bbox={bbox} ===")
        print("FULL LINE:", repr(line_text))
        for s in line_spans:
          print(
            "  span:",
            repr(s["text"]),
            "size=", s["size"],
            "font=", s.get("font"),
            "bbox=", s["bbox"],
          )

  def run_text_extraction(self, folder="markdown"):
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
      headers, footers = [], []
      dom_size = self.dominant_font_size_by_chars(spans)
      #dom_size = pick_body_font_size(spans, headers, footers, debug=False)

      # build two filtered PDFs in temp files
      big_doc = self.build_filtered_pdf_redaction(doc, spans, headers, footers, dom_size, keep_big=True)
      big_doc.save(str(big_pdf_path))
      big_doc.close()

      small_doc = self.build_filtered_pdf_redaction(doc, spans, headers, footers, dom_size, keep_big=False)
      small_doc.save(str(small_pdf_path))
      small_doc.close()

      # docling on both
      big_md = self.run_docling_on_pdf(str(big_pdf_path)).strip()
      small_md = self.run_docling_on_pdf(str(small_pdf_path)).strip()

      # combine
      combined = []
      if big_md:
        combined.append(big_md)
      if small_md:
        combined.append("\n# Small-font content\n")
        combined.append(small_md)

      final_md = "\n".join(combined).strip() + "\n"

      with out_path.open("w", encoding="utf-8") as out:
        out.write(final_md)

    print("Text extraction complete")

  def run_image_extraction(self, folder="images", min_size: int = 50):
    total = len(self.paper_files)
    print(f"Found {total} PDFs to process for image extraction.\n")

    for i, pdf in enumerate(self.paper_files, 1):
      print(f"[{i}/{total}] Processing {pdf.name}...")
      parts = pdf.stem.split("_")
      name = "_".join(parts[:2]) if len(parts) > 1 else parts[0]
      out_dir = self.output_dir / folder
      out_dir.mkdir(parents=True, exist_ok=True)

      if self.skip_existing and any(out_dir.glob(f"{name}_image_*.png")):
        print(f"Skipping {name} because images already exist in {out_dir}.")
        continue

      doc = fitz.open(pdf)
      self.extract_images_from_pdf(doc, out_dir, name, min_size)
      doc.close()

    print("Image extraction complete")

  def run_table_extraction(self, folder="tables"):
    total = len(self.paper_files)
    print(f"Found {total} PDFs to process for table extraction.\n")

    for i, pdf in enumerate(self.paper_files, 1):
      print(f"[{i}/{total}] Processing {pdf.name}...")
      parts = pdf.stem.split("_")
      name = "_".join(parts[:2]) if len(parts) > 1 else parts[0]
      out_dir = self.output_dir / folder
      out_dir.mkdir(parents=True, exist_ok=True)

      if self.skip_existing and any(out_dir.glob(f"{name}_table_*.png")):
        print(f"Skipping {name} because tables already exist in {out_dir}.")
        continue

      doc = fitz.open(pdf)
      self.extract_tables_from_pdf(doc, out_dir, name)
      doc.close()

    print("Table extraction complete")

  def __exit__(self, exc_type, exc_value, traceback):
    self.tmpdir.cleanup()
    print("### PDFExtractor - close ###")
    pass
