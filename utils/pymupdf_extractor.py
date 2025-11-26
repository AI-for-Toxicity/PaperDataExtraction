#!/usr/bin/env python3
import collections
import fitz  # PyMuPDF
from pathlib import Path
import tempfile
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

PDF_DIR = Path("data/raw/pdfs/")
TMP_DIR = Path("eval/")
OUTPUT_DIR = Path("data/raw/markdown/")
HEADER_FOOTER_MARGIN_RATIO = 0.10  # top/bottom 10% of page
DIVIDE_PDFS = True
CONVERT_PDF_WITH_DOCLING = True
SKIP_EXISTING = False

pipeline_options = PdfPipelineOptions(
    do_table_structure=True,
    do_ocr=True,
    generate_table_images=True,
    generate_picture_images=True,
)
docling_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

def extract_spans(doc):
    """Yield (page_idx, span_dict, page_height)."""
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

def is_in_header_or_footer_region(line_bbox, page_height, margin_ratio=HEADER_FOOTER_MARGIN_RATIO):
    y0, y1 = line_bbox[1], line_bbox[3]
    return (y0 <= page_height * margin_ratio) or (y1 >= page_height * (1 - margin_ratio))

def dominant_font_size_by_chars(spans, debug=False):
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
            print("dominant_font_size_by_chars: no chars counted. spans sample:", spans[:5])
        return 0.0

    # pick size with the highest total char count
    dom_size, char_count = max(char_counter.items(), key=lambda x: x[1])

    if debug:
        print("font char counts:", char_counter)
        print("dominant:", dom_size, "chars:", char_count)

    return dom_size

def dominant_font_size_by_chars_old(spans):
    """
    Given spans as (page_idx, span_dict, line_bbox, page_height),
    return the font size that covers the largest number of characters.
    """
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

def dominant_font_size_by_spans(spans):
    counter = collections.Counter()
    for _, span, _, _ in spans:
        size = span.get("size", 0)
        counter[round(size, 1)] += 1
    if not counter:
        return 0
    return max(counter.items(), key=lambda x: x[1])[0]

def pick_body_font_size(spans, headers=None, footers=None, debug=False):
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

def build_filtered_pdf(src_doc, spans, headers, footers, threshold, keep_big=True):
    """
    Create a new PDF containing only spans that match the condition:
    keep_big=True  -> keep spans with size >= threshold
    keep_big=False -> keep spans with size < threshold
    Also skip repeated headers/footers.
    We preserve page sizes and place text at original coordinates.
    """
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

def build_filtered_pdf_redaction(src_path, spans, headers, footers, threshold, keep_big=True, out_path=None, eps=0.5):
    """
    Open original PDF, redact spans we DON'T want, keep the rest.
    This preserves layout and fonts.
    keep_big=True  -> keep size >= threshold, redact smaller
    keep_big=False -> keep size < threshold, redact bigger
    """
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

def run_docling_on_pdf(path_str: str) -> str:
    res = docling_converter.convert(path_str)
    return res.document.export_to_markdown()

def debug_spans_for_word(spans, needle="variability"):
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

def main():
    tmpdir = tempfile.TemporaryDirectory(delete=False, dir=TMP_DIR)
    paper_files = list(PDF_DIR.glob("*.pdf"))
    total = len(paper_files)
    print(f"Found {total} PDFs to process.\n")

    for i, pdf in enumerate(paper_files, 1):
        print(f"[{i}/{total}] Processing {pdf.name}...")
        parts = pdf.stem.split("_")
        name = "_".join(parts[:2]) if len(parts) > 1 else parts[0]
        output_dir = OUTPUT_DIR / name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{name}.md"

        tmp_path_exists = TMP_DIR / f"{name}_big.pdf"
        if SKIP_EXISTING and tmp_path_exists.exists():
            print(f"Skipping {name} because {tmp_path_exists} exists.")
            continue

        tmpdir_path = Path(tmpdir.name) if DIVIDE_PDFS else TMP_DIR
        big_pdf_path = tmpdir_path / f"{name}_big.pdf"
        small_pdf_path = tmpdir_path / f"{name}_small.pdf"
        
        if DIVIDE_PDFS:
            doc = fitz.open(str(pdf))

            # first pass spans
            spans = list(extract_spans(doc))
            headers, footers = [], []
            dom_size = dominant_font_size_by_chars(spans)
            #dom_size = pick_body_font_size(spans, headers, footers, debug=False)

            debug_spans_for_word(spans, needle="variability")
            exit(0)

            # build two filtered PDFs in temp files
            big_doc = build_filtered_pdf_redaction(doc, spans, headers, footers, dom_size, keep_big=True)
            big_doc.save(str(big_pdf_path))
            big_doc.close()

            small_doc = build_filtered_pdf_redaction(doc, spans, headers, footers, dom_size, keep_big=False)
            small_doc.save(str(small_pdf_path))
            small_doc.close()
        
        if CONVERT_PDF_WITH_DOCLING:
            if not big_pdf_path.exists() or not small_pdf_path.exists():
                print(f"Divided PDFs not found for {pdf.name}, returning.")
                return

            # docling on both
            big_md = run_docling_on_pdf(str(big_pdf_path)).strip()
            small_md = run_docling_on_pdf(str(small_pdf_path)).strip()

            # combine
            combined = []
            if big_md:
                combined.append(big_md)
            if small_md:
                combined.append("\n# Small-font content\n")
                combined.append(small_md)

            final_md = "\n".join(combined).strip() + "\n"

            with output_path.open("w", encoding="utf-8") as out:
                out.write(final_md)

if __name__ == "__main__":
    main()
