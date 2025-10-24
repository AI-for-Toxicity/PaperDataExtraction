from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.document import TableItem, PictureItem
from pathlib import Path
import re

PDF_DIR = Path("data/raw/pdfs/")
OUTPUT_DIR = Path("data/raw/markdown/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Configurable Docling behavior ---
pipeline_options = PdfPipelineOptions(
    do_table_structure=True,
    do_ocr=True,
    generate_table_images=True,
    generate_picture_images=True,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

paper_files = list(PDF_DIR.glob("*.pdf"))
total = len(paper_files)
print(f"Found {total} PDFs to process.\n")

for i, pdf in enumerate(paper_files, 1):
    print(f"[{i}/{total}] Processing {pdf.name}...")
    try:
        # output naming
        parts = pdf.stem.split("_")
        name = "_".join(parts[:2]) if len(parts) > 1 else parts[0]
        output_dir = OUTPUT_DIR / name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{name}.md"

        # --- convert to markdown ---
        result = converter.convert(pdf)
        md_text = result.document.export_to_markdown()

        # Save images of figures and tables
        table_counter = 0
        picture_counter = 0
        for element, _level in result.document.iterate_items():
            if isinstance(element, TableItem):
                table_counter += 1
                element_image_filename = (
                    output_dir / f"{name}_table_{table_counter}.png"
                )
                with element_image_filename.open("wb") as fp:
                    element.get_image(result.document).save(fp, "PNG")

            if isinstance(element, PictureItem):
                picture_counter += 1
                element_image_filename = (
                    output_dir / f"{name}_picture_{picture_counter}.png"
                )
                with element_image_filename.open("wb") as fp:
                    element.get_image(result.document).save(fp, "PNG")
        
        # save cleaned markdown
        with output_path.open("w", encoding="utf-8") as out:
            out.write(md_text)

    except Exception as e:
        print(f"  [!] Failed to process {pdf.name}: {e}")

print("\nText extraction completed successfully.")
