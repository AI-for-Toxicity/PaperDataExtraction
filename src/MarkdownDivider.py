import json
from pathlib import Path
from typing import List, Dict, Any
import re

class MarkdownDivider:
  def __init__(self, md_files: List[Path], output_dir: Path, skip_existing: bool = False) -> None:
    print("### MarkdownDivider - init ###")
    self.md_files = md_files
    self.output_dir = output_dir
    self.skip_existing = skip_existing

  def __enter__(self):
    return self

  def _is_heading(self, line: str) -> bool:
    # Basic: any line starting with '#' after left-stripping
    return line.lstrip().startswith("#")

  def _clean_heading(self, line: str) -> str:
    # Remove leading '#' chars and surrounding whitespace
    return line.lstrip().lstrip("#").strip()

  def _split_sentences(self, text: str) -> List[str]:
    # Remove trailing spaces first to avoid weird empty chunks
    text = text.strip()
    if not text:
      return []

    parts = re.split(r'\.\s+', text)
    sentences = []

    for i, part in enumerate(parts):
      part = part.strip()
      if not part:
        continue

      # If it's not the last part and doesn't already end with '.', add it back
      if i < len(parts) - 1 and not part.endswith('.'):
        part = part + '.'

      sentences.append(part)

    return sentences

  def divide_files(self, folder: str):
    total = len(self.md_files)
    print(f"Found {total} markdown files to process for markdown division.\n")

    for i, md_file in enumerate(self.md_files, 1):
      print(f"[{i}/{total}] Processing {md_file.name}...")
      result: Dict[str, Any] = {}
      
      if not md_file.is_file():
        # You asked for code, not babysitting, so skip invalid paths.
        continue

      # filename without .md
      stem = md_file.stem
      output_json_path = self.output_dir / folder / f"{stem}_divided.json"
      output_json_path.parent.mkdir(parents=True, exist_ok=True)
      if self.skip_existing and output_json_path.exists():
        print(f"Skipping {stem} because {output_json_path} exists.")
        continue

      with md_file.open("r", encoding="utf-8") as f:
        raw_lines = f.readlines()

      file_lines: List[str] = []        # non-heading, non-empty
      file_sentences: List[str] = []    # from those lines
      paragraphs: List[Dict[str, str]] = []

      current_title: str | None = None
      current_body_lines: List[str] = []

      for raw in raw_lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        # Skip empty lines globally
        if not stripped:
          continue

        if self._is_heading(stripped):
          # Flush previous paragraph if any
          if current_title is not None:
            body = "\n".join(current_body_lines).strip()
            paragraphs.append({
              "title": current_title,
              "body": body
            })
            current_body_lines = []

          # Start new paragraph with this heading
          current_title = self._clean_heading(stripped)
          continue  # headings are not in lines/sentences

        # Non-heading, non-empty line
        file_lines.append(stripped)

        # Add to sentences
        file_sentences.extend(self._split_sentences(stripped))

        # If we are inside a paragraph, accumulate lines for its body
        if current_title is not None:
          current_body_lines.append(stripped)

      # Flush last paragraph if any
      if current_title is not None:
        body = "\n".join(current_body_lines).strip()
        paragraphs.append({
          "title": current_title,
          "body": body
        })

      result = {
        "lines": file_lines,
        "sentences": file_sentences,
        "paragraphs": paragraphs
      }

      with output_json_path.open("w", encoding="utf-8") as json_file:
        json.dump(result, json_file, indent=2, ensure_ascii=False)
    
    print("Markdown division complete")
  
  def __exit__(self, exc_type, exc_value, traceback):
    print("### MarkdownDivider - exit ###")
    pass
