"""
This module implements a class that takes a list of markdown files and an output directory, and divides each markdown file (in a redundant way) into:
- Lines
- Sentences
- Paragraphs (sections with title and body)
- Sections (like paragraphs but with heading level and filtered by blacklist/heuristics)
- Chunks (assembled from sentences, with a target size to comply with model token limits)
The results are saved as JSON files. The division is based on markdown headings, with some heuristics to filter out uninformative sections (like references) and to keep sentence-like headings.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re

HEADING_BLACKLIST = {
  "references", "reference",
  "bibliography", "citation", "citations",
  "acknowledgements", "acknowledgments",
  "funding", "conflicts of interest", "conflict of interest",
  "author contributions", "ethical approval",
  "supplementary material", "supplementary materials",
}

ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "vs.", "dr.", "mr.", "mrs.", "ms.",
    "fig.", "tab.", "eq.", "ref.", "al.", "et al."
}

class MarkdownDivider:
  # INITIALIZATION
  def __init__(
    self,
    md_files: List[Path],
    output_dir: Path,
    skip_existing: bool = False,

    # token-aware chunking
    min_chunk_tokens: int = 96,
    target_chunk_tokens: int = 160,
    max_chunk_tokens: int = 220,

    # full-example budgeting
    model_context_tokens: int = 8192,
    reserved_prompt_tokens: int = 1200,
    reserved_output_tokens: int = 300,
    safety_margin_tokens: int = 96,

    tokenizer=None,
  ) -> None:
    print("### MarkdownDivider - init ###")
    self.md_files = md_files
    self.output_dir = output_dir
    self.skip_existing = skip_existing

    # Section headings blacklist
    self.blacklisted_headings = HEADING_BLACKLIST

    # Chunking parameters
    self.min_chunk_tokens = min_chunk_tokens
    self.target_chunk_tokens = target_chunk_tokens
    self.max_chunk_tokens = max_chunk_tokens

    self.model_context_tokens = model_context_tokens
    self.reserved_prompt_tokens = reserved_prompt_tokens
    self.reserved_output_tokens = reserved_output_tokens
    self.safety_margin_tokens = safety_margin_tokens

    # Tokenizer
    self.tokenizer = tokenizer

    # Effective hard budget for chunk text only
    self.available_chunk_budget = max(
        32,
        self.model_context_tokens
        - self.reserved_prompt_tokens
        - self.reserved_output_tokens
        - self.safety_margin_tokens
    )

    # Final guardrail
    self.max_chunk_tokens = min(self.max_chunk_tokens, self.available_chunk_budget)
    self.target_chunk_tokens = min(self.target_chunk_tokens, self.max_chunk_tokens)
    self.min_chunk_tokens = min(self.min_chunk_tokens, self.target_chunk_tokens)

  def __enter__(self):
    return self

  # BASIC HELPERS

  def _is_heading(self, line: str) -> bool:
    """
    Matches any line starting with '#' after left-stripping
    """
    return line.lstrip().startswith("#")

  def _clean_heading(self, line: str) -> str:
    """
    Remove leading '#' chars and surrounding whitespace
    """
    return line.lstrip().lstrip("#").strip()

  def _heading_level(self, line: str) -> int:
    """
    Count the number of leading '#' chars to determine heading level
    """
    stripped = line.lstrip()
    return len(stripped) - len(stripped.lstrip("#"))
    
  def _normalize_heading(self, title: str) -> str:
    """
    Normalize heading by collapsing multiple spaces and converting to lowercase
    """
    return re.sub(r"\s+", " ", title).strip().lower()

  def _is_blacklisted_heading(self, title: str) -> bool:
    """
    Check if the normalized heading is in the blacklist or starts with a blacklisted keyword.
    """
    norm = self._normalize_heading(title)
    # exact match or starts with keyword
    if norm in self.blacklisted_headings:
      return True
    for bad in self.blacklisted_headings:
      if norm.startswith(bad + " "):
        return True
    return False

  def _looks_like_sentence_heading(self, title: str) -> bool:
    """
    Heuristic: keep headings that can plausibly be a real sentence.
    We prefer to be permissive to avoid losing events.
    """
    t = title.strip()
    if not t:
      return False

    # If it ends with sentence punctuation, likely a sentence
    if re.search(r"[.!?]$", t):
      return True

    # If it has at least 6 words and contains lowercase letters,
    # it might be a sentence-like statement.
    words = t.split()
    if len(words) >= 6 and re.search(r"[a-z]", t):
      return True

    # If it contains an obvious verb-ish connector in English
    # (cheap heuristic)
    if re.search(r"\b(explains|shows|demonstrates|indicates|suggests|causes|leads)\b", t.lower()):
      return True

    return False
  
  # TOKEN COUNTING

  def _count_tokens(self, text: str) -> int:
      """
      Count tokens in text. If a tokenizer is provided, use it. Otherwise, use a rough approximation based on word count.
      """
      text = text.strip()
      if not text:
          return 0

      if self.tokenizer is not None:
          ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
          return len(ids)

      words = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
      return max(1, int(len(words) * 1.25))

  # SENTENCE SPLITTING

  def _protect_abbreviations(self, text: str) -> str:
    """
    Replace common abbreviations with a protected version to prevent splitting on their periods.
    For example, "e.g." becomes "e<DOT>g<DOT>" internally during sentence splitting, and is restored afterward.
    """
    protected = text
    for abbr in sorted(ABBREVIATIONS, key=len, reverse=True):
        safe = abbr.replace(".", "<DOT>")
        protected = re.sub(
            re.escape(abbr),
            safe,
            protected,
            flags=re.IGNORECASE
        )
    return protected

  def _restore_abbreviations(self, text: str) -> str:
    """
    Restore protected abbreviations back to their original form after sentence splitting.
    """
    return text.replace("<DOT>", ".")
  
  def _split_sentences(self, text: str) -> List[str]:
    """
    Better scientific-ish sentence splitter without external deps.
    Handles:
    - ., !, ?
    - keeps decimals like 3.5 together
    - protects common abbreviations
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    protected = self._protect_abbreviations(text)

    # avoid splitting decimal numbers: 3.5
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DECIMAL>", protected)

    # split after punctuation followed by space + uppercase/number/bracket/quote
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9(\["\'])', protected)

    sentences = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        part = part.replace("<DECIMAL>", ".")
        part = self._restore_abbreviations(part)
        sentences.append(part)

    return sentences

    """
    Naive sentence splitter based on periods followed by whitespace.
    """
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

  # SECTION EXTRACTION

  def _extract_sections(
    self,
    raw_lines: List[str]
  ) -> Tuple[List[str], List[str], List[Dict[str, str]], List[Dict[str, Any]]]:
    file_lines: List[str] = []
    file_sentences: List[str] = []
    paragraphs: List[Dict[str, str]] = []
    sections: List[Dict[str, Any]] = []

    current_title: Optional[str] = None
    current_level: int = 0
    current_body_lines: List[str] = []

    def flush_current_section():
        nonlocal current_title, current_level, current_body_lines

        if current_title is None:
            current_body_lines = []
            return

        body = "\n".join(current_body_lines).strip()

        sec = {
            "title": current_title,
            "level": current_level,
            "body": body
        }
        sections.append(sec)
        paragraphs.append({
            "title": current_title,
            "body": body
        })

        current_body_lines = []

    for raw in raw_lines:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            continue

        if self._is_heading(stripped):
            flush_current_section()
            current_title = self._clean_heading(stripped)
            current_level = self._heading_level(stripped)
            continue

        file_lines.append(stripped)
        file_sentences.extend(self._split_sentences(stripped))

        if current_title is not None:
            current_body_lines.append(stripped)

    flush_current_section()

    filtered_sections: List[Dict[str, Any]] = []
    for sec in sections:
        title = sec.get("title", "") or ""
        body = sec.get("body", "") or ""

        if self._is_blacklisted_heading(title):
            continue

        if not body.strip() and not self._looks_like_sentence_heading(title):
            continue

        filtered_sections.append(sec)

    return file_lines, file_sentences, paragraphs, filtered_sections

  # CHUNK BUILDING

  def _section_to_units(self, section: Dict[str, Any]) -> List[str]:
    """
    Build chunkable units for a single section.
    Keeps heading context local to the section.
    """
    title = (section.get("title") or "").strip()
    body = (section.get("body") or "").strip()

    units: List[str] = []

    # preserve heading context explicitly
    heading_prefix = f"[SECTION] {title}" if title else ""

    if heading_prefix:
        units.append(heading_prefix)

    if body:
        for line in body.split("\n"):
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                continue
            units.extend(self._split_sentences(line))

    return units
  
  def _split_oversized_unit(self, text: str) -> List[str]:
    """
    Last-resort split for a single oversized sentence/unit.
    Tries punctuation first, then falls back to token-ish windows.
    """
    if self._count_tokens(text) <= self.max_chunk_tokens:
        return [text]

    # first attempt: split on semicolons / commas / colons
    parts = re.split(r'(?<=[;,:])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) > 1 and all(self._count_tokens(p) < self._count_tokens(text) for p in parts):
        out: List[str] = []
        cur = ""
        for p in parts:
            candidate = f"{cur} {p}".strip() if cur else p
            if self._count_tokens(candidate) <= self.max_chunk_tokens:
                cur = candidate
            else:
                if cur:
                    out.append(cur)
                cur = p
        if cur:
            out.append(cur)
        return out

    # hard fallback: split by words
    words = text.split()
    out = []
    cur_words: List[str] = []

    for w in words:
        candidate = " ".join(cur_words + [w]).strip()
        if self._count_tokens(candidate) <= self.max_chunk_tokens:
            cur_words.append(w)
        else:
            if cur_words:
                out.append(" ".join(cur_words))
            cur_words = [w]

    if cur_words:
        out.append(" ".join(cur_words))

    return out

  def _chunk_section_units(self, units: List[str]) -> List[str]:
    """
    Chunk within one section only.
    Respects token budget.
    Avoids generating trash tiny chunks when possible.
    """
    if not units:
        return []

    normalized_units: List[str] = []
    for u in units:
        if self._count_tokens(u) > self.max_chunk_tokens:
            print(f"Warning: single unit exceeds max chunk tokens ({self._count_tokens(u)} > {self.max_chunk_tokens}). Attempting to split it.")
            normalized_units.extend(self._split_oversized_unit(u))
        else:
            normalized_units.append(u)

    chunks: List[str] = []
    cur: List[str] = []

    def cur_text() -> str:
        return " ".join(cur).strip()

    for unit in normalized_units:
        if not cur:
            cur = [unit]
            continue

        candidate = f"{cur_text()} {unit}".strip()
        candidate_tokens = self._count_tokens(candidate)

        # grow until hard max
        if candidate_tokens <= self.max_chunk_tokens:
            cur.append(unit)
            continue

        # flush current
        chunks.append(cur_text())
        cur = [unit]

    if cur:
        chunks.append(cur_text())

    # merge too-small adjacent chunks, but never exceed max
    merged: List[str] = []
    i = 0
    while i < len(chunks):
        current = chunks[i]

        if self._count_tokens(current) >= self.min_chunk_tokens or i == len(chunks) - 1:
            merged.append(current)
            i += 1
            continue

        nxt = chunks[i + 1]
        candidate = f"{current} {nxt}".strip()
        if self._count_tokens(candidate) <= self.max_chunk_tokens:
            merged.append(candidate)
            i += 2
        else:
            merged.append(current)
            i += 1

    return merged

  def _build_chunks_from_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for sec_idx, sec in enumerate(sections):
        units = self._section_to_units(sec)
        section_chunks = self._chunk_section_units(units)

        for ch in section_chunks:
            chunks.append({
                "chunk_id": chunk_id,
                "section_index": sec_idx,
                "section_title": sec.get("title", ""),
                "section_level": sec.get("level", 0),
                "token_count": self._count_tokens(ch),
                "text": ch,
                "events": []
            })
            chunk_id += 1

    return chunks

  # FULL DIVISION PIPELINE

  def divide_files(self, folder: str):
    """
    Run the full markdown division pipeline on the provided files, saving results to JSON.
    Each JSON will contain:
    - lines: all non-empty, non-heading lines
    - sentences: all sentences from those lines
    - paragraphs: sections with title and body (body is all text under that heading until next heading)
    - sections: like paragraphs but with heading level and filtered by blacklist/heuristics
    """
    total = len(self.md_files)
    print(f"Found {total} markdown files to process for markdown division.")

    for i, md_file in enumerate(self.md_files, 1):
      print(f"[{i}/{total}] Processing {md_file.name}...")
      
      if not md_file.is_file():
        continue

      # filename without .md
      stem = md_file.stem
      output_json_path = self.output_dir / folder / f"{stem}_divided.json"
      output_json_path.parent.mkdir(parents=True, exist_ok=True)

      if self.skip_existing and output_json_path.exists():
        print(f"\tSkipping {stem} because {output_json_path} exists.")
        continue

      with md_file.open("r", encoding="utf-8") as f:
        raw_lines = f.readlines()

      file_lines, file_sentences, paragraphs, filtered_sections = self._extract_sections(raw_lines)
      chunks = self._build_chunks_from_sections(filtered_sections)

      result = {
        "lines": file_lines,
        "sentences": file_sentences,
        "paragraphs": paragraphs,
        "sections": filtered_sections,
        "chunks": chunks,
        "chunking_config": {
            "min_chunk_tokens": self.min_chunk_tokens,
            "target_chunk_tokens": self.target_chunk_tokens,
            "max_chunk_tokens": self.max_chunk_tokens,
            "model_context_tokens": self.model_context_tokens,
            "reserved_prompt_tokens": self.reserved_prompt_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
          "safety_margin_tokens": self.safety_margin_tokens,
          "available_chunk_budget": self.available_chunk_budget,
          "tokenizer_aware": self.tokenizer is not None,
        }
     }

      with output_json_path.open("w", encoding="utf-8") as json_file:
        json.dump(result, json_file, indent=2, ensure_ascii=False)
    
    print("Markdown division complete")
  
  # EXIT

  def __exit__(self, exc_type, exc_value, traceback):
    print("### MarkdownDivider - exit ###")
    pass
