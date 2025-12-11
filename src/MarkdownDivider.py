import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import spacy

class MarkdownDivider:
  def __init__(self, md_files: List[Path], output_dir: Path, skip_existing: bool = False, min_tokens: int = 60, max_tokens: int = 250) -> None:
    print("### MarkdownDivider - init ###")
    self.md_files = md_files
    self.output_dir = output_dir
    self.skip_existing = skip_existing

    # Chunking parameters
    self.min_tokens = min_tokens
    self.max_tokens = max_tokens

    self._nlp = spacy.load("en_core_web_sm")

    # Section headings blacklist
    self.blacklisted_headings = {
      "references", "reference",
      "bibliography", "citation", "citations",
      "acknowledgements", "acknowledgments",
      "funding", "conflicts of interest", "conflict of interest",
      "author contributions", "ethical approval",
      "supplementary material", "supplementary materials",
    }

  def __enter__(self):
    return self

  def _is_heading(self, line: str) -> bool:
    # Basic: any line starting with '#' after left-stripping
    return line.lstrip().startswith("#")

  def _clean_heading(self, line: str) -> str:
    # Remove leading '#' chars and surrounding whitespace
    return line.lstrip().lstrip("#").strip()

  def _normalize_heading(self, title: str) -> str:
    return re.sub(r"\s+", " ", title).strip().lower()

  def _is_blacklisted_heading(self, title: str) -> bool:
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

  def _split_sentences_spacy(self, text: str) -> List[str]:
    text = text.strip()
    if not text:
      return []

    # Prefer spaCy if available
    doc = self._nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    return sents

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

  def _build_chunks_from_sentences(self, sentences: List[str]) -> List[str]:
    """
    Assemble sentence list into chunks of roughly min_tokens..max_tokens (word-count based).
    Never cuts sentences.
    """
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for sent in sentences:
      sent_len = len(sent.split())

      # Hard case: a single sentence is absurdly long
      if not cur and sent_len > self.max_tokens:
        words = sent.split()
        for i in range(0, len(words), self.max_tokens):
          sub = " ".join(words[i:i + self.max_tokens])
          chunks.append(sub.strip())
        continue

      # If adding this sentence would overflow
      if cur_len + sent_len > self.max_tokens:
        if cur:
          chunks.append(" ".join(cur).strip())
        cur = [sent]
        cur_len = sent_len
      else:
        cur.append(sent)
        cur_len += sent_len

    if cur:
      chunks.append(" ".join(cur).strip())

    # Merge too-short chunks with next when possible
    merged: List[str] = []
    buffer = ""

    for ch in chunks:
      ch_len = len(ch.split())
      if ch_len < self.min_tokens:
        buffer = (buffer + " " + ch).strip()
        if len(buffer.split()) >= self.min_tokens:
          merged.append(buffer)
          buffer = ""
      else:
        if buffer:
          merged.append(buffer)
          buffer = ""
        merged.append(ch)

    if buffer:
      merged.append(buffer)

    return merged

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
      sections: List[Dict[str, Any]] = []

      current_title: Optional[str] = None
      current_body_lines: List[str] = []
      current_level: int = 0

      def flush_current_section():
        nonlocal current_title, current_body_lines, current_level
        if current_title is None:
          current_body_lines = []
          return

        body = "\n".join(current_body_lines).strip()

        sections.append({
          "title": current_title,
          "level": current_level,
          "body": body
        })

        # Keep old "paragraphs" behavior
        paragraphs.append({
          "title": current_title,
          "body": body
        })

        current_body_lines = []

      for raw in raw_lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        # Skip empty lines globally
        if not stripped:
          continue

        if self._is_heading(stripped):
          flush_current_section()
          level = len(stripped) - len(stripped.lstrip("#"))
          title = self._clean_heading(stripped)

          current_title = title
          current_level = level
          continue

        # Non-heading, non-empty line
        file_lines.append(stripped)

        # Add to sentences
        file_sentences.extend(self._split_sentences(stripped))

        # If we are inside a paragraph, accumulate lines for its body
        if current_title is not None:
          current_body_lines.append(stripped)

      # Flush last paragraph if any
      flush_current_section()
      filtered_sections: List[Dict[str, Any]] = []

      for sec in sections:
        title = sec.get("title", "") or ""
        body = sec.get("body", "") or ""

        # Rule #3: drop useless sections like references/citations
        if self._is_blacklisted_heading(title):
          continue

        # Rule #2: remove short structural headings that have no body
        if not body.strip() and not self._looks_like_sentence_heading(title):
          continue

        filtered_sections.append(sec)

      chunk_sentences: List[str] = []
      for sec in filtered_sections:
          title = sec.get("title", "") or ""
          body = sec.get("body", "") or ""

          if self._looks_like_sentence_heading(title):
            # treat heading as a sentence-like line
            chunk_sentences.extend(self._split_sentences(title))

          if body.strip():
            # split body by lines then into sentences
            for bl in body.split("\n"):
              bl = bl.strip()
              if not bl:
                continue
              chunk_sentences.extend(self._split_sentences(bl))

      chunk_texts = self._build_chunks_from_sentences(chunk_sentences)

      # Build chunk objects ready for your event-enricher
      # Keep it simple.
      chunks = []
      for idx, ch in enumerate(chunk_texts):
        chunks.append({
          "chunk_id": idx,
          "text": ch,
          "events": []  # placeholder for your module
        })

      result = {
        "lines": file_lines,
        "sentences": file_sentences,
        "paragraphs": paragraphs,
        "sections": filtered_sections,
        "chunks": chunks
      }

      with output_json_path.open("w", encoding="utf-8") as json_file:
        json.dump(result, json_file, indent=2, ensure_ascii=False)
    
    print("Markdown division complete")
  
  def __exit__(self, exc_type, exc_value, traceback):
    print("### MarkdownDivider - exit ###")
    pass
