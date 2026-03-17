"""
This module implements a class that takes a list of markdown files and an output directory, and cleans each markdown file by stripping various boilerplate elements, normalizing whitespace, and applying heuristics to remove uninformative sections (like references).
The cleaning process is designed to prepare the markdown files for subsequent division and processing steps.
The cleaned markdown files are saved in the output directory with a "_cleaned.md" suffix.
"""

import re
import unicodedata
import re
from pathlib import Path
from spellchecker import SpellChecker

class MarkdownCleaner:
  # INITIALIZATION

  def __init__(self, md_files: list, output_dir: Path, skip_existing: bool = False) -> None:
    print("### MarkdownCleaner - init ###")
    self.spell = SpellChecker(language="en")
    self.md_files = md_files
    self.output_dir = output_dir
    self.skip_existing = skip_existing

  def __enter__(self):
    return self

  # CLEANING HELPERS

  def _levenshtein(self, a: str, b: str) -> int:
    """
    Simple _levenshtein distance.
    """
    if a == b:
      return 0
    if len(a) < len(b):
      a, b = b, a
    # now len(a) >= len(b)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
      current = [i]
      for j, cb in enumerate(b, 1):
        insert_cost = current[j-1] + 1
        delete_cost = previous[j] + 1
        replace_cost = previous[j-1] + (ca != cb)
        current.append(min(insert_cost, delete_cost, replace_cost))
      previous = current
    return previous[-1]

  def _dehyphenate(self, text):
    """
    Join foo-\nbar -> foobar, keep hyphenated terms like “endocrine-related” within a line
    """
    return re.sub(r'(\w)-\n(\w)', r'\1\2', text)

  def _fix_linebreaks(self, text):
    """
    Join mid-sentence line breaks into spaces, keep paragraph breaks
    """
    text = re.sub(r'([^\.\?\!:])\n(?!\n)', r'\1 ', text)
    return text

  def _remove_citations(self, text):
    """
    Remove citations like [1], (Smith et al., 2020), etc.
    """
    text = re.sub(
      r"\(\s*(?:[A-Z][A-Za-z'\-]+(?:\s+et\s+al\.)?"
      r"(?:,\s*(?:19|20)\d{2})"
      r"(?:\s*;\s*[A-Z][A-Za-z'\-]+(?:\s+et\s+al\.)?(?:,\s*(?:19|20)\d{2}))*"
      r")\s*\)",
      "",
      text,
    )
    # Remove bracketed citations [1], [2-4], [1, 3, 5]
    bracket_cite_re = re.compile(
      r"\[(?:\s*\d+\s*(?:-\s*\d+)?\s*(?:,\s*\d+\s*(?:-\s*\d+)?\s*)*)\]"
    )
    text = bracket_cite_re.sub("", text)

    # Remove citations in parentheses (Surname et al., 2024)
    paren_cite_re = re.compile(
      r"\((?:[^()]*?(?:et\s+al\.?|(?:19|20)\d{2})[^()]*)\)",
      flags=re.IGNORECASE
    )
    text = paren_cite_re.sub("", text)
    return text

  def _normalize_coding(self, text):
    """
    Normalize unicode characters, fix common OCR artifacts, and remove unrecognized control characters.
    """
    norm = unicodedata.normalize("NFC", text)
    # Fix dashes and micro sign
    dash_map = {
      "\u2013": "-", "\u2014": "-", "\u2212": "-",
      "\u2012": "-", "\u2015": "-"
    }
    for k, v in dash_map.items():
      norm = norm.replace(k, v)
    norm = norm.replace("μ", "µ")
    # Space before and after -> no space
    norm = norm.replace(' /uniFB00 ', 'ff')
    norm = norm.replace(' /uniFB01 ', 'fi')
    norm = norm.replace(' /uniFB02 ', 'fl')
    norm = norm.replace(' /uniFB03 ', 'ffi')
    norm = norm.replace(' /uniFB04 ', 'ffl')
    norm = norm.replace(' /uniFB05 ', 'ft')
    norm = norm.replace(' /uniFB06 ', 'st')
    # Space before only -> space before (except ff, ffi, ffl because can't be start of word)
    norm = norm.replace(' /uniFB00', 'ff')
    norm = norm.replace(' /uniFB01', ' fi')
    norm = norm.replace(' /uniFB02', ' fl')
    norm = norm.replace(' /uniFB03', 'ffi')
    norm = norm.replace(' /uniFB04', 'ffl')
    norm = norm.replace(' /uniFB05', ' ft')
    norm = norm.replace(' /uniFB06', ' st')
    # Space after only -> space before (docling artifact)
    norm = norm.replace('/uniFB00 ', ' ff')
    norm = norm.replace('/uniFB01 ', ' fi')
    norm = norm.replace('/uniFB02 ', ' fl')
    norm = norm.replace('/uniFB03 ', ' ffi')
    norm = norm.replace('/uniFB04 ', ' ffl')
    norm = norm.replace('/uniFB05 ', ' ft')
    norm = norm.replace('/uniFB06 ', ' st')
    # No spaces to clean any remaining
    norm = norm.replace('/uniFB00', 'ff')
    norm = norm.replace('/uniFB01', 'fi')
    norm = norm.replace('/uniFB02', 'fl')
    norm = norm.replace('/uniFB03', 'ffi')
    norm = norm.replace('/uniFB04', 'ffl')
    norm = norm.replace('/uniFB05', 'ft')
    norm = norm.replace('/uniFB06', 'st')

    norm = norm.replace('&amp;', '&')
    norm = norm.replace('&lt;', '<')
    norm = norm.replace('&gt;', '>')
    norm = norm.replace('&le;', '<=')
    norm = norm.replace('&ge;', '>=')

    # Find any other unrecognized control characters and remove them
    for c in norm:
      if unicodedata.category(c) in ("Cc", "Cf") and c not in ("\n", "\t", "\r"):
        print(f"Found control character U+{ord(c):04X} in text, removing it.")
    norm = "".join(c for c in norm if not (unicodedata.category(c) in ("Cc", "Cf") and c not in ("\n", "\t", "\r")))

    return norm

  def _strip_html_comments(self, text):
    """
    Remove Docling's HTML comments like <!-- This is a comment -->
    """
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

  def _strip_artifacts(self, text):
    """
    Remove lines that consist solely of common OCR artifacts or decorative characters.
    """
    lines = []
    cruft_re = re.compile(r"^[\-\=\·\•\*一、]+$")
    for line in text.splitlines():
      if cruft_re.match(line.strip()):
        continue
      lines.append(line)
    text = "\n".join(lines)
    return text

  def _strip_various(self, text):
    """
    Remove lost 1 or 2 digit integer numbers with no measure unit after them (e.g., " 5 ", " 12 ")
    Check if a measure unit (like "mg", "cm", etc.) is present within 5 words before or after the number
    Remove spaces before punctuation and multiple spaces
    """
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r" {2,}", " ", text)
    return text

  def _normalize_headings(self, text: str) -> str:
    """
    Normalize headings by ensuring proper spacing and formatting.
    """
    lines = text.splitlines()
    out = []

    for i, line in enumerate(lines):
      is_heading = line.lstrip().startswith("#")
      if is_heading:
        # if it's not the first line and the previous line is not empty, add a blank line
        if i > 0 and lines[i - 1].strip() != "":
          out.append("")  # blank line
      out.append(line)

    # Make lines that start with "Fig. 1:" or "Figure 2:" headings
    fig_re = re.compile(r"^(Fig\.|Figure)\s+\d+:")
    tab_re = re.compile(r"^(Tab\.|Table)\s+\d+:")
    fig2_re = re.compile(r"^(FIGURE)\s*(\d+)\s*\|\s*(.+)$")
    tab2_re  = re.compile(r"^(TABLE)\s*(\d+)\s*\|\s*(.+)$")
    final_out = []

    for line in out:
      if fig_re.match(line) or tab_re.match(line):
        line = f"## {line}"
      if fig2_re.match(line):
        line = fig2_re.sub(r"## FIGURE \2\n\3", line)
      if tab2_re.match(line):
        line = tab2_re.sub(r"## TABLE \2\n\3", line)
      final_out.append(line)
    text = "\n".join(final_out)

    # Remove newlines immediately after headings and before the next non-empty line (not if another heading)
    cleaned_out = []
    skip_next_blank = False
    for i, line in enumerate(text.splitlines()):
      if skip_next_blank:
        if line.strip() == "":
          skip_next_blank = False
          continue
        else:
          skip_next_blank = False

      cleaned_out.append(line)

      if line.lstrip().startswith("#"):
        # look ahead for the next non-empty line
        j = i + 1
        while j < len(text.splitlines()) and text.splitlines()[j].strip() == "":
          j += 1
        if j < len(text.splitlines()):
          next_line = text.splitlines()[j]
          if not next_line.lstrip().startswith("#"):
            skip_next_blank = True

    return "\n".join(cleaned_out)

  def _fix_whitespace(self, text: str) -> str:
    """
    Fix excessive whitespace by reducing multiple newlines to a maximum of two and stripping leading/trailing spaces from each line.
    """
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip every line
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(lines)

  def _join_broken_lines(self, text: str) -> str:
    """
    Join lines that were broken in the middle of a sentence due to OCR or formatting issues.
    """
    lines = text.splitlines()
    out = []
    i = 0

    def is_heading(line: str) -> bool:
      s = line.lstrip()
      return bool(re.match(r'^#{1,6}\s', s))

    end_punct = (".", ":", ";", "!", "?")

    while i < len(lines):
      line = lines[i].rstrip()

      # find next non-empty line (could be several blank lines ahead)
      j = i + 1
      while j < len(lines) and lines[j].strip() == "":
        j += 1

      # if there is no next non-empty line, just push current and stop
      if j >= len(lines):
        out.append(line)
        break

      next_line = lines[j]

      should_join = (
        line.strip() != ""                       # current not empty
        and not line.endswith(end_punct)         # no punctuation at end
        and not is_heading(next_line)            # next non-empty not a heading
        and not is_heading(line)                 # current not a heading
      )

      if should_join:
        if line.strip().endswith("-"):
          # remove hyphen and join directly
          merged = line[:-1].rstrip() + next_line.lstrip()
        else:
          # merge current with next non-empty, remove all blanks in between
          merged = line + " " + next_line.lstrip()
        out.append(merged)
        i = j + 1  # skip over the consumed next line
      else:
        out.append(line)
        i += 1

    return "\n".join(out)

  def _remove_meaningless_lines(self, text: str) -> str:
    """
    Remove lines that contain only one visible character (e.g. 'a', '.', '#', etc.)
    Empty lines are preserved only if you want them; change as needed.
    """
    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()
        # keep only lines that are longer than 1 visible character
        if len(stripped) > 1 or stripped == "":
            cleaned.append(line)
    return "\n".join(cleaned)

  def _remove_bare_link_lines(self, text: str) -> str:
    """
    Remove lines that are basically just a link (optionally with a leading number).
    Examples removed:
      '4  http://www.epa.gov/ncct/toxcast'
      '1 https://doi.org/10.1016/j.taap.2019.114876'
      'www.example.com'
    """
    BARE_LINK_RE = re.compile(
      r"""^
        \s*
        \d*                # optional leading number (e.g. '4')
        \s*
        (                  # the "link"
        https?://\S+     # http or https URL
        | www\.\S+       # or starts with www.
        | \S+\.\S+       # or generic domain-like token (foo.bar)
        )
        \s*
        $
      """,
      re.VERBOSE | re.IGNORECASE,
    )

    cleaned = []
    for line in text.splitlines():
      if BARE_LINK_RE.match(line):
        continue
      cleaned.append(line)
    return "\n".join(cleaned)

  def _remove_trailing_dashes(self, text: str) -> str:
    """
    Remove dashes at the end of words when they are immediately followed by
    a space or punctuation.

    Examples:
      'apple- .'   -> 'apple .'
      'apple-.'    -> 'apple.'
      'apple- ,'   -> 'apple ,'
      'apple-,'    -> 'apple,'
      'apple- and' -> 'apple and'
    """
    # (\w)          capture last word character
    # -            literal dash
    # (?=\s|[.,;:!?])  only if next char is whitespace or punctuation
    return re.sub(r'(\w)-(?=\s|[.,;:!?])', r'\1', text)

  def _remove_markdown_tables(self, text: str) -> str:
    """
    Remove GitHub-style markdown tables from a text.

    A table is detected as:
    - a header line starting with '|' 
    - followed by a separator line made of |, -, :, + and spaces
    - followed by 0+ lines starting with '|'

    The entire block is removed.
    """
    lines = text.splitlines()
    out_lines = []
    i = 0
    n = len(lines)

    def is_table_sep_line(line: str) -> bool:
      s = line.strip()
      if not (s.startswith("|") and s.endswith("|")):
        return False
      # Remove edge '|' and check only valid chars remain
      inner = s[1:-1].strip()
      return bool(inner) and all(c in "-:|+ " for c in inner)

    while i < n:
      line = lines[i]
      stripped = line.lstrip()

      # Possible table header
      if stripped.startswith("|") and i + 1 < n and is_table_sep_line(lines[i + 1]):
        # Skip header + separator
        i += 2
        # Skip all following rows that look like table rows (start with '|')
        while i < n and lines[i].lstrip().startswith("|"):
          i += 1
        # Done skipping the table block, do not add anything to out_lines
        continue
      else:
        out_lines.append(line)
        i += 1

    return "\n".join(out_lines)

  # FULL CLEANING PIPELINE

  def clean_file(self, md_text: str) -> str:
    """
    Run cleaning pipeline on a single markdown file.
    """
    text = self._normalize_coding(md_text)
    text = self._strip_html_comments(text)
    text = self._remove_meaningless_lines(text)
    text = self._remove_bare_link_lines(text)
    text = self._strip_artifacts(text)
    text = self._remove_markdown_tables(text)
    text = self._dehyphenate(text)
    text = self._fix_linebreaks(text)
    text = self._remove_citations(text)
    text = self._fix_whitespace(text)
    text = self._strip_various(text)
    text = self._normalize_headings(text)
    text = self._join_broken_lines(text)
    text = self._join_broken_lines(text)
    text = self._remove_trailing_dashes(text)

    return text.strip()

  def clean_markdowns(self, folder="cleaned_markdown"):
    """
    Run the full cleaning pipeline on all markdown files and save cleaned versions.
    """
    total = len(self.md_files)
    print(f"Found {total} markdown files to process for markdown cleanup.\n")

    for i, md_file in enumerate(self.md_files, 1):
      print(f"[{i}/{total}] Processing {md_file.name}...")
      parts = md_file.stem.split("_")
      name = "_".join(parts[:2]) if len(parts) > 1 else parts[0]
      out_dir = self.output_dir / folder
      out_dir.mkdir(parents=True, exist_ok=True)
      out_path = out_dir / f"{name}.md"

      if self.skip_existing and out_path.exists():
        print(f"Skipping {name} because {out_path} exists.")
        continue

      md_text: str = Path(md_file).read_text(encoding="utf-8", errors="ignore")
      with out_path.open("w", encoding="utf-8") as out:
        out.write(self.clean_file(md_text))

    print("Markdown cleanup complete")

  # EXIT

  def __exit__(self, exc_type, exc_value, traceback):
    print("### MarkdownCleaner - exit ###")
    pass
