import re, collections
import unicodedata
import re
from pathlib import Path
from wordfreq import word_frequency
from spellchecker import SpellChecker

class MarkdownCleaner:
  def __init__(self, md_files: list, output_dir: Path, skip_existing: bool = False) -> None:
    print("### MarkdownCleaner - init ###")
    self.spell = SpellChecker(language="en")
    self.md_files = md_files
    self.output_dir = output_dir
    self.skip_existing = skip_existing

  def __enter__(self):
    return self

  '''
  Simple Levenshtein distance.
  '''
  def levenshtein(self, a: str, b: str) -> int:
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

  '''
  Find 'broken' words like 'variabi ty' where even the joined form is not valid,
  but is close to a valid dictionary word.

  Returns a list of dicts:
  {
      "original": "variabi ty",
      "joined": "variabity",
      "fixed": "variability",
      "token1": "variabi",
      "token2": "ty",
      "span": (start, end)
  }
  '''
  def find_broken_words_with_correction(self, text: str, max_distance: int = 2, min_joined_len: int = 6):
    # Tokenize into words and punctuation, keep spans
    tokens = []
    for m in re.finditer(r"\w+|[^\w\s]", text):
      tok = m.group(0)
      tokens.append({
        "text": tok,
        "start": m.start(),
        "end": m.end(),
        "is_word": bool(re.match(r"^\w+$", tok))
      })

    results = []

    for i in range(len(tokens) - 1):
      t1 = tokens[i]
      t2 = tokens[i + 1]

      if not (t1["is_word"] and t2["is_word"]):
        continue

      # Must be separated by whitespace only
      gap = text[t1["end"]:t2["start"]]
      if not gap or not gap.isspace():
        continue

      w1 = t1["text"]
      w2 = t2["text"]
      joined = (w1 + w2).lower()

      if len(joined) < min_joined_len:
        continue

      # At least one fragment should look "wrong"
      unknown1 = w1.lower() not in self.spell
      unknown2 = w2.lower() not in self.spell
      if not (unknown1 or unknown2):
        # both look like valid words → probably not a broken word case
        continue

      # If joined is already a valid word, this is the previous simpler case,
      # but your problem is when it is NOT, so skip if it's valid.
      if joined in self.spell:
        continue

      # Get best correction candidate
      candidate = self.spell.correction(joined)
      if not candidate:
        continue

      # Require small edit distance
      dist = self.levenshtein(joined, candidate)
      if dist <= max_distance:
        results.append({
          "original": text[t1["start"]:t2["end"]],
          "joined": joined,
          "fixed": candidate,
          "token1": w1,
          "token2": w2,
          "span": (t1["start"], t2["end"])
        })

    return results

  '''
  Find 'broken words' like 'variabi ty' in a text.

  Logic:
  - Take adjacent word tokens separated ONLY by whitespace.
  - Join them: w_join = w1 + w2
  - If w_join is a reasonably frequent word in the dictionary
    and at least one of w1/w2 is rare, treat it as a broken word.

  Returns list of dicts:
  {
      "original": "variabi ty",
      "fixed": "variability",
      "token1": "variabi",
      "token2": "ty",
      "span": (start, end)
      }
  '''
  def find_broken_words(self, text: str, lang: str = "en", min_joined_len: int = 6, min_joined_freq: float = 1e-8):
    # Tokenize: words + punctuation, keep spans
    tokens = []
    for m in re.finditer(r"\w+|[^\w\s]", text):
      tok = m.group(0)
      tokens.append({
        "text": tok,
        "start": m.start(),
        "end": m.end(),
        "is_word": bool(re.match(r"^\w+$", tok))
      })

    results = []

    for i in range(len(tokens) - 1):
      t1 = tokens[i]
      t2 = tokens[i + 1]

      # Need word + word
      if not (t1["is_word"] and t2["is_word"]):
        continue

      # Make sure they are separated only by whitespace
      gap = text[t1["end"]:t2["start"]]
      if not gap or not gap.isspace():
        continue

      w1 = t1["text"]
      w2 = t2["text"]
      joined = (w1 + w2).lower()

      if len(joined) < min_joined_len:
        continue

      # Frequencies
      f1 = word_frequency(w1.lower(), lang)
      f2 = word_frequency(w2.lower(), lang)
      fj = word_frequency(joined, lang)

      # Heuristic:
      # - joined word must be reasonably common
      # - at least one of the components must be rare
      if fj >= min_joined_freq and (f1 < fj * 0.01 or f2 < fj * 0.01):
        results.append({
          "original": text[t1["start"]:t2["end"]],
          "fixed": joined,
          "token1": w1,
          "token2": w2,
          "span": (t1["start"], t2["end"])
        })

    return results

  '''
  Return text with broken words merged.
  '''
  def fix_broken_words(self, text: str, lang: str = "en", min_joined_len: int = 6, min_joined_freq: float = 1e-8) -> str:
    corrections = self.find_broken_words(
      text,
      lang=lang,
      min_joined_len=min_joined_len,
      min_joined_freq=min_joined_freq,
    )

    if not corrections:
      return text

    # Apply from right to left to avoid shifting spans
    corrected = text
    for c in sorted(corrections, key=lambda x: x["span"][0], reverse=True):
      start, end = c["span"]
      corrected = corrected[:start] + c["fixed"] + corrected[end:]

    return corrected

  def guess_repeated_lines(self, lines, at_end=True, min_len=8, min_freq=0.15):
    spots = [ln.strip() for ln in lines if len(ln.strip())>=min_len]
    # footer/header candidates are lines that repeat a lot
    counter = collections.Counter(spots)
    total = len(lines)
    reps = {k for k,v in counter.items() if v/total >= min_freq}
    return reps

  def dehyphenate(self, text):
    # join foo-\nbar -> foobar, keep hyphenated terms like “endocrine-related” within a line
    return re.sub(r'(\w)-\n(\w)', r'\1\2', text)

  def fix_linebreaks(self, text):
    # join mid-sentence line breaks into spaces, keep paragraph breaks
    text = re.sub(r'([^\.\?\!:])\n(?!\n)', r'\1 ', text)
    return text

  def remove_citations(self, text):
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

  def normalize_coding(self, text):
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

  def strip_html_comments(self, text):
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

  def strip_artifacts(self, text):
    lines = []
    cruft_re = re.compile(r"^[\-\=\·\•\*一、]+$")
    for line in text.splitlines():
      if cruft_re.match(line.strip()):
        continue
      lines.append(line)
    text = "\n".join(lines)
    return text

  def strip_various(self, text):
    # Remove lost 1 or 2 digit integer numbers with no measure unit after them (e.g., " 5 ", " 12 ")
    # Check if a measure unit (like "mg", "cm", etc.) is present within 5 words before or after the number
    
    # Remove spaces before punctuation and multiple spaces
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r" {2,}", " ", text)
    return text

  def normalize_headings(self, text: str) -> str:
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

  def fix_whitespace(self, text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip every line
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(lines)

  def join_broken_lines(self, text: str) -> str:
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

  '''
  Remove lines that contain only one visible character (e.g. 'a', '.', '#', etc.)
  Empty lines are preserved only if you want them; change as needed.
  '''
  def remove_meaningless_lines(self, text: str) -> str:
    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()
        # keep only lines that are longer than 1 visible character
        if len(stripped) > 1 or stripped == "":
            cleaned.append(line)
    return "\n".join(cleaned)

  '''
  Remove lines that are basically just a link (optionally with a leading number).
  Examples removed:
    '4  http://www.epa.gov/ncct/toxcast'
    '1 https://doi.org/10.1016/j.taap.2019.114876'
    'www.example.com'
  '''
  def remove_bare_link_lines(self, text: str) -> str:
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

  '''
  Remove dashes at the end of words when they are immediately followed by
  a space or punctuation.

  Examples:
    'apple- .'   -> 'apple .'
    'apple-.'    -> 'apple.'
    'apple- ,'   -> 'apple ,'
    'apple-,'    -> 'apple,'
    'apple- and' -> 'apple and'
  '''
  def remove_trailing_dashes(self, text: str) -> str:
    # (\w)          capture last word character
    # -            literal dash
    # (?=\s|[.,;:!?])  only if next char is whitespace or punctuation
    return re.sub(r'(\w)-(?=\s|[.,;:!?])', r'\1', text)

  '''
  Remove GitHub-style markdown tables from a text.

  A table is detected as:
  - a header line starting with '|' 
  - followed by a separator line made of |, -, :, + and spaces
  - followed by 0+ lines starting with '|'

  The entire block is removed.
  '''
  def remove_markdown_tables(self, text: str) -> str:
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

  def inspect_lines(self, text: str, indices):
    lines = text.splitlines()
    for i in indices:
      if 0 <= i < len(lines):
        line = lines[i]
        print(f"Line {i}: {repr(line)}")
        print("  chars:", [f"{c} (U+{ord(c):04X})" for c in line])
      else:
        print(f"Line {i}: <out of range>")

  def clean_file(self, md_text: str) -> str:
    # lines = md_text.splitlines()
    # repeated = guess_repeated_lines(lines)
    # lines = [ln for ln in lines if ln.strip() not in repeated]
    # text = "\n".join(lines)
    text = self.normalize_coding(md_text)
    text = self.strip_html_comments(text)
    text = self.remove_meaningless_lines(text)
    text = self.remove_bare_link_lines(text)
    text = self.strip_artifacts(text)
    text = self.remove_markdown_tables(text)
    text = self.dehyphenate(text)
    text = self.fix_linebreaks(text)
    text = self.remove_citations(text)
    text = self.fix_whitespace(text)
    text = self.strip_various(text)
    text = self.normalize_headings(text)
    text = self.join_broken_lines(text)
    text = self.join_broken_lines(text)
    text = self.remove_trailing_dashes(text)

    #print(find_broken_words(text))

    return text.strip()

  def clean_markdowns(self, folder="cleaned_markdown"):
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

  def __exit__(self, exc_type, exc_value, traceback):
    print("### MarkdownCleaner - exit ###")
    pass
