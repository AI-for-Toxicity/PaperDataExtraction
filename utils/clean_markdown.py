# src/clean_markdown.py
import re, sys, pathlib, collections
import unicodedata

INPUT_PATH = "data/raw/markdown/0_test"
OUTPUT_PATH = "data/raw/markdown/0_test/output"

def guess_repeated_lines(lines, at_end=True, min_len=8, min_freq=0.15):
    spots = [ln.strip() for ln in lines if len(ln.strip())>=min_len]
    # footer/header candidates are lines that repeat a lot
    counter = collections.Counter(spots)
    total = len(lines)
    reps = {k for k,v in counter.items() if v/total >= min_freq}
    return reps

def dehyphenate(text):
    # join foo-\nbar -> foobar, keep hyphenated terms like “endocrine-related” within a line
    return re.sub(r'(\w)-\n(\w)', r'\1\2', text)

def fix_linebreaks(text):
    # join mid-sentence line breaks into spaces, keep paragraph breaks
    text = re.sub(r'([^\.\?\!:])\n(?!\n)', r'\1 ', text)
    return text

def remove_citations(text):
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

def normalize_coding(text):
    norm = unicodedata.normalize("NFC", text)
    # Fix dashes and micro sign
    dash_map = {
        "\u2013": "-", "\u2014": "-", "\u2212": "-",
        "\u2012": "-", "\u2015": "-"
    }
    for k, v in dash_map.items():
        norm = norm.replace(k, v)
    norm = norm.replace("μ", "µ")
    return norm

def strip_html_comments(text):
  return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

def strip_artifacts(text):
    lines = []
    cruft_re = re.compile(r"^[\-\=\·\•\*一、]+$")
    for line in text.splitlines():
        if cruft_re.match(line.strip()):
            continue
        lines.append(line)
    text = "\n".join(lines)
    return text

def strip_various(text):
    # Remove lost 1 or 2 digit integer numbers with no measure unit after them (e.g., " 5 ", " 12 ")
    # Check if a measure unit (like "mg", "cm", etc.) is present within 5 words before or after the number
    
    # Remove spaces before punctuation and multiple spaces
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r" {2,}", " ", text)
    return text

def normalize_headings(text: str) -> str:
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
    fig_re = re.compile(r"^(Fig\.|Figure)\s+\d+:", re.IGNORECASE)
    final_out = []

    for line in out:
        if fig_re.match(line):
            line = f"## {line}"
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

def fix_whitespace(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip every line
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(lines)

def join_broken_lines(text: str) -> str:
    lines = text.splitlines()
    out = []
    i = 0

    def is_heading(line: str) -> bool:
        return line.lstrip().startswith("#")

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
            # merge current with next non-empty, remove all blanks in between
            merged = line + " " + next_line.lstrip()
            out.append(merged)
            i = j + 1  # skip over the consumed next line
        else:
            out.append(line)
            i += 1

    return "\n".join(out)

def clean(md_text):
    # lines = md_text.splitlines()
    # repeated = guess_repeated_lines(lines)
    # lines = [ln for ln in lines if ln.strip() not in repeated]
    # text = "\n".join(lines)
    text = normalize_coding(md_text)
    text = strip_html_comments(text)
    text = strip_artifacts(text)
    text = dehyphenate(text)
    text = fix_linebreaks(text)
    text = remove_citations(text)
    text = fix_whitespace(text)
    text = strip_various(text)
    text = normalize_headings(text)
    text = join_broken_lines(text)
    return text.strip()

if __name__ == "__main__":
    i = 0
    base = pathlib.Path(INPUT_PATH)
    out = pathlib.Path(OUTPUT_PATH)
    out.mkdir(parents=True, exist_ok=True)
    # Delete output contents
    for f in out.glob("*"):
        if f.is_file():
            f.unlink()
    files = sorted([p for p in base.glob("*/*.md") if p.is_file()])
    print(f"Found {len(files)} markdown files to clean.")
    for md_file in files:
        i += 1
        inp = md_file
        outp = out / md_file.name
        outp.write_text(clean(inp.read_text(encoding="utf-8", errors="ignore")), encoding="utf-8")
        print(f"[{i}/{len(files)}] Cleaned {md_file.name} -> {outp}")

