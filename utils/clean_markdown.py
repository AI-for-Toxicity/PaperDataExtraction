# src/clean_markdown.py
import re, sys, pathlib, collections

INPUT_PATH = "data/raw/markdown/"
OUTPUT_PATH = "data/processed/"

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
    return text

def clean(md_text):
    lines = md_text.splitlines()
    repeated = guess_repeated_lines(lines)
    # drop repeated header/footer lines
    lines = [ln for ln in lines if ln.strip() not in repeated]
    text = "\n".join(lines)
    text = dehyphenate(text)
    text = fix_linebreaks(text)
    text = remove_citations(text)
    # collapse absurd whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

if __name__ == "__main__":
    i = 0
    base = pathlib.Path(INPUT_PATH)
    files = sorted([p for p in base.glob("*/*.md") if p.is_file()])
    for md_file in files:
        i += 1
        inp = md_file
        outp = pathlib.Path(OUTPUT_PATH) / md_file.name
        outp.write_text(clean(inp.read_text(encoding="utf-8", errors="ignore")), encoding="utf-8")
        print(f"[{i}/{len(files)}] Cleaned {md_file.name} -> {outp}")

