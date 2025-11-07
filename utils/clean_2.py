from pathlib import Path
import re, json, unicodedata

### Configuration
INPUT = Path("data/processed/step_1/")
OUTPUT = Path("data/processed/step_2/")
###

# Create output directories
OUTPUT.mkdir(parents=True, exist_ok=True)

# Loop over files
i = 0
original_files = list(INPUT.glob("*.md"))
for original in original_files:
  i += 1 
  print(f"[{i}/{len(original_files)}] Processing {original.name}...")

  # Load text
  text = original.read_text(encoding="utf-8", errors="ignore")
  orig_len = len(text)

  # Normalize Unicode
  norm = unicodedata.normalize("NFC", text)

  # Fix dashes and micro sign
  dash_map = {
      "\u2013": "-", "\u2014": "-", "\u2212": "-",
      "\u2012": "-", "\u2015": "-"
  }
  for k, v in dash_map.items():
      norm = norm.replace(k, v)
  norm = norm.replace("μ", "µ")

  # Strip HTML comments
  norm = re.sub(r"<!--.*?-->", "", norm, flags=re.DOTALL)

  # Drop page artifacts like "-----" or "====="
  lines = []
  cruft_re = re.compile(r"^[\-\=\·\•\*一、]+$")
  for line in norm.splitlines():
      if cruft_re.match(line.strip()):
          continue
      lines.append(line)
  norm = "\n".join(lines)

  # Remove bracketed citations [1], [2-4], [1, 3, 5]
  bracket_cite_re = re.compile(
      r"\[(?:\s*\d+\s*(?:-\s*\d+)?\s*(?:,\s*\d+\s*(?:-\s*\d+)?\s*)*)\]"
  )
  norm = bracket_cite_re.sub("", norm)

  # Remove citations in parentheses (Surname et al., 2024)
  paren_cite_re = re.compile(
      r"\((?:[^()]*?(?:et\s+al\.?|(?:19|20)\d{2})[^()]*)\)",
      flags=re.IGNORECASE
  )
  norm = paren_cite_re.sub("", norm)

  # Write output
  output_file_path = OUTPUT / original.name
  output_file_path.write_text(norm, encoding="utf-8")

  print(f"\tCleaned {orig_len:,} chars → {len(norm):,} chars.")

print(f"Removed citations, normalized Unicode, output files stored in {OUTPUT}.")
