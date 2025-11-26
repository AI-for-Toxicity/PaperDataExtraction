import csv
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict

INPUT_CSV = "data/labels/papers.csv"
MD_DIR = "data/raw/markdown/"
OUTPUT_JSON = "data/paragraphs_scored.json"
OUTPUT_CSV = "data/papers_output.csv"

# How many digits the paper ID should be padded to (e.g. 4 -> 0001)
PAD_WIDTH = 4

# Very small manual stopword list to avoid scoring on junk words
STOPWORDS = {
    "and", "or", "the", "a", "an", "of", "to", "in", "on", "for", "with", "by",
    "as", "at", "from", "that", "this", "these", "those", "is", "are", "was",
    "were", "be", "been", "it", "its", "into", "about", "also", "both", "may",
    "can", "could", "should", "would", "will", "than", "then", "such", "per",
    "via", "within", "between", "during", "over", "under", "no", "not"
}

@dataclass
class Paragraph:
    title: Optional[str]
    text: str
    score: float = 0.0
    matched_words: List[str] = field(default_factory=list)
    matched_event: Optional[str] = None
    event_type: Optional[str] = None


def tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase and keep only alphanumeric 'words'."""
    return re.findall(r"[a-z0-9]+", text.lower())


def compute_event_score(event_text: str, paragraph_text: str) -> (float, List[str]):
    """
    Score a paragraph against a single event sentence.

    +3 for each unique non-stopword token that appears at least once.
    +0.5 for each repetition of that token beyond the first.
    """
    event_tokens = [t for t in tokenize(event_text) if t not in STOPWORDS]
    if not event_tokens:
        return 0.0, []

    para_tokens = tokenize(paragraph_text)
    if not para_tokens:
        return 0.0, []

    para_counts = Counter(para_tokens)

    score = 0.0
    matched = []

    for w in set(event_tokens):
        count = para_counts.get(w, 0)
        if count > 0:
            score += 3.0
            matched.append(w)
            if count > 1:
                score += 0.5 * (count - 1)

    return score, matched


def split_markdown_paragraphs(md_text: str) -> List[Paragraph]:
    """
    Split markdown into paragraphs:
    - Paragraph = heading + all following text until next heading.
    - First block before any heading becomes a paragraph with title=None.
    """
    lines = md_text.splitlines()
    paragraphs: List[Paragraph] = []

    current_title: Optional[str] = None
    current_lines: List[str] = []

    def flush_paragraph():
        nonlocal current_lines, current_title, paragraphs
        if current_lines or current_title is not None:
            text = "\n".join(current_lines).strip()
            paragraphs.append(Paragraph(title=current_title, text=text))
            current_lines = []

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):  # heading line
            # Flush previous paragraph (title + its body)
            flush_paragraph()
            # New heading becomes title of next paragraph
            heading_text = stripped.lstrip("#").strip()
            current_title = heading_text if heading_text else None
        else:
            current_lines.append(line)

    # Flush last paragraph
    flush_paragraph()

    # Keep the first paragraph even if title is None (as requested)
    return paragraphs


def process_aop_for_paragraphs(aop_value: str, paragraphs: List[Paragraph]) -> None:
    """
    Update Paragraph objects with score, matched_words, matched_event, event_type
    based on the given AOP string.
    """
    aop_value = aop_value.strip()
    if not aop_value or aop_value == "-":
        # No event: nothing to score
        return

    # Normalize possible ASCII arrows "->" to "→"
    normalized = aop_value.replace("->", "→")
    events = [e.strip() for e in normalized.split("→") if e.strip()]

    if not events:
        return

    multi = len(events) > 1

    for para in paragraphs:
        best_score = 0.0
        best_event = None
        best_words: List[str] = []
        best_type: Optional[str] = None

        for idx, ev in enumerate(events):
            score, words = compute_event_score(ev, para.text)
            if score > best_score:
                best_score = score
                best_event = ev
                best_words = words

                if multi:
                    if idx == 0:
                        best_type = "MIE"
                    elif idx == len(events) - 1:
                        best_type = "AO"
                    else:
                        best_type = "KE"
                else:
                    best_type = "ANY"

        if best_score > 0.0:
            para.score = best_score
            para.matched_event = best_event
            para.matched_words = best_words
            para.event_type = best_type


def main(input_csv: str, md_dir: str, output_json: str, output_csv: str):
    results_json: Dict[str, List[Dict]] = {}
    csv_rows_out = []

    with open(input_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "ID" not in reader.fieldnames or "AOP" not in reader.fieldnames:
            raise ValueError("CSV must contain 'ID' and 'AOP' columns.")

        for row in reader:
            raw_id = str(row["ID"]).strip()
            aop_value = str(row["AOP"]).strip()

            # Pad ID for filename: paper_{ID_0_PADDED}.md
            paper_id_padded = raw_id.zfill(PAD_WIDTH)
            md_folder = f"paper_{paper_id_padded}/"
            md_filename = f"paper_{paper_id_padded}.md"
            md_path = os.path.join(md_dir, md_folder, md_filename)

            # Default CSV label behavior:
            if aop_value == "-":
                label = "No label found"
            else:
                label = aop_value

            csv_rows_out.append({"ID": paper_id_padded, "label": label})

            # If md file doesn't exist, skip everything else for this row
            if not os.path.isfile(md_path):
                continue

            # Read markdown
            with open(md_path, "r", encoding="utf-8") as mdf:
                md_text = mdf.read()

            # Extract paragraphs (only title & text set for now)
            paragraphs = split_markdown_paragraphs(md_text)

            # Fill scores, matched words, event type, etc.
            process_aop_for_paragraphs(aop_value, paragraphs)

            # Filter out paragraphs with score 0
            non_zero = [p for p in paragraphs if p.score > 0.0]

            # Sort by score (descending)
            non_zero.sort(key=lambda p: p.score, reverse=True)

            # Save into global JSON structure
            results_json[paper_id_padded] = [asdict(p) for p in non_zero]

    print(f"Saving results to {output_json} and {output_csv}...")

    # Save JSON
    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(results_json, jf, indent=2, ensure_ascii=False)

    # Save result CSV
    fieldnames = ["ID", "label"]
    with open(output_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows_out:
            writer.writerow(r)


if __name__ == "__main__":
    main(INPUT_CSV, MD_DIR, OUTPUT_JSON, OUTPUT_CSV)
