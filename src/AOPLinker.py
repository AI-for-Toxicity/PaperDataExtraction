import json
import csv
from pathlib import Path

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    from difflib import SequenceMatcher
    RAPIDFUZZ_AVAILABLE = False


DATA_DIR = Path("test_data/processed/divided_markdown")
LABELS_DIR = Path("test_data/labels/aop_raw")
OUTPUT_DIR = Path("test_data/labels/scored")
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------
# Utilities
# -----------------------------

def norm(s: str) -> str:
    return " ".join(s.lower().split())

def contains_exact(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return norm(needle) in norm(haystack)

def fuzzy_score(text: str, pattern: str) -> float:
    """Return a 0..1 similarity score."""
    if not text or not pattern:
        return 0.0

    t = norm(text)
    p = norm(pattern)

    if RAPIDFUZZ_AVAILABLE:
        # partial_ratio is good for "pattern contained-ish in long text"
        pr = fuzz.partial_ratio(t, p) / 100.0
        ts = fuzz.token_set_ratio(t, p) / 100.0
        # blend: token_set helps paraphrase-y overlaps, partial helps near-substring
        return max(pr * 0.7 + ts * 0.3, ts * 0.6 + pr * 0.4)
    else:
        # fallback: rough similarity
        return SequenceMatcher(None, t, p).ratio()

def proximity_bonus(text: str, chemical: str, desc_short: str) -> float:
    """Small bonus if both appear and the text is short-ish.
       We keep it simple: bonus only if both are present exactly.
    """
    if not chemical or not desc_short:
        return 0.0
    if contains_exact(text, chemical) and contains_exact(text, desc_short):
        # likely a very strong line/sentence alignment
        return 1.0
    return 0.0

# -----------------------------
# Load events
# -----------------------------

def load_events(label_path: Path):
    events = []
    with open(label_path, "r", encoding="utf8") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if len(row) < 4:
                continue
            chemical, etype, desc_s, desc_l = row[:4]
            events.append({
                "event_id": idx,
                "chemical": chemical.strip(),
                "event_type": etype.strip(),
                "event_description_short": desc_s.strip(),
                "event_description_long": desc_l.strip()
            })
    return events

# -----------------------------
# Scoring
# -----------------------------

def compute_score(text: str, event: dict, block_kind: str) -> float:
    """
    Returns a score in 0..100 based on description matching only.
    Perfect (100) only if short description exact match.
    Chemical presence is handled separately as a boolean flag.
    """
    desc_s = event.get("event_description_short", "")
    desc_l = event.get("event_description_long", "")

    # Perfect condition: short exact
    if contains_exact(text, desc_s):
        base = 95.0
        # small sanity bonus for short blocks where exact short appears
        if block_kind in {"line", "sentence"}:
            base += 5.0
        return min(100.0, base)

    # Graded scoring based on fuzzy similarity
    fs = fuzzy_score(text, desc_s)  # 0..1
    fl = fuzzy_score(text, desc_l)  # 0..1

    # Base anchored on short
    score = 10.0
    score += 70.0 * fs
    score += 20.0 * fl

    # Optional tiny structural bonus for short blocks when short fuzzy is high
    if block_kind in {"line", "sentence"} and fs >= 0.75:
        score += 3.0

    # Cap so that non-exact short never reaches 100
    return min(99.0, score)

# -----------------------------
# Annotation per block type
# -----------------------------

def annotate_blocks(blocks, events, key_text: str, block_kind: str):
    out = []
    matched_event_ids = set()

    for block in blocks:
        text = block.get(key_text, "") or ""
        block_events = []

        for ev in events:
            chemical = ev.get("chemical", "")
            chemical_found = contains_exact(text, chemical)

            score = compute_score(text, ev, block_kind)

            # You can choose your own threshold.
            # Keep >0 to preserve "weak signals", or raise to reduce noise.
            if score > 0:
                ev_with_score = dict(ev)
                ev_with_score["score"] = round(score, 3)
                ev_with_score["chemical_found"] = chemical_found

                # OPTIONAL micro-bonus that doesn't dominate:
                # if you want strictly independent, delete these 3 lines.
                # if chemical_found:
                #     ev_with_score["score"] = round(min(99.0, ev_with_score["score"] + 1.0), 3)

                block_events.append(ev_with_score)

                # consider "matched" if either:
                # - strong description match
                # - or chemical found with decent description
                if score >= 50.0 or (chemical_found and score >= 30.0):
                    matched_event_ids.add(ev["event_id"])

        # Sort events by score desc
        block_events.sort(key=lambda x: x["score"], reverse=True)

        new_block = dict(block)
        new_block["events"] = block_events
        out.append(new_block)

    return out, matched_event_ids

# -----------------------------
# Main processing
# -----------------------------

def process_file(json_path: Path):
    base = json_path.name.replace("_divided.json", "")
    label_path = LABELS_DIR / f"{base}.txt"

    if not label_path.exists():
        print(f"[WARN] No label file for {base}")
        return

    with open(json_path, "r", encoding="utf8") as f:
        data = json.load(f)

    events = load_events(label_path)

    # Prepare blocks
    line_dicts = [{"text": ln} for ln in data.get("lines", [])]
    sent_dicts = [{"text": s} for s in data.get("sentences", [])]
    para_dicts = [{"title": p.get("title", ""), "text": p.get("body", "")}
                  for p in data.get("paragraphs", [])]

    # Annotate
    lines_annotated, matched_lines = annotate_blocks(line_dicts, events, "text", "line")
    sentences_annotated, matched_sents = annotate_blocks(sent_dicts, events, "text", "sentence")
    paragraphs_annotated, matched_paras = annotate_blocks(para_dicts, events, "text", "paragraph")

    matched_any = matched_lines | matched_sents | matched_paras

    # Unmatched events
    unmatched_events = [ev for ev in events if ev["event_id"] not in matched_any]

    output = {
        "lines": lines_annotated,
        "sentences": sentences_annotated,
        "paragraphs": paragraphs_annotated,
        "unmatched_events": unmatched_events
    }

    out_path = OUTPUT_DIR / f"{base}_events.json"
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved {out_path} | events={len(events)} matched={len(matched_any)} unmatched={len(unmatched_events)}")


def main():
    for json_path in DATA_DIR.glob("paper_*_divided.json"):
        process_file(json_path)

if __name__ == "__main__":
    main()
