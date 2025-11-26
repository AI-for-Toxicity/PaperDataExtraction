import os
import re
import json
from pathlib import Path

# --------------------------
# CONFIG
# --------------------------
CREATE_FILES = True  # toggle as needed
RAW_DIR = "data/raw/pdfs"
LABELS_DIR = "data/labels/nbllm"
JSON_OUT = "data/labels/annotations.json"

# pattern: "chemical","event_type","event_description"
TRIPLET_RE = re.compile(
    r'"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"'
)

# --------------------------
# UTIL
# --------------------------
def extract_id_from_pdf_name(name: str) -> str:
    # paper_XXXX_other.pdf OR paper_XXXX.pdf
    m = re.match(r"paper_(\d+)", name)
    return m.group(1) if m else None

def ensure_labels_files(raw_dir, labels_dir):
    os.makedirs(labels_dir, exist_ok=True)
    for fname in os.listdir(raw_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        pid = extract_id_from_pdf_name(fname)
        if not pid:
            continue
        out = Path(labels_dir) / f"paper_{pid}.txt"
        if not out.exists():
            out.write_text("", encoding="utf8")

def parse_triplets(text: str):
    return TRIPLET_RE.findall(text)

def build_json(labels_dir):
    data = {}
    for fname in os.listdir(labels_dir):
        if not fname.lower().endswith(".txt"):
            continue
        pid = extract_id_from_pdf_name(fname)
        if not pid:
            continue

        text = Path(labels_dir, fname).read_text(encoding="utf8")
        hits = parse_triplets(text)

        # structure: chemicals -> events
        chem_map = {}
        order_counters = {}

        for chem, etype, elong in hits:
            if chem not in chem_map:
                chem_map[chem] = []
                order_counters[chem] = 1

            chem_map[chem].append({
                "event_type": etype,
                "event_long": elong,
                "event_short": None,
                "order": order_counters[chem]
            })
            order_counters[chem] += 1

        data[f"paper_{pid}"] = {
            "id": pid,
            "chemicals": [
                {
                    "name": chem,
                    "events": events
                }
                for chem, events in chem_map.items()
            ]
        }

    return data

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    if CREATE_FILES:
        ensure_labels_files(RAW_DIR, LABELS_DIR)
    else:
        dataset = build_json(LABELS_DIR)
        with open(JSON_OUT, "w", encoding="utf8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
