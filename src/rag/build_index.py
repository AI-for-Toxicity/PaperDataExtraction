"""
Build the AOP RAG index from new_data/aop_extracted_data.json.

Run from project root:
    python src/rag/build_index.py

Output: new_data/aop_rag_index.json
Each entry:
    {
        "aop_id": str,
        "title": str,
        "stressor_names": [str, ...],   # lowercase, for exact matching
        "search_text": str,             # concatenated text fed to BM25
        "snippet": str,                 # compact multi-line block injected into prompts
    }
"""

import json
from pathlib import Path


def _build_snippet(aop: dict) -> str:
    """Compact multi-line block (~60-80 tokens) suitable for prompt injection."""
    lines = [f"[AOP] {aop['title']}"]

    stressors = aop.get("stressors") or []
    stressor_names = [s.get("name", "") for s in stressors if s.get("name")]
    if stressor_names:
        lines.append("Stressors: " + ", ".join(stressor_names))

    events = aop.get("events") or {}

    mies = events.get("MIE") or []
    mie_titles = [e["title"] for e in mies if e.get("title") and e["title"] != "N/A, Unknown"]
    if mie_titles:
        lines.append("MIE: " + " | ".join(mie_titles))

    kes = events.get("KE") or []
    ke_titles = [e["title"] for e in kes if e.get("title")]
    if ke_titles:
        lines.append("KE: " + " | ".join(ke_titles))

    aos = events.get("AO") or []
    ao_titles = [e["title"] for e in aos if e.get("title")]
    if ao_titles:
        lines.append("AO: " + " | ".join(ao_titles))

    return "\n".join(lines)


def _build_search_text(aop: dict) -> str:
    """
    Concatenated text used by BM25 for retrieval.
    Includes title, stressor names, event titles, event descriptions (if any),
    and a truncated abstract for richer matching.
    """
    parts = [aop.get("title", ""), aop.get("short_name", "")]

    stressors = aop.get("stressors") or []
    parts.extend(s.get("name", "") for s in stressors if s.get("name"))

    events = aop.get("events") or {}
    for etype in ("MIE", "KE", "AO"):
        for e in events.get(etype) or []:
            if e.get("title"):
                parts.append(e["title"])
            if e.get("description"):
                parts.append(e["description"])

    # Truncate abstract to avoid dominating the BM25 term frequencies
    abstract = aop.get("abstract") or ""
    if abstract:
        parts.append(abstract[:800])

    context = aop.get("context") or ""
    if context:
        parts.append(context[:400])

    return " ".join(p for p in parts if p)


def main():
    src = Path("new_data/aop_extracted_data.json")
    dst = Path("new_data/aop_rag_index.json")

    print(f"Loading {src} ...")
    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    index = []
    for aop in data:
        stressors = aop.get("stressors") or []
        stressor_names = [
            s.get("name", "").lower()
            for s in stressors
            if s.get("name")
        ]
        index.append({
            "aop_id": aop["aop_id"],
            "title": aop["title"],
            "stressor_names": stressor_names,
            "search_text": _build_search_text(aop),
            "snippet": _build_snippet(aop),
        })

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    n_with_stressors = sum(1 for e in index if e["stressor_names"])
    print(f"Built index: {len(index)} AOPs ({n_with_stressors} with stressors) -> {dst}")


if __name__ == "__main__":
    main()
