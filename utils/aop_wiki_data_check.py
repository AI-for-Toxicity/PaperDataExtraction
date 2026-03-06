import json
import re
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from rapidfuzz import fuzz
RAPIDFUZZ = True


_WS_RE = re.compile(r"\s+")
_PUNCT_SPACE_RE = re.compile(r"\s+([,.;:!?])")


def normalize_text(s: str) -> str:
    """Lowercase + collapse whitespace; keep punctuation because titles often include commas."""
    s = s or ""
    s = s.replace("\u00a0", " ")  # nbsp
    s = _WS_RE.sub(" ", s).strip()
    s = _PUNCT_SPACE_RE.sub(r"\1", s)
    return s.lower()


def safe_join_text(aop: Dict[str, Any]) -> str:
    abstract = aop.get("abstract") or ""
    context = aop.get("context") or ""
    return normalize_text(f"{abstract}\n{context}")


def candidates_for_event(ev: Dict[str, Any]) -> List[str]:
    # Either title OR short_name is ok (you can include both as candidates)
    cands = []
    t = ev.get("title")
    s = ev.get("short_name")
    if t: cands.append(t)
    if s: cands.append(s)
    # de-dup
    out = []
    seen = set()
    for x in cands:
        nx = normalize_text(x)
        if nx and nx not in seen:
            seen.add(nx)
            out.append(x)
    return out


def match_in_text(
    haystack_norm: str,
    needle_raw: str,
    *,
    mode: str = "exact",
    fuzzy_threshold: int = 90,
    min_len: int = 4
) -> bool:
    """
    mode:
      - exact: substring match after normalization
      - fuzzy: token_set_ratio against sliding chunks (best-effort) if RapidFuzz is available
    """
    if not needle_raw:
        return False
    needle_norm = normalize_text(needle_raw)
    if len(needle_norm) < min_len:
        return False

    if mode == "exact" or not RAPIDFUZZ:
        return needle_norm in haystack_norm

    # Fuzzy mode: use token_set_ratio against the whole text first (cheap)
    score_whole = fuzz.token_set_ratio(needle_norm, haystack_norm)
    if score_whole >= fuzzy_threshold:
        return True

    # Then try a lightweight window scan around occurrences of a rare token
    tokens = [t for t in re.split(r"\W+", needle_norm) if t]
    if not tokens:
        return False
    anchor = min(tokens, key=len)  # not perfect, but cheap
    idx = haystack_norm.find(anchor)
    if idx == -1:
        return False

    # Evaluate a window around the anchor
    start = max(0, idx - 400)
    end = min(len(haystack_norm), idx + 400)
    window = haystack_norm[start:end]
    return fuzz.token_set_ratio(needle_norm, window) >= fuzzy_threshold


def iter_applicability_names(aop: Dict[str, Any], key: str, field_name: str) -> List[str]:
    """
    applicability section items look like:
      {"sex": [{"sex":"Female","evidence":"High"}], ...}
      {"taxonomy":[{"name":"human","evidence":"Moderate"}, ...]}
      {"life_stage":[{"life_stage":"Adult,...","evidence":"Moderate"}]}
    """
    app = aop.get("applicability") or {}
    items = app.get(key) or []
    out = []
    for it in items:
        if isinstance(it, dict):
            v = it.get(field_name)
            if v:
                out.append(v)
        elif isinstance(it, str):
            out.append(it)
    return out


def audit_aops(
    aops: List[Dict[str, Any]],
    *,
    mode: str = "exact",
    fuzzy_threshold: int = 90,
    verbose: bool = False
) -> Dict[str, Any]:
    stats = {
        "aops_total": 0,

        "stressors_checked": 0,
        "stressors_found": 0,

        "events_checked": 0,
        "events_found": 0,

        "sex_checked": 0,
        "sex_found": 0,

        "taxonomy_checked": 0,
        "taxonomy_found": 0,

        "life_stage_checked": 0,
        "life_stage_found": 0,

        "aops_with_all_categories_present": 0,
    }

    per_aop_failures = []

    for aop in aops:
        stats["aops_total"] += 1
        aop_id = aop.get("aop_id")
        text = safe_join_text(aop)

        # --- Stressors ---
        stressors = aop.get("stressors") or []
        stressor_names = []
        for st in stressors:
            if isinstance(st, dict) and st.get("name"):
                stressor_names.append(st["name"])
            elif isinstance(st, str):
                stressor_names.append(st)

        stressor_found_any = True
        for name in stressor_names:
            stats["stressors_checked"] += 1
            ok = match_in_text(text, name, mode=mode, fuzzy_threshold=fuzzy_threshold)
            stats["stressors_found"] += int(ok)
            if not ok:
                stressor_found_any = False
                if verbose:
                    per_aop_failures.append(("stressor", aop_id, name))

        # --- Events (MIE/KE/AO grouped) ---
        events_obj = aop.get("events") or {}
        all_events = []
        if isinstance(events_obj, dict):
            for _, ev_list in events_obj.items():
                if isinstance(ev_list, list):
                    all_events.extend([ev for ev in ev_list if isinstance(ev, dict)])

        events_found_any = True
        for ev in all_events:
            stats["events_checked"] += 1
            cands = candidates_for_event(ev)
            ok = any(match_in_text(text, cand, mode=mode, fuzzy_threshold=fuzzy_threshold) for cand in cands)
            stats["events_found"] += int(ok)
            if not ok:
                events_found_any = False
                if verbose:
                    per_aop_failures.append(("event", aop_id, (ev.get("title") or ev.get("short_name") or ev.get("event_id"))))

        # --- Sex / Taxonomy / Life stage ---
        sexes = iter_applicability_names(aop, "sex", "sex")
        tax = iter_applicability_names(aop, "taxonomy", "name")
        life = iter_applicability_names(aop, "life_stage", "life_stage")

        sex_found_any = True
        for s in sexes:
            stats["sex_checked"] += 1
            ok = match_in_text(text, s, mode=mode, fuzzy_threshold=fuzzy_threshold)
            stats["sex_found"] += int(ok)
            if not ok:
                sex_found_any = False
                if verbose:
                    per_aop_failures.append(("sex", aop_id, s))

        tax_found_any = True
        for t in tax:
            stats["taxonomy_checked"] += 1
            ok = match_in_text(text, t, mode=mode, fuzzy_threshold=fuzzy_threshold)
            stats["taxonomy_found"] += int(ok)
            if not ok:
                tax_found_any = False
                if verbose:
                    per_aop_failures.append(("taxonomy", aop_id, t))

        life_found_any = True
        for l in life:
            stats["life_stage_checked"] += 1
            ok = match_in_text(text, l, mode=mode, fuzzy_threshold=fuzzy_threshold)
            stats["life_stage_found"] += int(ok)
            if not ok:
                life_found_any = False
                if verbose:
                    per_aop_failures.append(("life_stage", aop_id, l))

        # AOP-level “all categories” metric:
        # If a category has zero items, we treat it as "not applicable" (doesn't fail the AOP)
        def cat_ok(checked_count: int, found_any: bool) -> bool:
            return True if checked_count == 0 else found_any

        # Determine if this AOP passes all applicable categories
        # (computed from per-AOP booleans)
        if (
            cat_ok(len(stressor_names), stressor_found_any) and
            cat_ok(len(all_events), events_found_any) and
            cat_ok(len(sexes), sex_found_any) and
            cat_ok(len(tax), tax_found_any) and
            cat_ok(len(life), life_found_any)
        ):
            stats["aops_with_all_categories_present"] += 1

    return {"stats": stats, "failures": per_aop_failures}


def pct(found: int, checked: int) -> str:
    if checked == 0:
        return "n/a"
    return f"{(found / checked) * 100:.1f}%"


def main(INPUT_PATH, MODE="exact", FUZZY_THRESHOLD=90, VERBOSE=False):
    inp = Path(INPUT_PATH)
    aops = json.loads(inp.read_text(encoding="utf-8"))

    if not isinstance(aops, list):
        raise ValueError("Input JSON must be a list of AOP objects.")

    res = audit_aops(aops, mode=MODE, fuzzy_threshold=FUZZY_THRESHOLD, verbose=VERBOSE)
    s = res["stats"]

    print("\n=== AOP grounding audit ===")
    print(f"AOPs total: {s['aops_total']}")
    print(f"AOPs with all (applicable) categories found: {s['aops_with_all_categories_present']} / {s['aops_total']} "
          f"({(s['aops_with_all_categories_present']/s['aops_total']*100):.1f}%)")

    print("\n--- Items ---")
    print(f"Stressors:   checked {s['stressors_checked']}, found {s['stressors_found']} ({pct(s['stressors_found'], s['stressors_checked'])})")
    print(f"Events:      checked {s['events_checked']}, found {s['events_found']} ({pct(s['events_found'], s['events_checked'])})")
    print(f"Sex:         checked {s['sex_checked']}, found {s['sex_found']} ({pct(s['sex_found'], s['sex_checked'])})")
    print(f"Taxonomy:    checked {s['taxonomy_checked']}, found {s['taxonomy_found']} ({pct(s['taxonomy_found'], s['taxonomy_checked'])})")
    print(f"Life stage:  checked {s['life_stage_checked']}, found {s['life_stage_found']} ({pct(s['life_stage_found'], s['life_stage_checked'])})")

    if VERBOSE:
        print("\n--- Missing items (verbose) ---")
        for kind, aop_id, item in res["failures"]:
            print(f"[{kind}] aop_id={aop_id} :: {item}")


if __name__ == "__main__":
    main("new_data/aop_extracted_data.json") # mode can also be "fuzzy" with a --threshold argument (e.g. 85-95)
