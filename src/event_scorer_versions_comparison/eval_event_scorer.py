import re
import math
import statistics as stats
from pathlib import Path
from typing import Dict, List, Tuple

cmp1 = Path("src/event_scorer_versions_comparison/event_scored_v2_result")
cmp2 = Path("src/event_scorer_versions_comparison/event_scored_v3_result")


LINE_RE = re.compile(
    r"""
    ^\[OK\]\ Saved\ (?P<path>.+?)\ \|\ 
    events=(?P<events>\d+)\ 
    matched=(?P<matched>\d+)\ 
    unmatched=(?P<unmatched>\d+)\ 
    ls=(?P<ls>-?\d+(?:\.\d+)?)\ 
    hs=(?P<hs>-?\d+(?:\.\d+)?)
    $
    """,
    re.VERBOSE,
)


def parse_log_file(log_path: Path) -> Dict[str, dict]:
    """
    Parse one scorer output log.

    Returns:
        dict[path_str] = {
            "events": int,
            "matched": int,
            "unmatched": int,
            "ls": float,
            "hs": float,
            "raw_line": str
        }
    """
    results = {}

    with log_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            m = LINE_RE.match(line)
            if not m:
                continue

            path_str = m.group("path").strip()

            if path_str in results:
                raise ValueError(
                    f"Duplicate output path in {log_path} at line {line_num}: {path_str}"
                )

            results[path_str] = {
                "events": int(m.group("events")),
                "matched": int(m.group("matched")),
                "unmatched": int(m.group("unmatched")),
                "ls": float(m.group("ls")),
                "hs": float(m.group("hs")),
                "raw_line": line,
            }

    return results


def mean_or_nan(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def median_or_nan(values: List[float]) -> float:
    return stats.median(values) if values else float("nan")


def stdev_or_nan(values: List[float]) -> float:
    return stats.pstdev(values) if values else float("nan")


def fmt(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{x:.{digits}f}"


def compare_logs(v1_path: Path, v2_path: Path) -> None:
    v1 = parse_log_file(v1_path)
    v2 = parse_log_file(v2_path)

    if len(v1) != len(v2):
        raise ValueError(
            f"Different number of parsed [OK] entries: v1={len(v1)} vs v2={len(v2)}"
        )

    missing_in_v2 = sorted(set(v1) - set(v2))
    missing_in_v1 = sorted(set(v2) - set(v1))

    if missing_in_v2 or missing_in_v1:
        raise ValueError(
            "The two logs do not contain the same output paths.\n"
            f"Missing in v2: {missing_in_v2[:5]}\n"
            f"Missing in v1: {missing_in_v1[:5]}"
        )

    rows = []
    for path in sorted(v1.keys()):
        a = v1[path]
        b = v2[path]

        if a["events"] != b["events"]:
            raise ValueError(
                f"Event count mismatch for {path}: v1={a['events']} vs v2={b['events']}"
            )

        rows.append({
            "path": path,
            "events": a["events"],

            "matched_v1": a["matched"],
            "unmatched_v1": a["unmatched"],
            "ls_v1": a["ls"],
            "hs_v1": a["hs"],

            "matched_v2": b["matched"],
            "unmatched_v2": b["unmatched"],
            "ls_v2": b["ls"],
            "hs_v2": b["hs"],

            "delta_matched": b["matched"] - a["matched"],
            "delta_unmatched": b["unmatched"] - a["unmatched"],
            "delta_ls": b["ls"] - a["ls"],
            "delta_hs": b["hs"] - a["hs"],
        })

    total_files = len(rows)

    total_events = sum(r["events"] for r in rows)

    total_matched_v1 = sum(r["matched_v1"] for r in rows)
    total_matched_v2 = sum(r["matched_v2"] for r in rows)
    total_unmatched_v1 = sum(r["unmatched_v1"] for r in rows)
    total_unmatched_v2 = sum(r["unmatched_v2"] for r in rows)

    matched_rate_v1 = total_matched_v1 / total_events if total_events else float("nan")
    matched_rate_v2 = total_matched_v2 / total_events if total_events else float("nan")

    ls_v1 = [r["ls_v1"] for r in rows]
    ls_v2 = [r["ls_v2"] for r in rows]
    hs_v1 = [r["hs_v1"] for r in rows]
    hs_v2 = [r["hs_v2"] for r in rows]

    delta_matched = [r["delta_matched"] for r in rows]
    delta_ls = [r["delta_ls"] for r in rows]
    delta_hs = [r["delta_hs"] for r in rows]

    matched_better = [r for r in rows if r["delta_matched"] > 0]
    matched_worse = [r for r in rows if r["delta_matched"] < 0]
    matched_same = [r for r in rows if r["delta_matched"] == 0]

    ls_better = [r for r in rows if r["delta_ls"] > 0]
    ls_worse = [r for r in rows if r["delta_ls"] < 0]
    ls_same = [r for r in rows if r["delta_ls"] == 0]

    hs_better = [r for r in rows if r["delta_hs"] > 0]
    hs_worse = [r for r in rows if r["delta_hs"] < 0]
    hs_same = [r for r in rows if r["delta_hs"] == 0]

    print("=" * 80)
    print("SCORER COMPARISON")
    print("=" * 80)
    print(f"v1 log: {v1_path}")
    print(f"v2 log: {v2_path}")
    print(f"Comparable files: {total_files}")
    print(f"Total events: {total_events}")
    print()

    print("-" * 80)
    print("GLOBAL MATCH COUNTS")
    print("-" * 80)
    print(f"v1 matched   : {total_matched_v1}")
    print(f"v2 matched   : {total_matched_v2}")
    print(f"delta matched: {total_matched_v2 - total_matched_v1:+d}")
    print()
    print(f"v1 unmatched   : {total_unmatched_v1}")
    print(f"v2 unmatched   : {total_unmatched_v2}")
    print(f"delta unmatched: {total_unmatched_v2 - total_unmatched_v1:+d}")
    print()
    print(f"v1 matched rate: {fmt(matched_rate_v1 * 100, 2)}%")
    print(f"v2 matched rate: {fmt(matched_rate_v2 * 100, 2)}%")
    print(f"rate delta     : {fmt((matched_rate_v2 - matched_rate_v1) * 100, 2)} pp")
    print()

    print("-" * 80)
    print("AVERAGE / MEDIAN SCORE METRICS")
    print("-" * 80)
    print(f"avg ls v1: {fmt(mean_or_nan(ls_v1))}")
    print(f"avg ls v2: {fmt(mean_or_nan(ls_v2))}")
    print(f"avg ls Δ : {fmt(mean_or_nan(delta_ls))}")
    print()
    print(f"med ls v1: {fmt(median_or_nan(ls_v1))}")
    print(f"med ls v2: {fmt(median_or_nan(ls_v2))}")
    print()
    print(f"avg hs v1: {fmt(mean_or_nan(hs_v1))}")
    print(f"avg hs v2: {fmt(mean_or_nan(hs_v2))}")
    print(f"avg hs Δ : {fmt(mean_or_nan(delta_hs))}")
    print()
    print(f"med hs v1: {fmt(median_or_nan(hs_v1))}")
    print(f"med hs v2: {fmt(median_or_nan(hs_v2))}")
    print()
    print(f"ls std v1: {fmt(stdev_or_nan(ls_v1))}")
    print(f"ls std v2: {fmt(stdev_or_nan(ls_v2))}")
    print(f"hs std v1: {fmt(stdev_or_nan(hs_v1))}")
    print(f"hs std v2: {fmt(stdev_or_nan(hs_v2))}")
    print()

    print("-" * 80)
    print("PER-FILE WIN COUNTS")
    print("-" * 80)
    print(f"matched improved in v2 : {len(matched_better)}")
    print(f"matched worsened in v2 : {len(matched_worse)}")
    print(f"matched unchanged      : {len(matched_same)}")
    print()
    print(f"ls improved in v2      : {len(ls_better)}")
    print(f"ls worsened in v2      : {len(ls_worse)}")
    print(f"ls unchanged           : {len(ls_same)}")
    print()
    print(f"hs improved in v2      : {len(hs_better)}")
    print(f"hs worsened in v2      : {len(hs_worse)}")
    print(f"hs unchanged           : {len(hs_same)}")
    print()

    # Useful extra metrics
    both_matched_more_and_ls_up = [
        r for r in rows if r["delta_matched"] > 0 and r["delta_ls"] > 0
    ]
    matched_more_but_ls_down = [
        r for r in rows if r["delta_matched"] > 0 and r["delta_ls"] < 0
    ]
    matched_same_ls_up = [
        r for r in rows if r["delta_matched"] == 0 and r["delta_ls"] > 0
    ]

    print("-" * 80)
    print("USEFUL COMBINED METRICS")
    print("-" * 80)
    print(f"v2 matched more AND raised ls : {len(both_matched_more_and_ls_up)}")
    print(f"v2 matched more BUT lowered ls: {len(matched_more_but_ls_down)}")
    print(f"same matched count, higher ls : {len(matched_same_ls_up)}")
    print()

    # Biggest changes
    top_matched_gain = sorted(rows, key=lambda r: r["delta_matched"], reverse=True)[:10]
    top_matched_loss = sorted(rows, key=lambda r: r["delta_matched"])[:10]
    top_ls_gain = sorted(rows, key=lambda r: r["delta_ls"], reverse=True)[:10]
    top_ls_loss = sorted(rows, key=lambda r: r["delta_ls"])[:10]

    def print_top(title: str, items: List[dict], delta_key: str) -> None:
        print("-" * 80)
        print(title)
        print("-" * 80)
        for r in items:
            print(
                f"{r['path']} | "
                f"events={r['events']} | "
                f"matched {r['matched_v1']}->{r['matched_v2']} ({r['delta_matched']:+d}) | "
                f"ls {fmt(r['ls_v1'])}->{fmt(r['ls_v2'])} ({fmt(r['delta_ls'])}) | "
                f"hs {fmt(r['hs_v1'])}->{fmt(r['hs_v2'])} ({fmt(r['delta_hs'])})"
            )
        print()

    print_top("TOP MATCHED GAINS IN V2", top_matched_gain, "delta_matched")
    print_top("TOP MATCHED LOSSES IN V2", top_matched_loss, "delta_matched")
    print_top("TOP LS GAINS IN V2", top_ls_gain, "delta_ls")
    print_top("TOP LS LOSSES IN V2", top_ls_loss, "delta_ls")


if __name__ == "__main__":
    compare_logs(cmp1, cmp2)
