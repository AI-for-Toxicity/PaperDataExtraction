#!/usr/bin/env python3
import csv
import sys
from collections import defaultdict

def process_tsv(input_path, output_path):
    # Read TSV
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    required_cols = ["name", "casrn", "dtxsid", "ctd_id", "source"]
    for col in required_cols:
        if col not in reader.fieldnames:
            print(f"ERROR: Required column '{col}' not found in input file.", file=sys.stderr)
            sys.exit(1)

    # 1) Check every row has casrn AND dtxsid non-empty
    bad_rows = []
    for i, row in enumerate(rows, start=2):  # start=2 because row 1 is header
        casrn = (row.get("casrn") or "").strip()
        dtxsid = (row.get("dtxsid") or "").strip()
        if not casrn or not dtxsid:
            bad_rows.append((i, casrn, dtxsid))

    if bad_rows:
        print("ERROR: Found rows with empty casrn or dtxsid:", file=sys.stderr)
        for line_num, casrn, dtxsid in bad_rows:
            print(f"  Line {line_num}: casrn='{casrn}', dtxsid='{dtxsid}'", file=sys.stderr)
        sys.exit(1)

    # 2) Check casrn <-> dtxsid mapping consistency
    casrn_to_dtx = defaultdict(set)
    dtx_to_casrn = defaultdict(set)

    for row in rows:
        casrn = row["casrn"].strip()
        dtxsid = row["dtxsid"].strip()
        casrn_to_dtx[casrn].add(dtxsid)
        dtx_to_casrn[dtxsid].add(casrn)

    inconsistent = False

    for casrn, dtxs in casrn_to_dtx.items():
        if len(dtxs) > 1:
            inconsistent = True
            print(f"ERROR: casrn '{casrn}' maps to multiple dtxsid values: {sorted(dtxs)}", file=sys.stderr)

    for dtxsid, casrns in dtx_to_casrn.items():
        if len(casrns) > 1:
            inconsistent = True
            print(f"ERROR: dtxsid '{dtxsid}' maps to multiple casrn values: {sorted(casrns)}", file=sys.stderr)

    if inconsistent:
        print("Terminating due to inconsistent casrn/dtxsid mappings.", file=sys.stderr)
        sys.exit(1)

    # 3) Add 'preferred' column and set first row per (casrn, dtxsid) pair to true
    seen_pairs = set()
    for row in rows:
        casrn = row["casrn"].strip()
        dtxsid = row["dtxsid"].strip()
        key = (casrn, dtxsid)
        if key not in seen_pairs:
            row["preferred"] = "true"
            seen_pairs.add(key)
        else:
            row["preferred"] = "false"

    # Prepare output header: original columns + preferred
    fieldnames = reader.fieldnames.copy()
    if "preferred" not in fieldnames:
        fieldnames.append("preferred")

    # 4) Write new TSV
    with open(output_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    input = "data/resources/dsstox_extraction.tsv"
    output = "data/resources/dsstox_extraction_preferred.tsv"

    process_tsv(input, output)
    