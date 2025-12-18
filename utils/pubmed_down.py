#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests
import xml.etree.ElementTree as ET
from ftplib import FTP
from urllib.parse import urlparse
from pathlib import Path


# ----------------------------
# Helpers: parsing & normalize
# ----------------------------

PMID_RE = re.compile(r"(?:pubmed\.ncbi\.nlm\.nih\.gov|ncbi\.nlm\.nih\.gov/pubmed)/(\d+)")
DIGITS_RE = re.compile(r"\b(\d{6,10})\b")  # PMID often 7-9 digits, keep flexible

def parse_citation_urls(raw: Optional[str]) -> List[str]:
    """
    Extract URLs from a 'citation_urls' field.
    Expected examples:
      "http://.../18297081; http://.../20143881; https://pubmed.ncbi.nlm.nih.gov/30263952/"
    We'll split mainly by ';' but also handle commas and extra junk.
    """
    if raw is None:
        return []
    s = str(raw).strip()
    if not s or s.lower() in {"null", "none", "nan"}:
        return []

    # Strip common wrappers
    s = s.strip().strip('"').strip("'")
    s = s.strip("[](){}")

    # Prefer semicolon split; fallback to comma if needed
    parts = [p.strip() for p in s.split(";")] if ";" in s else [p.strip() for p in s.split(",")]

    urls: List[str] = []
    for p in parts:
        if not p:
            continue
        p = p.strip().strip('"').strip("'")
        # Some exports jam multiple spaces
        p = re.sub(r"\s+", " ", p).strip()
        # Keep only http(s) things
        if p.startswith("http://") or p.startswith("https://"):
            urls.append(p)

    # De-dup preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def canonicalize_pubmed_url(url: str) -> str:
    """
    If it's a PubMed link, normalize to: https://pubmed.ncbi.nlm.nih.gov/<PMID>/
    Otherwise keep as-is.
    """
    u = url.strip()
    m = PMID_RE.search(u)
    if m:
        pmid = m.group(1)
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    # If it contains a PMID-like number but not matching patterns, try to extract
    if "pubmed" in u.lower():
        d = DIGITS_RE.search(u)
        if d:
            pmid = d.group(1)
            return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    return u


def extract_pmid(url: str) -> Optional[str]:
    m = PMID_RE.search(url)
    if m:
        return m.group(1)
    if "pubmed" in url.lower():
        d = DIGITS_RE.search(url)
        if d:
            return d.group(1)
    return None


# ----------------------------
# I/O: build links dict + CSV
# ----------------------------

def iter_tsv_files(input_dir: Path) -> List[Path]:
    files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".tsv"])
    return files


def read_tsv_rows(tsv_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        rows = [row for row in reader]
    return rows, fieldnames


def build_links_dict(input_dir: Path, citation_col: str = "citation_urls") -> Tuple[Dict[str, List[Dict[str, str]]], List[str]]:
    links: Dict[str, List[Dict[str, str]]] = {}
    all_fieldnames: List[str] = []
    for tsv in iter_tsv_files(input_dir):
        rows, fieldnames = read_tsv_rows(tsv)
        if not all_fieldnames:
            all_fieldnames = fieldnames
        # allow varying columns: union later
        for row in rows:
            raw = row.get(citation_col, "")
            urls = parse_citation_urls(raw)
            if not urls:
                continue
            for u in urls:
                link = canonicalize_pubmed_url(u)
                r = deepcopy(row)
                r["chemical_filename"] = tsv.name
                links.setdefault(link, []).append(r)

    # Union fieldnames across all rows, keep stable order with new extras appended
    union = []
    seen = set()
    for fn in all_fieldnames + ["chemical_filename"]:
        if fn not in seen:
            union.append(fn); seen.add(fn)
    # also include any unexpected columns that appeared
    for link, rows in links.items():
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    union.append(k); seen.add(k)

    return links, union


def save_links_json(links: Dict[str, List[Dict[str, str]]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(links, f, ensure_ascii=False, indent=2)


def assign_paper_ids(links: Dict[str, List[Dict[str, str]]], start_id: int = 1000) -> Dict[str, str]:
    """
    Deterministic: sort links so reruns produce stable IDs (as long as link set is same).
    """
    mapping: Dict[str, str] = {}
    for i, link in enumerate(sorted(links.keys())):
        mapping[link] = f"paper_{start_id + i}"
    return mapping


def write_csv_rows(out_csv: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_csv(out_csv: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        rows = [r for r in reader]
    return rows, fieldnames


def atomic_write_csv(out_csv: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    write_csv_rows(tmp, rows, fieldnames)
    tmp.replace(out_csv)


# ----------------------------
# Download: PubMed -> PMC -> PDF
# ----------------------------

def pmid_to_pmcid(session: requests.Session, pmid: str, timeout: int = 30) -> Optional[str]:
    # PubMed -> PMC link via E-utilities
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    params = {"dbfrom": "pubmed", "db": "pmc", "id": pmid, "retmode": "json"}
    r = session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    try:
        for ls in data.get("linksets", []):
            for ldb in ls.get("linksetdbs", []):
                links = ldb.get("links", [])
                if links:
                    return f"PMC{links[0]}"
    except Exception:
        return None
    return None


def pmcid_to_pdf_url_via_oa_service(session: requests.Session, pmcid: str, timeout: int = 30) -> Optional[str]:
    # PMC OA Web Service (returns XML with <link format="pdf" href="...">)
    base = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
    r = session.get(base, params={"id": pmcid}, timeout=timeout)
    r.raise_for_status()

    root = ET.fromstring(r.content)

    # find link format="pdf"
    for link in root.findall(".//record/link"):
        if link.attrib.get("format") == "pdf":
            return link.attrib.get("href")
    return None


def download_ftp_file(ftp_url: str, out_path: Path, timeout: int = 60) -> None:
    """
    Download ftp://ftp.ncbi.nlm.nih.gov/... file via ftplib.
    """
    u = urlparse(ftp_url)
    if u.scheme != "ftp":
        raise ValueError(f"Not an ftp url: {ftp_url}")

    host = u.hostname
    path = u.path  # includes leading "/"
    if not host or not path:
        raise ValueError(f"Bad ftp url: {ftp_url}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with FTP(host, timeout=timeout) as ftp:
        ftp.login()  # anonymous
        with out_path.open("wb") as f:
            ftp.retrbinary(f"RETR {path}", f.write)


def try_download_pubmed_pdf_official(
    pubmed_url: str,
    pdf_path: Path,
    timeout: int = 45,
    sleep_s: float = 0.25,
) -> tuple[bool, str]:
    """
    Returns (ok, reason). Uses:
      PMID -> PMCID (eutils)
      PMCID -> PDF link (PMC OA service)
      download via FTP
    """
    # Extract PMID from URL (keep your regex or implement similarly)
    import re
    m = re.search(r"(?:pubmed\.ncbi\.nlm\.nih\.gov|ncbi\.nlm\.nih\.gov/pubmed|pubmed\.ncbi\.nlm\.nih\.gov)/(\d+)", pubmed_url)
    if not m:
        m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", pubmed_url)
    if not m:
        m = re.search(r"\b(\d{6,10})\b", pubmed_url)
    if not m:
        return False, "Could not extract PMID"

    pmid = m.group(1)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "paper-downloader/1.0 (mailto:you@example.com)",
        "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
    })

    try:
        pmcid = pmid_to_pmcid(session, pmid, timeout=timeout)
        if not pmcid:
            return False, "No PMCID (not in PMC)"
        time.sleep(sleep_s)

        pdf_url = pmcid_to_pdf_url_via_oa_service(session, pmcid, timeout=timeout)
        if not pdf_url:
            return False, "No OA-subset PDF (oa.fcgi returned no pdf link)"

        # If already downloaded, skip
        if pdf_path.exists() and pdf_path.stat().st_size > 1024:
            return True, "Already downloaded"

        # oa.fcgi commonly returns ftp://... (download with ftplib)
        if pdf_url.startswith("ftp://"):
            download_ftp_file(pdf_url, pdf_path, timeout=max(timeout, 60))
        else:
            # Rare, but handle http(s) too
            rr = session.get(pdf_url, timeout=timeout, stream=True)
            rr.raise_for_status()
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            with pdf_path.open("wb") as f:
                for chunk in rr.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)

        return True, "OK"
    except Exception as e:
        # cleanup partial
        try:
            if pdf_path.exists():
                pdf_path.unlink()
        except Exception:
            pass
        return False, f"Download error: {e}"

@dataclass
class DownloadResult:
    ok: bool
    reason: str
    pdf_url: Optional[str] = None
    pmcid: Optional[str] = None


def get_pmcid_from_pmid(session: requests.Session, pmid: str, timeout: int = 30) -> Optional[str]:
    """
    Use NCBI E-utilities elink: pubmed -> pmc.
    Returns numeric PMC id (without 'PMC') if available.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": pmid,
        "retmode": "json",
    }
    r = session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    try:
        linksets = data.get("linksets", [])
        for ls in linksets:
            ldbs = ls.get("linksetdbs", [])
            for ldb in ldbs:
                links = ldb.get("links", [])
                if links:
                    # links are numeric PMC IDs (without PMC prefix)
                    return str(links[0])
    except Exception:
        return None
    return None


def find_pdf_url_on_pmc_page(html: str, base: str = "https://pmc.ncbi.nlm.nih.gov") -> Optional[str]:
    """
    Find a reasonable PDF link in PMC article page HTML.
    """
    # Common patterns:
    #  - href="/articles/PMC1234567/pdf/..."
    #  - href="/articles/PMC1234567/pdf/PMC1234567.pdf"
    m = re.search(r'href="([^"]+/pdf/[^"]+\.pdf)"', html, flags=re.IGNORECASE)
    if m:
        href = m.group(1)
        if href.startswith("http"):
            return href
        return base + href

    # fallback: sometimes pdf is just /articles/PMCxxxxxxx/pdf/
    m2 = re.search(r'href="([^"]+/pdf/)"', html, flags=re.IGNORECASE)
    if m2:
        href = m2.group(1)
        if href.startswith("http"):
            return href
        return base + href

    return None


def try_download_pdf_for_pubmed_link(
    session: requests.Session,
    link: str,
    pdf_path: Path,
    timeout: int = 45,
    sleep_s: float = 0.25,
) -> DownloadResult:
    """
    Tries to download a PDF for a PubMed link using PMC if possible.
    """
    pmid = extract_pmid(link)
    if not pmid:
        return DownloadResult(False, "Could not extract PMID from link")

    # E-utilities: PubMed -> PMC
    try:
        pmcid_num = get_pmcid_from_pmid(session, pmid, timeout=timeout)
    except Exception as e:
        return DownloadResult(False, f"E-utilities error: {e}")

    if not pmcid_num:
        return DownloadResult(False, "No PMCID (not in PubMed Central)")

    pmc_article_url = f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid_num}/"
    try:
        r = session.get(pmc_article_url, timeout=timeout)
        r.raise_for_status()
        pdf_url = find_pdf_url_on_pmc_page(r.text)
        if not pdf_url:
            # Last-ditch: some pages accept /pdf/ directly
            pdf_url = f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid_num}/pdf/"
    except Exception as e:
        return DownloadResult(False, f"PMC page fetch error: {e}", pmcid=f"PMC{pmcid_num}")

    # Download PDF (stream)
    try:
        time.sleep(sleep_s)
        rr = session.get(pdf_url, timeout=timeout, stream=True, allow_redirects=True)
        rr.raise_for_status()

        # Validate content looks like PDF
        first = rr.raw.read(5, decode_content=True)
        if first != b"%PDF-":
            return DownloadResult(False, "Downloaded content is not a PDF", pdf_url=pdf_url, pmcid=f"PMC{pmcid_num}")

        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with pdf_path.open("wb") as f:
            f.write(first)
            for chunk in rr.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

        return DownloadResult(True, "OK", pdf_url=pdf_url, pmcid=f"PMC{pmcid_num}")
    except Exception as e:
        # if partial file exists, remove it
        try:
            if pdf_path.exists():
                pdf_path.unlink()
        except Exception:
            pass
        return DownloadResult(False, f"PDF download error: {e}", pdf_url=pdf_url, pmcid=f"PMC{pmcid_num}")


def download_missing_pdfs_and_update_csv(
    out_csv: Path,
    pdf_dir: Path,
    timeout: int,
    user_agent: str,
) -> None:
    rows, fieldnames = load_csv(out_csv)

    # Identify the columns we rely on
    required = {"paper_id", "downloaded", "link"}
    missing_cols = required - set(fieldnames)
    if missing_cols:
        raise RuntimeError(f"CSV missing required columns: {sorted(missing_cols)}")

    # Build unique paper list (paper_id, link)
    unique: Dict[str, str] = {}
    for r in rows:
        pid = r.get("paper_id", "").strip()
        link = r.get("link", "").strip()
        if pid and link:
            unique[pid] = link

    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
    })

    total = len(unique)
    done_count = 0

    # Pre-index rows per paper_id for fast updates
    idx_by_pid: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        idx_by_pid.setdefault(r["paper_id"], []).append(i)

    for n, (paper_id, link) in enumerate(sorted(unique.items(), key=lambda x: x[0])):
        pdf_path = pdf_dir / f"{paper_id}.pdf"

        # Already downloaded? (file exists)
        if pdf_path.exists() and pdf_path.stat().st_size > 1024:
            # mark all rows for this paper as downloaded True
            for i in idx_by_pid.get(paper_id, []):
                rows[i]["downloaded"] = "True"
            done_count += 1
            if n % 25 == 0:
                atomic_write_csv(out_csv, rows, fieldnames)
            continue

        # Skip if CSV already says True everywhere
        all_true = True
        for i in idx_by_pid.get(paper_id, []):
            if rows[i].get("downloaded", "").strip().lower() != "true":
                all_true = False
                break
        if all_true:
            done_count += 1
            continue

        # Attempt download (only really works for PMC)
        ok, reason = try_download_pubmed_pdf_official(link, pdf_path)

        # Update rows
        for i in idx_by_pid.get(paper_id, []):
            rows[i]["downloaded"] = "True" if ok else "False"

        # Persist progress after each attempt (so reruns resume cleanly)
        atomic_write_csv(out_csv, rows, fieldnames)

        status = "OK" if ok else "FAIL"
        print(f"[{n+1}/{total}] {paper_id} {status} | {link} | {reason}")

    print(f"Download phase finished. Papers with PDF present/marked: {done_count}/{total}")


# ----------------------------
# Main: two-phase with resume
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Group TSV rows by citation URLs and download PDFs from PMC when possible.")
    ap.add_argument("--input-dir", type=str, default="data/exports", help="Folder containing TSV files.")
    ap.add_argument("--out-dir", type=str, default="data/exports/out", help="Output folder (links.json, papers.csv, pdfs/).")
    ap.add_argument("--citation-col", type=str, default="citation_urls", help="Column name containing citation URLs.")
    ap.add_argument("--start-id", type=int, default=1000, help="Start number for paper_XXXX ids.")
    ap.add_argument("--timeout", type=int, default=45, help="HTTP timeout (seconds).")
    ap.add_argument("--user-agent", type=str, default="paper-downloader/1.0 (mailto:example@example.com)",
                    help="User-Agent string (set a real mailto if you care about being polite).")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    links_json = out_dir / "links.json"
    out_csv = out_dir / "papers.csv"
    pdf_dir = out_dir / "pdfs"

    # Phase 1: Build CSV + links.json (only if CSV doesn't exist)
    if not out_csv.exists():
        links, union_fieldnames = build_links_dict(input_dir, citation_col=args.citation_col)
        print(f"Collected unique links: {len(links)}")

        save_links_json(links, links_json)
        print(f"Saved links dict to: {links_json}")

        id_map = assign_paper_ids(links, start_id=args.start_id)

        # Flatten to CSV rows
        flattened: List[Dict[str, str]] = []
        # CSV columns: required + rest (union)
        # Keep citation_urls too (user said "everything else"; you can drop it if you want)
        base_cols = ["paper_id", "downloaded", "link", "chemical_filename"]
        rest_cols = [c for c in union_fieldnames if c not in base_cols]
        fieldnames = base_cols + rest_cols

        for link, rows_for_link in links.items():
            paper_id = id_map[link]
            for r in rows_for_link:
                out_row = deepcopy(r)
                out_row["paper_id"] = paper_id
                out_row["downloaded"] = "False"
                out_row["link"] = link
                # ensure chemical_filename exists
                out_row.setdefault("chemical_filename", "")
                flattened.append(out_row)

        # Save CSV BEFORE downloads (as requested)
        write_csv_rows(out_csv, flattened, fieldnames)
        print(f"Saved CSV to: {out_csv} (rows: {len(flattened)})")
    else:
        print(f"CSV already exists -> skipping parsing: {out_csv}")

    # Phase 2: Download PDFs (resume-friendly)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    download_missing_pdfs_and_update_csv(
        out_csv=out_csv,
        pdf_dir=pdf_dir,
        timeout=args.timeout,
        user_agent=args.user_agent,
    )
    print(f"PDF folder: {pdf_dir}")


if __name__ == "__main__":
    main()
