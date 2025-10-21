import csv, os, re, time, json
from urllib.parse import quote
import requests
from pathlib import Path

# ====== CONFIG ======
INPUT_CSV = "data/raw/papers.csv"   # your file
DOI_COLUMN = "DOI"                 # exact header name
OUT_DIR = Path("data/pdfs")
REPORT_CSV = f"data/pdfs/download_report_{time.strftime('%Y%m%d')}.csv"
UNPAYWALL_EMAIL = "davidelugli1@gmail.com"  # REQUIRED by Unpaywall's ToS
RATE_DELAY = 0.4  # seconds between calls to be polite

OUT_DIR.mkdir(exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": f"bulk-doi-fetch/1.0 ({UNPAYWALL_EMAIL})"})

doi_re = re.compile(r"(10\.\d{4,9}/\S+)", re.IGNORECASE)

def extract_doi(s):
    if not s: return None
    m = doi_re.search(str(s).strip())
    return m.group(1).rstrip(".,);]") if m else None

def unpaywall_pdf(doi):
    url = f"https://api.unpaywall.org/v2/{quote(doi)}?email={quote(UNPAYWALL_EMAIL)}"
    r = session.get(url, timeout=20)
    if r.status_code != 200:
        return None, None
    data = r.json()
    # prefer best_oa_location.url_for_pdf
    loc = data.get("best_oa_location") or {}
    pdf = loc.get("url_for_pdf")
    host_type = loc.get("host_type")
    version = loc.get("version")
    if not pdf:
        # fall back to any oa_location with url_for_pdf
        for l in data.get("oa_locations", []) or []:
            if l.get("url_for_pdf"):
                pdf = l["url_for_pdf"]
                host_type = l.get("host_type"); version = l.get("version")
                break
    return pdf, {"source":"unpaywall", "host_type":host_type, "version":version}

def crossref_pdf(doi):
    url = f"https://api.crossref.org/works/{quote(doi)}"
    r = session.get(url, timeout=20)
    if r.status_code != 200:
        return None, None
    msg = r.json().get("message", {})
    for link in msg.get("link", []) or []:
        if link.get("content-type") == "application/pdf" and link.get("URL"):
            return link["URL"], {"source":"crossref"}
    return None, None

def europe_pmc_pdf(doi):
    # search by DOI, ask for resultType=core to include fullTextUrlList
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:{quote(doi)}&format=json&resultType=core&pageSize=1"
    r = session.get(url, timeout=20)
    if r.status_code != 200:
        return None, None
    res = r.json().get("resultList", {}).get("result", [])
    if not res: return None, None
    full = res[0].get("fullTextUrlList", {}).get("fullTextUrl", []) or []
    # prefer pdf
    for f in full:
        if f.get("documentStyle","").lower()=="pdf" and f.get("url"):
            return f["url"], {"source":"europe_pmc"}
    return None, None

def safe_filename(doi_id, doi):
    doi_id = str(doi_id).zfill(4)
    return f"paper_{doi_id}_" + re.sub(r"[^a-zA-Z0-9._-]+", "_", doi) + ".pdf"

def try_download(url, outpath):
    try:
        with session.get(url, timeout=60, stream=True, allow_redirects=True) as r:
            ct = (r.headers.get("Content-Type") or "").lower()
            if r.status_code == 200 and ("pdf" in ct or url.lower().endswith(".pdf")):
                with open(outpath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                return True, f"OK {r.status_code}"
            # sometimes servers don’t set content-type right; still attempt
            if r.status_code == 200 and outpath.suffix.lower()==".pdf":
                with open(outpath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk: f.write(chunk)
                return True, f"OK-noct {r.status_code}"
            return False, f"HTTP {r.status_code} ct={ct}"
    except requests.RequestException as e:
        return False, f"ERR {e}"

def pipeline(dois):
    report = []
    for doi_raw in dois:
        doi = extract_doi(doi_raw[0])
        doi_id = doi_raw[1]
        print(f"Processing doi {doi_id}/{len(dois)}...")
        if not doi:
            report.append({"id": doi_id, "doi": str(doi_raw[0]), "status": "no_doi_found", "source": "", "url": "", "note": ""})
            continue

        outpath = OUT_DIR / safe_filename(doi_id, doi)
        if outpath.exists():
            report.append({"id": doi_id, "doi": doi, "status": "already_downloaded", "source": "", "url": "", "note": ""})
            continue

        # step 1: unpaywall
        pdf_url, meta = unpaywall_pdf(doi); time.sleep(RATE_DELAY)
        # step 2: crossref
        if not pdf_url:
            pdf_url, meta = crossref_pdf(doi); time.sleep(RATE_DELAY)
        # step 3: europe pmc
        if not pdf_url:
            pdf_url, meta = europe_pmc_pdf(doi); time.sleep(RATE_DELAY)

        if not pdf_url:
            report.append({"id": doi_id, "doi": doi, "status": "no_open_pdf_found", "source": "", "url": "", "note": ""})
            continue
        if meta is None:
            report.append({"id": doi_id, "doi": doi, "status": "no_meta_found", "source": "", "url": "", "note": ""})
            continue

        ok, note = try_download(pdf_url, outpath)
        
        if ok:
            report.append({"id": doi_id, "doi": doi, "status": "downloaded", "source": meta.get("source",""), "url": pdf_url, "note": note})
        else:
            report.append({"id": doi_id, "doi": doi, "status": "download_failed", "source": meta.get("source",""), "url": pdf_url, "note": note})
    return report

# ---- read input ----
with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
    dois = [(row.get(DOI_COLUMN), row.get("ID")) for row in rows]

rep = pipeline(dois)

# ---- write report ----
with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id","doi","status","source","url","note"])
    w.writeheader()
    w.writerows(rep)

print(f"Done. PDFs in {OUT_DIR.resolve()}. Report: {REPORT_CSV}")
