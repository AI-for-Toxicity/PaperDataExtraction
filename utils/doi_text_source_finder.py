import csv
import re
import time
import requests

INPUT_CSV = "data/labels/papers.csv"
DOI_COLUMN = "DOI"

doi_re = re.compile(r"(10\.\d{4,9}/\S+)", re.IGNORECASE)

def extract_doi(s):
    if not s:
        return None
    m = doi_re.search(str(s).strip())
    return m.group(1).rstrip(".,);]") if m else None

def check_pmc(doi):
    """Return PMC ID if found, else None."""
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={doi}&format=json"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()
    records = data.get("records", [])
    if not records:
        return None
    rec = records[0]
    return rec.get("pmcid")

def check_unpaywall(doi, email="dummy@example.com"):
    """Return True if any full-text OA location exists."""
    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()
    oa_loc = data.get("best_oa_location") or {}
    return oa_loc.get("url_for_pdf") or oa_loc.get("url_for_fulltext")

def pipeline(dois):
    report = []
    total = len(dois)
    for i, (doi_raw, doi_id) in enumerate(dois, start=1):
        doi = extract_doi(doi_raw)
        if not doi:
            continue
        print(f"[{i}/{total}] Checking DOI {doi}...")

        pmc_id = check_pmc(doi)
        if pmc_id:
            print(f"  ✅ Found on PubMed Central: {pmc_id}")
            report.append((doi, "PubMed Central", pmc_id))
            continue

        fulltext_url = check_unpaywall(doi)
        if fulltext_url:
            print(f"  🟢 Found OA text elsewhere: {fulltext_url}")
            report.append((doi, "Other OA Source", fulltext_url))
        else:
            print(f"  ❌ No plain-text/full-text found.")
            report.append((doi, "Not Available", ""))

        time.sleep(1)  # polite rate limiting
    return report

def main():
    with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
        dois = [(row.get(DOI_COLUMN), row.get("ID")) for row in rows]

    results = pipeline(dois)

    # Save results for sanity
    with open("doi_text_availability.csv", "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["DOI", "Source", "Link"])
        writer.writerows(results)

if __name__ == "__main__":
    main()
