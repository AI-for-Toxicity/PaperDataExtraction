import re, csv, uuid, spacy, json
from chemdataextractor import Document
from rapidfuzz import process

nlp_species = spacy.load("en_ner_bionlp13cg_md")
nlp_chemicals = spacy.load("en_ner_bc5cdr_md")

IN_JSON = "data/processed/paragraphs_eval_sciscpacy.json"
OUT_CSV_SPECIES = "data/processed/species_extraction_scispacy.csv"
OUT_CSV_CHEMICALS = "data/processed/chemical_extraction_scispacy.csv"

GAZ = {
    "human": 9606, "homo sapiens": 9606, "ips-derived": 9606, "hipsc": 9606,
    "rat": 10116, "rattus norvegicus": 10116,
    "mouse": 10090, "mice": 10090, "murine": 10090,
    "zebrafish": 7955, "danio rerio": 7955,
    "drosophila": 7227, "drosophila melanogaster": 7227,
    "c. elegans": 6239, "caenorhabditis elegans": 6239
}
CHEM_DB = {
    "deltamethrin": {"CASRN": "52918-63-5", "DTXSID": "DTXSID3025404"},
    "bisphenol a": {"CASRN": "80-05-7", "DTXSID": "DTXSID3020206"},
    "pfos": {"CASRN": "1763-23-1", "DTXSID": "DTXSID6025720"}
}
CHEM_NAMES = list(CHEM_DB.keys())

CTX_SPECIES = re.compile(r"(cells?|primary|tissue|plasma|neuron|astrocyte|organoid|culture|in vivo|in vitro|larva|embryo|co-?culture)", re.I)
BLACKLIST_SPECIES = re.compile(r"(reference genome|human subjects committee|human studies approval)", re.I)
CTX_CHEMICALS = re.compile(r"(expos|treat|incubat|administ|dose|µm|mg/kg|concentration)", re.I)

def normalize_species(txt):
    s = txt.lower()
    for k, tax in GAZ.items():
        if k in s:
            canon = [n for n,t in GAZ.items() if t==tax and " " in n] or [k]
            return canon[0].title(), tax
    return txt, None

def extract_species(paragraph, pid, expected):
    if BLACKLIST_SPECIES.search(paragraph): 
        # Lower confidence if blacklist hit
        blacklist_hit = True
    else:
        blacklist_hit = False

    doc = nlp_species(paragraph)
    found = []

    # NER hits
    for ent in doc.ents:
        if ent.label_.lower() in {"species","organism"} or ent.text.lower() in GAZ:
            has_ctx = bool(CTX_SPECIES.search(paragraph))
            norm, tax = normalize_species(ent.text)
            conf = 0.8 if has_ctx else 0.5
            if ent.text.lower() in GAZ and has_ctx:
                conf = 1.0
            if blacklist_hit and conf < 1.0:
                conf -= 0.2
            found.append((ent.start_char, ent.end_char, ent.text, norm, tax, max(conf, 0.0), "ner"))

    # Gazetteer sweep
    low = paragraph.lower()
    for k, tax in GAZ.items():
        for m in re.finditer(rf"\b{re.escape(k)}\b", low):
            has_ctx = bool(CTX_SPECIES.search(paragraph))
            conf = 1.0 if has_ctx else 0.6
            if blacklist_hit and conf < 1.0:
                conf -= 0.2
            norm = [n for n,t in GAZ.items() if t==tax and " " in n] or [k]
            found.append((m.start(), m.end(), paragraph[m.start():m.end()], norm[0].title(), tax, max(conf,0.0), "gaz"))

    # dedup per offset e preferisci conf più alta
    by_span = {}
    for s,e,txt,norm,tax,conf,src in found:
        key = (s,e)
        if key not in by_span or conf > by_span[key][-2]:
            by_span[key] = (s,e,txt,norm,tax,conf,src)

    rows = []
    for (s,e),(s,e,txt,norm,tax,conf,src) in by_span.items():
        rows.append({
            "PARAGRAPH_ID": pid,
            "SPECIES_MENTION": txt,
            "SPECIES_NORM": norm,
            "NCBI_TAXID": tax,
            "CONFIDENCE": round(conf, 2),
            "SOURCE": src,
            "EXPECTED_RESULT": expected
        })
    return rows

def normalize_chemical(name):
    match = process.extractOne(name.lower(), CHEM_NAMES, score_cutoff=85)
    if not match:
        return {"CHEMICAL_NAME_NORM": name, "CASRN": None, "DTXSID": None}
    best = CHEM_DB[match[0]]
    return {"CHEMICAL_NAME_NORM": match[0], **best}

def extract_chemicals(paragraph, pid, expected):
    doc = nlp_chemicals(paragraph)
    results = []

    # 1) scispaCy NER
    for ent in doc.ents:
        if ent.label_ == "CHEMICAL" and CTX_CHEMICALS.search(paragraph):
            norm = normalize_chemical(ent.text)
            results.append({
                "PARAGRAPH_ID": pid,
                "CHEMICAL_NAME_RAW": ent.text,
                **norm,
                "SOURCE": "scispaCy",
                "EXPECTED_RESULT": expected
            })

    # 2) ChemDataExtractor fallback
    doc2 = Document(paragraph)
    for c in doc2.cems:
        if CTX_CHEMICALS.search(paragraph):
            norm = normalize_chemical(c.text)
            results.append({
                "PARAGRAPH_ID": pid,
                "CHEMICAL_NAME_RAW": c.text,
                **norm,
                "SOURCE": "ChemDataExtractor",
                "EXPECTED_RESULT": expected
            })

    # dedup
    seen = set()
    unique = []
    for r in results:
        key = r["CHEMICAL_NAME_NORM"]
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique

# Esempio batch
def run(paragraphs, out_csv_species, out_csv_chemicals):
    with open(out_csv_species, "w", newline="", encoding="utf-8") as f, open(out_csv_chemicals, "w", newline="", encoding="utf-8") as f2:
        w = csv.DictWriter(f, fieldnames=["PARAGRAPH_ID","SPECIES_MENTION","SPECIES_NORM","NCBI_TAXID","CONFIDENCE","SOURCE","EXPECTED_RESULT"])
        w.writeheader()
        w2 = csv.DictWriter(f2, fieldnames=["PARAGRAPH_ID","CHEMICAL_NAME_RAW","CHEMICAL_NAME_NORM","CASRN","DTXSID","SOURCE","EXPECTED_RESULT"])
        w2.writeheader()
        for p in paragraphs:
            pid = p.get("id") or str(uuid.uuid4())
            expected_species = p.get("expected") or ""
            for r in extract_species(p["text"], pid, expected_species):
                w.writerow(r)
            expected_chem = p.get("expected_chemical") or ""
            for chem in extract_chemicals(p["text"], pid, expected_chem):
                w2.writerow(chem)



if __name__ == "__main__":
    with open(IN_JSON, "r", encoding="utf-8") as f:
      paragraphs = json.load(f)

    run(paragraphs, OUT_CSV_SPECIES, OUT_CSV_CHEMICALS)
    print("Extraction completed.")
