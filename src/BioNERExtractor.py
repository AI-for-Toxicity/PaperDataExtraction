import csv
import re
import json
import spacy
from typing import Dict, Iterable, List, Tuple, Optional, Any
from chemdataextractor import Document
from rapidfuzz import process


CTX_SPECIES = re.compile(
    r"(cells?|primary|tissue|plasma|neuron|astrocyte|organoid|culture|in vivo|in vitro|larva|embryo|co-?culture)",
    re.I
)
BLACKLIST_SPECIES = re.compile(
    r"(reference genome|human subjects committee|human studies approval)",
    re.I
)
CTX_CHEMICALS = re.compile(
    r"(expos|treat|incubat|administ|dose|µm|mg/kg|concentration)",
    re.I
)
CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")

class SpeciesGazetteer:
  def __init__(self, species_gazetteer_path: str = "data/resources/species_gazetteer_ncbi.json") -> None:
    with open(species_gazetteer_path, "r", encoding="utf-8") as f:
      data = json.load(f)

    # name -> taxid
    self._name_to_taxid: Dict[str, int] = {
      k.lower(): int(v) for k, v in data["name_to_taxid"].items()
    }
    # taxid (str) -> canonical name
    self._taxid_to_canonical: Dict[str, str] = data["taxid_to_canonical"]

  # ----- name -> taxid -----

  def get_taxid(self, name: str) -> Optional[int]:
    """Replacement for self.species_name_to_taxid.get(name)."""
    if name is None:
      return None
    return self._name_to_taxid.get(name.lower())

  def has_name(self, name: str) -> bool:
    """Replacement for `name in self.species_name_to_taxid`."""
    if name is None:
      return False
    return name.lower() in self._name_to_taxid

  def iter_name_taxid(self) -> Iterable[Tuple[str, int]]:
    """Replacement for self.species_name_to_taxid.items()."""
    return self._name_to_taxid.items()

  # ----- taxid -> canonical -----

  def get_canonical(
    self,
    taxid: Any,
    default: Optional[str] = None,
  ) -> Optional[str]:
    """
    Replacement for:
      - self.species_taxid_to_canonical.get(str(tax))
      - self.species_taxid_to_canonical.get(str(tax), name)
    """
    key = str(taxid)
    if default is None:
      return self._taxid_to_canonical.get(key)
    return self._taxid_to_canonical.get(key, default)

class ChemicalKB:
    """
    Lightweight chemical KB over a TSV with columns:
    name, casrn, dtxsid, ctd_id, source

    Provides:
      - exact lookup by name or CAS
      - fuzzy lookup by name using RapidFuzz
    """

    def __init__(self, tsv_path: str = "data/resources/CTD_chemicals_enriched.tsv", score_cutoff: int = 85):
      self.tsv_path = tsv_path
      self.score_cutoff = score_cutoff

      self.by_name = {}   # name_lc -> [records]
      self.by_cas = {}    # casrn -> [records]
      self.by_dtxsid = {} # dtxsid -> [records]
      self.by_ctd = {}    # ctd_id -> [records]
      self.records = []   # list of all records
      self.name_list = [] # list of lowercase names for fuzzy

      self._load()

    def _load(self):
      with open(self.tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
          name = (row.get("name") or "").strip()
          if not name:
            continue

          rec = {
            "name": name,
            "name_lc": name.lower(),
            "casrn": (row.get("casrn") or "").strip() or None,
            "dtxsid": (row.get("dtxsid") or "").strip() or None,
            "ctd_id": (row.get("ctd_id") or "").strip() or None,
            "primary": (row.get("primary") or False) or False,
          }

          idx = len(self.records)
          self.records.append(rec)
          self.name_list.append(rec["name_lc"])

          self.by_name.setdefault(rec["name_lc"], []).append(rec)
          if rec["casrn"]:
            self.by_cas.setdefault(rec["casrn"], []).append(rec)
          if rec["dtxsid"]:
            self.by_dtxsid.setdefault(rec["dtxsid"], []).append(rec)
          if rec["ctd_id"]:
            self.by_ctd.setdefault(rec["ctd_id"], []).append(rec)

      print(f"[ChemicalKB] Loaded {len(self.records)} chemical records from {self.tsv_path}")

    def _pick_best(self, recs):
      """
      Heuristic: prefer records with DTXSID, then CTD, then anything.
      """
      if not recs:
        return None

      def score(r):
        s = 0
        if r.get("dtxsid"):
          s += 2
        if r.get("ctd_id"):
          s += 1
        return s

      return max(recs, key=score)

    def lookup(self, text: str):
      """
      Return best-matching record for a given text (name or CAS).
      """
      if not text:
        return None

      raw = text.strip()
      key = raw.lower()

      # 1) CAS-style direct hit
      if CAS_RE.match(raw):
        recs = self.by_cas.get(raw)
        if recs:
          return self._pick_best(recs)

      # 2) exact name hit
      if key in self.by_name:
        return self._pick_best(self.by_name[key])

      # 3) fuzzy name hit
      match = process.extractOne(key, self.name_list, score_cutoff=self.score_cutoff)
      if not match:
        return None

      # RapidFuzz returns (choice, score, index)
      _, score, idx = match
      rec = self.records[idx]
      return rec

    def find_primary_name(self, record: Dict[str, Any]) -> str:
      """
      Return the primary name of a chemical record.
      """
      # Find all matching rows by DTXSID / CASRN / CTD_ID
      dtxsid = record.get("dtxsid")
      casrn = record.get("casrn")
      ctd_id = record.get("ctd_id")
      if dtxsid and dtxsid in self.by_dtxsid:
        for rec in self.by_dtxsid[dtxsid]:
          if rec.get("primary"):
            return rec["name"]
          
      if casrn and casrn in self.by_cas:
        for rec in self.by_cas[casrn]:
          if rec.get("primary"):
            return rec["name"]
      
      if ctd_id and ctd_id in self.by_ctd:
        for rec in self.by_ctd[ctd_id]:
          if rec.get("primary"):
            return rec["name"]
      
      return ""

    def find_synonyms(self, record: Dict[str, Any]) -> List[str]:
      """
      Return a list of synonyms for a chemical record.
      """
      synonyms = set()

      dtxsid = record.get("dtxsid")
      casrn = record.get("casrn")
      ctd_id = record.get("ctd_id")
      if dtxsid and dtxsid in self.by_dtxsid:
        for rec in self.by_dtxsid[dtxsid]:
          if rec["name"] != record["name"]:
            synonyms.add(rec["name"])
          
      if casrn and casrn in self.by_cas:
        for rec in self.by_cas[casrn]:
          if rec["name"] != record["name"]:
            synonyms.add(rec["name"])
      
      if ctd_id and ctd_id in self.by_ctd:
        for rec in self.by_ctd[ctd_id]:
          if rec["name"] != record["name"]:
            synonyms.add(rec["name"])

      return list(synonyms)

class BioNERExtractor:
  def __init__(
    self,
    species_model: str = "en_ner_bionlp13cg_md",
    chemical_model: str = "en_ner_bc5cdr_md"
  ):
    """
    Initialize NER pipelines and resources.
    """
    self.nlp_species = spacy.load(species_model)
    self.nlp_chemicals = spacy.load(chemical_model)

    self.species_gazetteer = SpeciesGazetteer()
    self.chem_kb = ChemicalKB()

  def __enter__(self):
    return self
  
  # --------------------
  # Species helpers
  # --------------------

  def normalize_species(self, txt: str):
    """
    Normalize a species mention using NCBI-based gazetteer.

    - Exact lowercase match on name_to_taxid
    - Canonical name from taxid_to_canonical if available
    """
    s = txt.lower().strip()
    tax = self.species_gazetteer.get_taxid(s)
    if tax is None:
      # Fallback: nothing found, return original text
      return txt, None

    canon = self.species_gazetteer.get_canonical(tax)
    if not canon:
      canon = txt
    return canon, tax

  def extract_species(self, paragraph: str):
    if BLACKLIST_SPECIES.search(paragraph):
      blacklist_hit = True
    else:
      blacklist_hit = False

    doc = self.nlp_species(paragraph)
    found = []

    # 1) NER hits
    for ent in doc.ents:
      ent_txt = ent.text
      ent_label = ent.label_.lower()
      ent_norm = ent_txt.lower().strip()
      
      # Check: species/organism label OR exact match in NCBI gazetteer
      if ent_label in {"species", "organism"} or self.species_gazetteer.has_name(ent_norm):
        has_ctx = bool(CTX_SPECIES.search(paragraph))
        norm, tax = self.normalize_species(ent_txt)  # uses NCBI gazetteer
        conf = 0.8 if has_ctx else 0.5

        # If it's an exact gazetteer match + context, trust it more
        if self.species_gazetteer.has_name(ent_norm) and has_ctx:
          conf = 1.0

        if blacklist_hit and conf < 1.0:
          conf -= 0.2

        found.append(
          (ent.start_char, ent.end_char, ent_txt, norm, tax, max(conf, 0.0), "ner")
        )

    # 2) Gazetteer sweep (NCBI)
    low = paragraph.lower()
    for name, tax in self.species_gazetteer.iter_name_taxid():
        # small cheap filter to avoid regex on every single name
        if name not in low:
          continue

        pattern = rf"\b{re.escape(name)}\b"
        for m in re.finditer(pattern, low):
          has_ctx = bool(CTX_SPECIES.search(paragraph))
          conf = 1.0 if has_ctx else 0.6
          if blacklist_hit and conf < 1.0:
            conf -= 0.2

          canon = self.species_gazetteer.get_canonical(tax, default=name)
          found.append(
            (
              m.start(),
              m.end(),
              paragraph[m.start():m.end()],
              canon,
              tax,
              max(conf, 0.0),
              "gaz",
            )
          )

    # dedup per span, keep highest confidence
    by_span = {}
    for s, e, txt, norm, tax, conf, src in found:
      key = (s, e)
      if key not in by_span or conf > by_span[key][-2]:
        by_span[key] = (s, e, txt, norm, tax, conf, src)

    results = []
    for (s, e), (s, e, txt, norm, tax, conf, source) in by_span.items():
      results.append({
        "type": "species",
        "text": txt,
        "norm": norm,
        "ncbi_taxid": tax,
        "confidence": round(conf, 2),
        "source": source,
        "start_char": s,
        "end_char": e,
      })

    return results

  # --------------------
  # Chemical helpers
  # --------------------

  def _build_abbrev_map_for_paper(self, paragraphs: list[dict]) -> dict:
    """
    Build a mapping short -> long form for abbreviations at paper level,
    using the chemical pipeline + AbbreviationDetector.

    paragraphs: list of {"title": ..., "body": ...}
    """
    bodies = [(p.get("body") or "").strip() for p in paragraphs]
    big_text = "\n\n".join(bodies)
    abbrev_map: dict[str, str] = {}

    if not big_text:
        return abbrev_map

    doc = self.nlp_chemicals(big_text)

    # doc._.abbreviations is provided by AbbreviationDetector
    if not hasattr(doc._, "abbreviations"):
        return abbrev_map

    for abrv in doc._.abbreviations:
        short = abrv.text.strip()
        long = abrv._.long_form.text.strip()

        # basic sanity checks to avoid garbage
        if not short or not long:
            continue
        if len(short) > 15:  # not really an abbreviation anymore
            continue
        if len(long) <= len(short):
            continue

        # store lowercased key
        abbrev_map[short.lower()] = long

    return abbrev_map


  def normalize_chemical(self, name: str):
    rec = self.chem_kb.lookup(name)
    if not rec:
      # No match in KB -> keep raw name, no IDs
      return {
        "matched": name,
        "norm": None,
        "casrn": None,
        "dtxsid": None,
        "ctd_id": None,
        "synonyms": [],
      }

    primary_name = self.chem_kb.find_primary_name(rec)
    synonyms = self.chem_kb.find_synonyms(rec)
    return {
      "matched": name,
      "norm": primary_name,
      "casrn": rec["casrn"],
      "dtxsid": rec["dtxsid"],
      "ctd_id": rec["ctd_id"],
      "synonyms": synonyms,
    }

  def extract_chemicals(self, paragraph: str, abbrev_map: dict | None = None):
    doc = self.nlp_chemicals(paragraph)
    results = []

    # also collect local abbreviations in this paragraph
    local_abbrev_map: dict[str, str] = {}
    if hasattr(doc._, "abbreviations"):
        for abrv in doc._.abbreviations:
            short = abrv.text.strip()
            long = abrv._.long_form.text.strip()
            if short and long and len(short) <= 15 and len(long) > len(short):
                local_abbrev_map[short.lower()] = long

    # merge: local definitions override paper-level if conflict
    merged_abbrev = dict(abbrev_map or {})
    merged_abbrev.update(local_abbrev_map)

    if CTX_CHEMICALS.search(paragraph):
      # 1) scispaCy NER
      for ent in doc.ents:
        if ent.label_ == "CHEMICAL":
          raw_text = ent.text
          expanded = merged_abbrev.get(raw_text.lower())

          # decide what to normalize on
          norm_input = expanded if expanded else raw_text
          norm = self.normalize_chemical(norm_input)
          results.append({
            "type": "chemical",
            "text": ent.text,
            "matched": norm["matched"],
            "norm": norm["norm"],
            "casrn": norm["casrn"],
            "dtxsid": norm["dtxsid"],
            "ctd_id": norm["ctd_id"],
            "synonyms": norm["synonyms"],
            "source": "scispaCy+KB",
            "start_char": ent.start_char,
            "end_char": ent.end_char,
          })

      # 2) ChemDataExtractor fallback
      doc2 = Document(paragraph)
      for c in doc2.cems:
        raw_text = c.text
        expanded = merged_abbrev.get(raw_text.lower())
        norm_input = expanded if expanded else raw_text
        norm = self.normalize_chemical(norm_input)
        results.append({
          "type": "chemical",
          "text": c.text,
          "matched": norm["matched"],
          "norm": norm["norm"],
          "casrn": norm["casrn"],
          "dtxsid": norm["dtxsid"],
          "ctd_id": norm["ctd_id"],
          "synonyms": norm["synonyms"],
          "source": "ChemDataExtractor+KB",
          "start_char": None,
          "end_char": None,
        })

    # dedup by normalized name + source
    seen = set()
    unique = []
    for r in results:
      key = (r["norm"], r.get("dtxsid"), r.get("ctd_id"))
      if key not in seen:
        seen.add(key)
        unique.append(r)
    return unique

  # --------------------
  # Public API
  # --------------------

  def extract_paragraph(self, text: str, abbrev_map: dict | None = None):
    """
    Run both species + chemical extraction on a paragraph.
    Returns a list of NER dicts.
    """
    species = self.extract_species(text)
    chems = self.extract_chemicals(text, abbrev_map=abbrev_map)
    return species + chems

  def process_corpus(self, corpus: dict):
    """
    corpus format:
    {
      "paper_id_1": {
        "paragraphs": [
          {"title": "...", "body": "..."},
          ...
        ]
      },
      ...
    }

    Returns same structure, but with each paragraph enriched with:
    "ner": [ {extraction}, ... ]
    """
    out = {}

    for paper_id, paper_data in corpus.items():
      paragraphs = paper_data.get("paragraphs", [])
      new_paragraphs = []

      abbrev_map = self._build_abbrev_map_for_paper(paragraphs)

      for para in paragraphs:
        body = para.get("body", "") or ""
        ner = self.extract_paragraph(body, abbrev_map=abbrev_map) if body.strip() else []
        new_para = dict(para)
        new_para["ner"] = ner
        new_paragraphs.append(new_para)

      new_paper = dict(paper_data)
      new_paper["paragraphs"] = new_paragraphs
      out[paper_id] = new_paper

    return out

  def process_file(self, in_path: str, out_path: str):
    """
    Load JSON, process, save JSON.
    """
    with open(in_path, "r", encoding="utf-8") as f:
      corpus = json.load(f)

    result = self.process_corpus(corpus)

    with open(out_path, "w", encoding="utf-8") as f:
      json.dump(result, f, ensure_ascii=False, indent=2)

  def __exit__(self, exc_type, exc_value, traceback):
    pass
