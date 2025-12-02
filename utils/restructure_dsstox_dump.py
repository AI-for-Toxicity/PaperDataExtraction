import pandas as pd
import re

def looks_like_cas(s: str) -> bool:
    s = s.strip()
    return bool(re.match(r'^\d{2,7}-\d{2}-\d$', s))

df = pd.read_csv("data/resources/dsstox_extraction.tsv", sep="\t", dtype=str).fillna("")

rows = []

for _, row in df.iterrows():
    base_name = row["name"].strip()
    iupac = row["iupac_name"].strip()
    identifier = row["identifier"].strip()
    
    casrn = row["casrn"].strip()
    dtxsid = row["dtxsid"].strip()
    ctd_id = row["ctd_id"].strip()
    source = row["source"].strip()

    # normalized names to avoid duplicates
    seen_names = set()
    seen = set()
    def add_name(n, d):
        n_clean = n.strip()
        if not n_clean:
            return
        key_n = n_clean.lower()
        key = (key_n, d.lower())
        if key not in seen:
          if key_n in seen_names:
            #warning, same name different DTXSID
            print(f"Warning: name '{n_clean}' associated with multiple DTXSIDs: '{dtxsid}' and another.")
          seen_names.add(key_n)
          seen.add(key)
          rows.append({
            "name": n_clean,
            "casrn": casrn,
            "dtxsid": dtxsid,
            "ctd_id": ctd_id,
            "source": source,
          })
            
    # primary name
    add_name(base_name, dtxsid)

    # iupac synonym
    if iupac and iupac.lower().strip() != base_name.lower().strip():
        add_name(iupac, dtxsid)

    # identifier synonyms
    if identifier:
        parts = [p.strip() for p in identifier.split("|") if p.strip()]
        if len(parts) > 1:
            # assume first part is CAS/ID
            for p in parts[1:]:
                if not looks_like_cas(p):
                    add_name(p, dtxsid)
        else:
            # single identifier entry
            p = parts[0]
            if not looks_like_cas(p):
                add_name(p, dtxsid)

print( f"Total unique names extracted: {len(rows)}" )
out_df = pd.DataFrame(rows)
out_df.to_csv("data/resources/output.tsv", sep="\t", index=False)
