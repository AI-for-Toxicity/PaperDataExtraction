import json
from pathlib import Path


def load_taxid_to_rank(nodes_path: str):
    """
    Parse nodes.dmp -> {taxid (int): rank (str)}
    """
    taxid_to_rank = {}
    with open(nodes_path, "r", encoding="utf-8") as f:
        for line in f:
            # Format: tax_id | parent_tax_id | rank | ...
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            taxid_str, _, rank = parts[:3]
            try:
                taxid = int(taxid_str)
            except ValueError:
                continue
            taxid_to_rank[taxid] = rank
    return taxid_to_rank


def build_ncbi_species_gazetteer(
    names_path: str,
    nodes_path: str,
    out_path = None,
    rank_whitelist = ("species", "subspecies", "strain", "no rank"),
    name_classes = ("scientific name", "synonym", "genbank common name", "common name"),
):
    """
    Build a species gazetteer from NCBI Taxonomy.

    Output structure:
    {
      "name_to_taxid": { "homo sapiens": 9606, "human": 9606, ... },
      "taxid_to_canonical": { "9606": "Homo sapiens", ... }
    }
    """
    taxid_to_rank = load_taxid_to_rank(nodes_path)

    name_to_taxid = {}
    taxid_to_canonical = {}

    with open(names_path, "r", encoding="utf-8") as f:
        for line in f:
            # Format: tax_id | name_txt | unique_name | name_class | ...
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue

            taxid_str, name_txt, _, name_class = parts[:4]
            try:
                taxid = int(taxid_str)
            except ValueError:
                continue

            rank = taxid_to_rank.get(taxid)
            if rank_whitelist and rank not in rank_whitelist:
                continue

            if name_class not in name_classes:
                continue

            name_norm = name_txt.strip().lower()
            if not name_norm:
                continue

            # Map name -> taxid (first wins; you can change policy if needed)
            if name_norm not in name_to_taxid:
                name_to_taxid[name_norm] = taxid

            # Prefer the scientific name as canonical label
            if name_class == "scientific name":
                taxid_to_canonical[str(taxid)] = name_txt.strip()

    gaz = {
        "name_to_taxid": name_to_taxid,
        "taxid_to_canonical": taxid_to_canonical,
    }

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(gaz, f, ensure_ascii=False)

    return gaz


if __name__ == "__main__":
    NAMES = "data/resources/names.dmp"
    NODES = "data/resources/nodes.dmp"
    OUT   = "data/resources/species_gazetteer_ncbi.json"

    build_ncbi_species_gazetteer(NAMES, NODES, OUT)
    print(f"Saved gazetteer to {OUT}")
