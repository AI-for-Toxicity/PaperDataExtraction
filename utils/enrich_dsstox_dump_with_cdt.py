# Open ctd dump, relevant columns: (ChemicalName, ChemicalID, CasRN, PubChemCID, PubChemSID, DTXSID, CTDCuratedSynonyms)
# Open dsstox dump, format: (name, casrn, dtxsid)
# Group dsstox entries by dtxsid in a dictionary like {dtxsid: {casrn: , names: [] }} (casrn will be the same for all entries with same dtxsid)
# For each entry in the group:
## Try to find matching entry in ctd by dtxsid
## If found, edit ctd entry to add any missing names from dsstox entry to CTDCuratedSynonyms (if not already present)
## If not found, try to find matching entry in ctd by casrn and do the same
## If multiple entries found by casrn or dtxsid, add missing names to all of them
## Save updated ctd dump to new file
## Save updated dsstox dump to new file, removing any entries that were used to enrich ctd


#!/usr/bin/env python3
import csv
from collections import defaultdict

class CTDDump:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        with open(self.input_path, newline="", encoding="utf-8") as f:
          reader = csv.DictReader(f, delimiter="\t")
          fieldnames = reader.fieldnames
          rows = list(reader)
        
        # Build new view for easy searching
        self.data = []
        self.dtxsid_index = defaultdict(list)
        self.casrn_index = defaultdict(list)
        self.name_index = defaultdict(list)
        self.synonyms_index = defaultdict(list)
        index = 0
        for row in rows:
            synonyms = self.split_synonyms(row.get("CTDCuratedSynonyms", ""))
            synonyms += self.split_synonyms(row.get("MESHSynonyms", ""))
            new_row = {"name": row["ChemicalName"].strip(), "cid": row["ChemicalID"].strip(), "casrn": row["CasRN"].strip(), "pubcid": row["PubChemCID"].strip(), "pubsid": row["PubChemSID"].strip(), "dtxsid": row["DTXSID"].strip(), "synonyms": synonyms}
            self.data.append(new_row)
            dtxsid = new_row["dtxsid"].lower().strip()
            casrn = new_row["casrn"].lower().strip()
            name = new_row["name"].lower().strip()
            if dtxsid:
                self.dtxsid_index[dtxsid].append(index)
            if casrn:
                self.casrn_index[casrn].append(index)
            if name:
                self.name_index[name].append(index)
            for synonym in synonyms:
                self.synonyms_index[synonym].append(index)
            index += 1
    
    def split_synonyms(self, value):  
        return [s.strip() for s in value.split("|") if s.strip()]
    
    def join_synonyms(self, syn_list):
        return "|".join(syn_list)

    def add_synonyms(self, indices, new_names, dtxsid, casrn):
        for idx in indices:
            if dtxsid and not self.data[idx]["dtxsid"]:
                self.data[idx]["dtxsid"] = dtxsid
            if casrn and not self.data[idx]["casrn"]:
                self.data[idx]["casrn"] = casrn
            existing_syns = set(s.lower().strip() for s in self.data[idx]["synonyms"])
            for name in new_names:
                name = name.lower().strip()
                if name not in existing_syns:
                    self.data[idx]["synonyms"].append(name)

    def match_by_dtxsid(self, dtxsid: str):
        return self.dtxsid_index.get(dtxsid.lower().strip(), [])
    
    def match_by_casrn(self, casrn: str):
        return self.casrn_index.get(casrn.lower().strip(), [])
    
    def match_by_name(self, name: str):
        return self.name_index.get(name.lower().strip(), [])

    def match_by_synonym(self, synonym: str):
        return self.synonyms_index.get(synonym.lower().strip(), [])

    def add_data(self, new_data: list):
        for row in new_data:
            name = ""
            synonyms = []
            i = 0
            for name in row["names"]:
                if i == 0:
                    name = name.strip()
                else:
                    synonyms.append(name.strip())
                i += 1
            if not name:
                print("WTFFFFFFFF")
                exit(1)
            
            new_row = {
                "name": name.strip(),
                "cid": "",
                "casrn": row["casrn"].strip(),
                "pubcid": "",
                "pubsid": "",
                "dtxsid": row["dtxsid"].strip(),
                "synonyms": synonyms
            }

            self.data.append(new_row)

    def to_tsv(self):
        fieldnames = ["name", "cid", "casrn", "pubcid", "pubsid", "dtxsid", "synonyms"]
        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for row in self.data:
                out_row = {
                    "name": row["name"],
                    "cid": row["cid"],
                    "casrn": row["casrn"],
                    "pubcid": row["pubcid"],
                    "pubsid": row["pubsid"],
                    "dtxsid": row["dtxsid"],
                    "synonyms": self.join_synonyms(row["synonyms"])
                }
                writer.writerow(out_row)

    def to_tsv_vertical(self):
        fieldnames = ["name", "cid", "casrn", "pubcid", "pubsid", "dtxsid", "primary"]
        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for row in self.data:
                first_row = {
                    "name": row["name"],
                    "cid": row["cid"],
                    "casrn": row["casrn"],
                    "pubcid": row["pubcid"],
                    "pubsid": row["pubsid"],
                    "dtxsid": row["dtxsid"],
                    "primary": True
                }
                writer.writerow(first_row)
                for syn in row["synonyms"]:
                    out_row = {
                        "name": syn,
                        "cid": row["cid"],
                        "casrn": row["casrn"],
                        "pubcid": row["pubcid"],
                        "pubsid": row["pubsid"],
                        "dtxsid": row["dtxsid"],
                        "primary": False
                    }
                    writer.writerow(out_row)

class DSSTOXDump:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        with open(self.input_path, newline="", encoding="utf-8") as f:
          reader = csv.DictReader(f, delimiter="\t")
          fieldnames = reader.fieldnames
          rows = list(reader)

        self.data = {}
        for row in rows:
            if row["dtxsid"] not in self.data:    
              self.data[row["dtxsid"]] = {
                  "names": [row["name"].strip()],
                  "casrn": row["casrn"].strip(),
                  "dtxsid": row["dtxsid"].strip()
              }
            else:
              if row["name"].strip() not in self.data[row["dtxsid"]]["names"]:
                  self.data[row["dtxsid"]]["names"].append(row["name"].strip())
        
        # Flatten back to list for easier processing
        self.data = list(self.data.values())

    def remove_used_entries(self, used_dtxsids):
        self.data = [entry for entry in self.data if entry["dtxsid"] not in used_dtxsids]
    
    def to_tsv(self):
        fieldnames = ["name", "casrn", "dtxsid"]
        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for entry in self.data:
                for name in entry["names"]:
                    out_row = {
                        "name": name,
                        "casrn": entry["casrn"],
                        "dtxsid": entry["dtxsid"]
                    }
                    writer.writerow(out_row)
      


def main():
    ctd_dump = CTDDump("data/resources/CTD_chemicals.tsv", "data/resources/CTD_chemicals_enriched.tsv")
    dsstox_dump = DSSTOXDump("data/resources/dsstox_extraction.tsv", "data/resources/dsstox_extraction_unused.tsv")

    print("Indexed DTXSIDs in CTD dump:", len(ctd_dump.dtxsid_index))
    print("Indexed CASRNs in CTD dump:", len(ctd_dump.casrn_index))
    print("Indexed Names in CTD dump:", len(ctd_dump.name_index))
    print("Indexed Synonyms in CTD dump:", len(ctd_dump.synonyms_index))
    print("Total entries in DSSTOX dump:", len(dsstox_dump.data))
    print("\n\n#############\n\n")

    used_dtxsids = set()
    dtxsid_matched = 0
    casrn_matched = 0
    name_matched = 0
    secondary_name_matched = 0
    synonym_matched = 0
    total_entries = len(dsstox_dump.data)
    for entry in dsstox_dump.data:
        dtxsid = entry["dtxsid"]
        casrn = entry["casrn"]
        names = entry["names"]

        matched_indices = ctd_dump.match_by_dtxsid(dtxsid)
        if matched_indices:
            if len(matched_indices) > 1:
                print(f"Warning: multiple matches by DTXSID {dtxsid}")
            ctd_dump.add_synonyms(matched_indices, names, dtxsid, casrn)
            used_dtxsids.add(dtxsid)
            dtxsid_matched += 1
            if "deltamethrin" in [name.lower().strip() for name in names]:
                print(f"Matched deltamethrin DTXSID {dtxsid} by DTXSID")
            continue
        
        matched_indices = ctd_dump.match_by_casrn(casrn)
        if matched_indices:
            if len(matched_indices) > 1:
                print(f"Warning: multiple matches by CASRN {casrn} (DTXSID {dtxsid})")
            ctd_dump.add_synonyms(matched_indices, names, dtxsid, casrn)
            used_dtxsids.add(dtxsid)
            casrn_matched += 1
            if "deltamethrin" in [name.lower().strip() for name in names]:
                print(f"Matched deltamethrin DTXSID {dtxsid} by CASRN")
            continue
        
        i = 0
        name_matched_flag = False
        for name in names:
            matched_indices = ctd_dump.match_by_name(name)
            if matched_indices:
                # Remove every index that maps a substance with different DTXSID or CASRN
                matched_indices = [idx for idx in matched_indices if ctd_dump.data[idx]["dtxsid"] != "" or ctd_dump.data[idx]["casrn"] != ""]
                if len(matched_indices) == 0:
                    continue
                if len(matched_indices) > 1:
                    print(f"Warning: multiple matches by name {name} (DTXSID {dtxsid})")
                ctd_dump.add_synonyms(matched_indices, names, dtxsid, casrn)
                used_dtxsids.add(dtxsid)
                name_matched_flag = True
                if "deltamethrin" in [name.lower().strip() for name in names]:
                  print(f"Matched deltamethrin DTXSID {dtxsid} by Name")
                if i == 0: name_matched += 1
                else: secondary_name_matched += 1
                break
            i += 1
        if name_matched_flag:
            continue
        
        for name in names:
            matched_indices = ctd_dump.match_by_synonym(name)
            if matched_indices:
                # Remove every index that maps a substance with different DTXSID or CASRN
                matched_indices = [idx for idx in matched_indices if ctd_dump.data[idx]["dtxsid"] != "" or ctd_dump.data[idx]["casrn"] != ""]
                if len(matched_indices) == 0:
                    continue
                if len(matched_indices) > 1:
                    print(f"Warning: multiple matches by synonym {name} (DTXSID {dtxsid})")
                ctd_dump.add_synonyms(matched_indices, names, dtxsid, casrn)
                used_dtxsids.add(dtxsid)
                synonym_matched += 1
                if "deltamethrin" in [name.lower().strip() for name in names]:
                    print(f"Matched deltamethrin DTXSID {dtxsid} by Synonym")
                break
    
    print("\n\n#############\n\n")
    print(f"Matched {dtxsid_matched} entries by DTXSID.")
    print(f"Matched {casrn_matched} entries by CASRN.")
    print(f"Matched {name_matched} entries by Primary Name.")
    print(f"Matched {secondary_name_matched} entries by Secondary Name.")
    print(f"Matched {synonym_matched} entries by Synonym.")
    print(f"Remaining unmatched entries: {total_entries - len(used_dtxsids)} out of {total_entries}.")

    ctd_total_entries_before = len(ctd_dump.data)
    dsstox_dump.remove_used_entries(used_dtxsids)
    ctd_dump.add_data(list(dsstox_dump.data))
    total_entries_after = len(ctd_dump.data)

    print("\n\n#############\n\n")
    print("Initial DSSTOX entries:", total_entries)
    print("Remaining DSSTOX entries after removing used ones:", total_entries - len(used_dtxsids))
    print("Initial CTD entries:", ctd_total_entries_before)
    print("Saving enriched CTD dump with", total_entries_after, "entries.")
    
    ctd_dump.to_tsv_vertical()

if __name__ == "__main__":
    main()
