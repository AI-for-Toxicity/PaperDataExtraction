import xml.etree.ElementTree as ET
import json
import re
import html
from bs4 import BeautifulSoup

def strip_html_bs4(text: str) -> str:
    if not text:
        return text
    return BeautifulSoup(text, "html.parser").get_text(" ")

_WS_RE = re.compile(r"\s+")
_ZW_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")  # zero-width chars
_CIT_RE = re.compile(r"\[(?:\d+|citation needed)\]")  # optional

def clean_paragraph(text: str, remove_citations: bool = True):
    if not text:
        return None

    # Decode entities like &nbsp; &amp;
    text = html.unescape(text)

    # Kill zero-width junk
    text = _ZW_RE.sub("", text)

    # Replace non-breaking space with normal space
    text = text.replace("\xa0", " ")

    # Optional: remove [12] style citations
    if remove_citations:
        text = _CIT_RE.sub("", text)

    # Normalize whitespace (newlines/tabs/multiple spaces -> single space)
    text = _WS_RE.sub(" ", text).strip()

    # Optional: clean spacing before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    return text or None

def remove_namespaces(tree):
    """Strips namespaces from all elements to make querying straightforward."""
    for elem in tree.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
    return tree

def safe_clean_text(element, path, remove_citations: bool = True):
    found = element.find(path)
    if found is None:
        return None

    # Join *all* descendant text nodes into one string
    raw = " ".join(t.strip() for t in found.itertext() if t and t.strip())

    return clean_paragraph(strip_html_bs4(raw), remove_citations=remove_citations)

def parse_aop_xml(xml_file_path, output_json_path):
    print(f"Loading and parsing XML: {xml_file_path}...")
    tree = ET.parse(xml_file_path)
    root = remove_namespaces(tree).getroot()

    # --- PASS 1: Build Global Dictionaries for Events and KERs ---
    print("Building global Key Event and Relationship dictionaries...")
    
    chemicals = {}
    for chem in root.findall('.//chemical'):
        chem_id = chem.get('id')
        inchi = safe_clean_text(chem, 'inchi')
        casrn = safe_clean_text(chem, 'casrn')
        jchem_inchi_key = safe_clean_text(chem, 'jchem-inchi-key')
        indigo_inchi_key = safe_clean_text(chem, 'indigo-inchi-key')
        iupac_name = safe_clean_text(chem, 'iupac-name')
        preferred_name = safe_clean_text(chem, 'preferred-name')
        formula = safe_clean_text(chem, 'formula')
        dsstox_id = safe_clean_text(chem, 'dsstox-id')
        synonyms = [safe_clean_text(syn, 'synonym') for syn in chem.findall('.//synonyms/synonym')]
        
        if chem_id:
            chemicals[chem_id] = {
                "inchi": inchi,
                "casrn": casrn,
                "jchem_inchi_key": jchem_inchi_key,
                "indigo_inchi_key": indigo_inchi_key,
                "iupac_name": iupac_name,
                "preferred_name": preferred_name,
                "formula": formula,
                "dsstox_id": dsstox_id,
                "synonyms": [syn for syn in synonyms if syn]
            }

    stressors = {}
    for stressor in root.findall('.//stressor'):
        stressor_id = stressor.get('id')
        name = safe_clean_text(stressor, 'name')
        chemicals_list = []
        for chem in stressor.findall('.//chemicals/chemical-initiator'):
            chem_id = chem.get('chemical-id')
            if chem_id and chem_id in chemicals:
                chemicals_list.append(chemicals[chem_id])
        
        if stressor_id:
            stressors[stressor_id] = {
                "name": name,
                "chemicals": chemicals_list
            }

    taxonomies = {}
    for tax in root.findall('.//taxonomy'):
        tax_id = tax.get('id')
        if tax_id:
            taxonomies[tax_id] = {
                "name": safe_clean_text(tax, 'name') or safe_clean_text(tax, 'scientific-name')
            }

    global_events = {}
    for ke in root.findall('.//key-event'):
        ke_id = ke.get('id')
        if ke_id:
            global_events[ke_id] = {
                "title": safe_clean_text(ke, 'title'),
                "short_name": safe_clean_text(ke, 'short-name'),
                "description": safe_clean_text(ke, 'description'),
                #"measurement-methodology": safe_clean_text(ke, 'measurement-methodology'),
                #"organ-term": safe_clean_text(ke, 'organ-term'),
                #"cell-term": safe_clean_text(ke, 'cell-term'),
                #"applicability": safe_clean_text(ke, 'applicability'),
                "biological_organization": safe_clean_text(ke, 'biological-organization-level')
            }

    # --- PASS 2: Extract AOPs and Hydrate with Global Data ---
    print("Extracting AOPs...")
    extracted_aops = []
    
    for aop in root.findall('.//aop'):
        aop_id = aop.get('id')
        
        # 1. Base Information
        aop_data = {
            "aop_id": aop_id,
            "title": safe_clean_text(aop, 'title'),
            "short_name": safe_clean_text(aop, 'short-name'),
            "status": safe_clean_text(aop, 'status'),
            "abstract": safe_clean_text(aop, 'abstract'),
            "context": safe_clean_text(aop, 'background'),
            "applicability": {
                "sex": [],
                "taxonomy": [],
                "life_stage": []
            },
            "stressors": [],
            "events": {
                "MIE": [],
                "KE": [],
                "AO": []
            }
        }

        # 2. Applicability (Taxonomy and Life Stages)
        for sex in aop.findall('.//applicability/sex'):
            name = safe_clean_text(sex, 'sex')
            evidence = safe_clean_text(sex, 'evidence')
            if name:
                aop_data["applicability"]["sex"].append({
                    "sex": name,
                    "evidence": evidence
                })
                
        for life_stage in aop.findall('.//applicability/life-stage'):
            name = safe_clean_text(life_stage, 'life-stage')
            evidence = safe_clean_text(life_stage, 'evidence')
            if name:
                aop_data["applicability"]["life_stage"].append({
                    "life_stage": name,
                    "evidence": evidence
                })

        for tax in aop.findall('.//applicability/taxonomy'):
            tax_id = tax.get('taxonomy-id')
            tax_info = {
                "name": taxonomies.get(tax_id, {}).get("name"),
                "evidence": safe_clean_text(tax, 'evidence')
            }
            if tax_id:
                aop_data["applicability"]["taxonomy"].append(tax_info)

        # 3. Key Events (Hydrate with Global Dictionary)
        for mie in aop.findall('.//molecular-initiating-event'):
            mie_id = mie.get('key-event-id')
            mie_info = {
                "event_id": mie_id,
                "type": "MIE"
            }
            if mie_id in global_events:
                mie_info.update(global_events[mie_id])
                
            aop_data["events"]["MIE"].append(mie_info)

        for ke in aop.findall('.//key-events/key-event'):
            ke_id = ke.get('key-event-id')
            
            # Merge local link info with global definition
            event_info = {
                "event_id": ke_id,
                "type": "KE"
            }
            if ke_id in global_events:
                event_info.update(global_events[ke_id])
                
            aop_data["events"]["KE"].append(event_info)

        for ao in aop.findall('.//adverse-outcome'):
            ao_id = ao.get('key-event-id')
            ao_info = {
                "event_id": ao_id,
                "type": "AO"
            }
            if ao_id in global_events:
                ao_info.update(global_events[ao_id])
                
            aop_data["events"]["AO"].append(ao_info)

        # 4. Stressors (Chemicals)
        for stressor in aop.findall('.//aop-stressors/aop-stressor'):
            stressor_id = stressor.get('stressor-id')
            evidence = safe_clean_text(stressor, 'evidence')
            if stressor_id in stressors:
                aop_data["stressors"].append({
                    "name": stressors[stressor_id].get("name"),
                    "chemicals": stressors[stressor_id].get("chemicals"),
                    "evidence": evidence
                })

        extracted_aops.append(aop_data)

    # --- FINAL: Save to JSON ---
    print(f"Extraction complete. Found {len(extracted_aops)} AOPs.")
    print(f"Saving to {output_json_path}...")
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_aops, f, indent=2, ensure_ascii=False)
        
    print("Done!")

# --- Execution ---
if __name__ == "__main__":
    # Replace these filenames with your actual file paths
    INPUT_XML = "new_data/aop-wiki-xml-2026-01-01.xml"  
    OUTPUT_JSON = "new_data/aop_extracted_data.json"
    
    parse_aop_xml(INPUT_XML, OUTPUT_JSON)
