from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pathlib, re, csv, sys

MODEL = "microsoft/biogpt"  # swap to any local biomed model you prefer

FEWSHOT = """You extract toxicology facts and output CSV rows with columns:
PAPER_NAME,SUBSTANCE,EVENT,SPECIES,CONCENTRATION_VALUE,CONCENTRATION_UNIT,RESPONSE,BMAD,NOAEL,BMD,ASSAY_NAME,ASSAY_TYPE,NOTES

If a value is missing in the text, leave it blank. Output only CSV rows, one per finding.

Example:
Text: "Chlorpyrifos induced neuronal apoptosis in zebrafish at 2.3 µM."
CSV:
P1,Chlorpyrifos,neuronal apoptosis,Danio rerio,2.3,µM,,, , , , ,llm_v1
"""

INPUT_FILE = "data/raw/markdown/paper_0001/paper_0001.md"
OUTPUT_FILE = "data/raw/markdown/paper_0001/biogpt_extracted_facts.csv"

def main():
    txt_path = pathlib.Path(INPUT_FILE)
    out_path = pathlib.Path(OUTPUT_FILE)
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    prompt = FEWSHOT + f"\nText: {text}\nCSV:\n"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL)
    gen = pipeline("text-generation", model=mdl, tokenizer=tok, device_map="auto", max_new_tokens=300)
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    lines = []
    for i, chunk in enumerate(chunks):
        prompt = FEWSHOT + f"\nText: {chunk}\nCSV:\n"
        out = gen(prompt)[0]["generated_text"]
        # crude parse: keep lines that look like CSV with at least 5 commas
        lines.extend(ln.strip() for ln in out.splitlines() if ln.count(",") >= 5)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
