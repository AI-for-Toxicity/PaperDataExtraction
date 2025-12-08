import json
from pathlib import Path
from MarkdownCleaner import MarkdownCleaner
from MarkdownDivider import MarkdownDivider
from PDFExtractor import PDFExtractor
from BioNERExtractor import BioNERExtractor

PDF_DIR = Path("test_data/raw/")
LABELS_DIR = Path("test_data/labels/")
OUTPUT_DIR = Path("test_data/processed/")

TEST = False

def test():
  bio_ner_extractor = BioNERExtractor()
  extractions = bio_ner_extractor.extract_paragraph(
    "4.3 Use of hNNF data on deltamethrin for the development of a putative AOP In 2021, the EFSA developed an IATA case study with the goal of including all available in vivo and in vitro data, among others the data generated within the DNT IVB for DNT hazard iden-tification for the Type II pyrethroid insecticide deltamethrin (Crofton and Mundy, 2021; Hernández-Jerez et al., 2021). Ep-idemiological studies revealed associations between childhood exposure to pyrethroids like deltamethrin and neurodevelopmen-tal disorders, e.g., attention deficit hyperactivity disorder or au-tism spectrum disorder (Oulhote and Bouchard, 2013; Shelton et al., 2014; Wagner-Schuman et al., 2015). As previously shown, deltamethrin negatively influenced 5 of 14 parameters describ-ing network function with “Number of spikes per network burst” as the most sensitive endpoint within the hNNF assay (BMC50 2.7 µM). Here, interference with voltage-gated sodium channels is the most commonly known MoA for pyrethroid insecticides like deltamethrin (Tapia et al., 2020), representing one of two molecular initiating events (MIE) within the stressor-specific AOP network (Fig. 7). This MIE is followed by key events 1-6 and 9, describing different cellular responses, like the disruption of sodium channel gate kinetics leading to disruption of action potential, that in the end cumulate in an impaired behavior-al function (adverse outcome). KE4 describes the alteration of neural network function as shown also by data assessed in the rNNF (BMC50 0.5 µM; Tab. 2) and hNNF assay. The 5-fold higher BMC of the hNNF assay compared to the rNNF assay might be explained by the different exposure paradigm and/or the different species as discussed above in more detail. Further-more, potential mechanisms or processes that are disrupted by a chemical agent can be revealed and used for the development of adverse outcome pathways (AOP) and also set a new focus for more hypothesis-driven in vivo studies (Hernández-Jerez et al., 2021). The postulated stressor-based AOP network (Fig. 7) is currently not included in the OECD AOP Wiki, but the EFSA"
  )
  print(json.dumps(extractions, indent=2, ensure_ascii=False))


if __name__ == "__main__":
  if TEST:
    test()
    exit(0)

  # Estrai testo e immagini dai PDF
  pdf_files = list(PDF_DIR.glob("*.pdf"))
  md_folder = "markdown"
  images_folder = "images"
  with PDFExtractor(pdf_files, OUTPUT_DIR, skip_existing=True, keep_divided_pdfs=True, divided_folder=PDF_DIR) as extractor:
    extractor.run_text_extraction(folder=md_folder)
    extractor.run_image_extraction(folder=images_folder)
    # Extract tables
  
  # Pulisci markdown estratto
  md_files = list((OUTPUT_DIR / md_folder).glob("*.md"))
  md_cleaned_folder = "cleaned_markdown"
  with MarkdownCleaner(md_files, OUTPUT_DIR, skip_existing=True) as cleaner:
    cleaner.clean_markdowns(folder=md_cleaned_folder)

  # Dividi markdown
  md_cleaned_files = list((OUTPUT_DIR / md_cleaned_folder).glob("*.md"))
  divided_md_folder = "divided_markdown"
  with MarkdownDivider(md_cleaned_files, OUTPUT_DIR, skip_existing=True) as divider:
    divided_markdowns = divider.divide_files(folder=divided_md_folder)
  
  # Processa markdown diviso con NER e BioMistral
  json_divided_files = list((OUTPUT_DIR / divided_md_folder).glob("*.json"))
  

  pass

