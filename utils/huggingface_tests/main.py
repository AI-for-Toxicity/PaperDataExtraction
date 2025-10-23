import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForCausalLM, pipeline

def biobert_ner(text):
  model_name = "kamalkraj/BioBERT-NER-BIO"

  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForTokenClassification.from_pretrained(model_name)

  ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

  return ner(text)

def biogpt_ner(text):
  model_name = "microsoft/biogpt"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(model_name)

  inputs = tokenizer(text, return_tensors="pt")
  output = model.generate(**inputs, max_new_tokens=60)

  return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
  sample_text = "Aspirin is used to reduce fever and relieve mild to moderate pain from conditions such as muscle aches, toothaches, common cold, and headaches."

  print("BioBERT NER Results:")
  # biobert_results = biobert_ner(sample_text)
  #for entity in biobert_results:
  #    print(entity)

  print("\nBioGPT NER Results:")
  biogpt_results = biogpt_ner(sample_text)
  print(biogpt_results)
