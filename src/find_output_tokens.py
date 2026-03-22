import json
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B", use_fast=True)

data = json.load(open("test_data/labels/scored/paper_0001_events.json", "r"))
max_count = 0
max_tokens = 0
for chunk in data["chunks"]:
  count = 0
  chunk_output = ""
  for ev in chunk["events"]:
    count += 1
    chunk_output += f"\"{ev['chemical']}\",\"{ev['event_type']}\",\"{ev['event_description_short']}\"\n"
  chunk_output += "### END\n"
  tokens = TOKENIZER(chunk_output, add_special_tokens=False)["input_ids"]
  if len(tokens) > max_tokens:
    max_tokens = len(tokens)
    max_count = count

print(f"Max tokens for a single chunk: {max_tokens} ({max_count} events)")