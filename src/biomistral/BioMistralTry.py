# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path

MODEL_ID = "BioMistral/BioMistral-7B"

# 1) Load instruction prompt from file
prompt_path = Path("prompt.txt")
system_prompt = prompt_path.read_text(encoding="utf-8").strip()

# 2) Your paper text goes here (or load from another file)
text_path = Path("paper.txt")
paper_text = text_path.read_text(encoding="utf-8").strip()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# If you have a GPU, this is usually better than dumping everything on CPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

# 3) Build chat messages
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": paper_text},
]

# 4) Apply chat template
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

# device_map="auto" may shard the model; safest is to move inputs to the first device
# This avoids the common "tensors on different devices" headache.
first_device = next(model.parameters()).device
inputs = {k: v.to(first_device) for k, v in inputs.items()}

# 5) Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False,          # deterministic extraction is your friend
    temperature=0.0,          # extra anti-chaos
)

# 6) Decode only the newly generated tokens
generated = outputs[0][inputs["input_ids"].shape[-1]:]
print(tokenizer.decode(generated, skip_special_tokens=True))
