import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

class PadWithLabels:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # pad input_ids/attention_mask
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        # tokenizer.pad will pad "labels" too, but with pad_token_id, not -100
        if "labels" in batch:
            labels = batch["labels"]
            labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

# ------------- DATASET UTILS ------------- #

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def canonicalize_events_text(text: str) -> str:
    """
    Prende l'output dell'assistant (righe CSV) e lo trasforma
    in una rappresentazione canonica:
    - split per righe
    - rimozione righe vuote
    - sort alfabetico
    - join con '\n'

    Questo rende la loss "order-invariant" rispetto all'ordine originale degli eventi,
    perché alleni sempre verso una forma ordinata del set.
    """
    if text is None:
        return ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    lines = sorted(lines)
    return "\n".join(lines)


class EventsChatDataset(Dataset):
    """
    Dataset per training chat-like di BioMistral:

    Ogni esempio nel file è:
    {
      "messages": [
        {"role": "system", "content": ...},
        {"role": "user", "content": ...}
      ]
    }

    La loss viene calcolata SOLO sui token della risposta.
    """

    def __init__(self, data, tokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.tokenizer.chat_template is None:
            raise ValueError("Tokenizer has no chat_template. Set tokenizer.chat_template manually.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        raw_messages = example["messages"]

        if len(raw_messages) < 2:
            raise ValueError(f"Example {idx} has too few messages: {raw_messages}")

        # estrai system (opzionale), user e assistant
        system_msg = None
        user_msg = None
        assistant_msg = None

        for m in raw_messages:
            if m["role"] == "system":
                system_msg = m
            elif m["role"] == "user":
                user_msg = m
            elif m["role"] == "assistant":
                assistant_msg = m

        if user_msg is None or assistant_msg is None:
            raise ValueError(f"Example {idx} missing user or assistant message: {raw_messages}")

        # canonicalizza la risposta dell'assistant (ordine eventi)
        canonical_assistant_text = canonicalize_events_text(assistant_msg["content"])

        # mergia system + user in un unico prompt utente
        merged_user_content_parts = []
        if system_msg is not None and system_msg.get("content"):
            merged_user_content_parts.append(system_msg["content"])
        if user_msg.get("content"):
            merged_user_content_parts.append(user_msg["content"])

        merged_user_content = "\n\n".join(merged_user_content_parts).strip()

        # nuova conversazione SOLO con ruoli user/assistant
        conv_messages = [
            {"role": "user", "content": merged_user_content},
            {"role": "assistant", "content": canonical_assistant_text},
        ]

        # stringa completa (user + assistant)
        full_str = self.tokenizer.apply_chat_template(
            conv_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Build the *exact* prompt boundary: user + assistant header/start tokens
        prompt_str = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": merged_user_content}],
            tokenize=False,
            add_generation_prompt=True,   # IMPORTANT: includes assistant-start tokens
        )

        full_tokens = self.tokenizer(
            full_str,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        prompt_tokens = self.tokenizer(
            prompt_str,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        input_ids = full_tokens["input_ids"][0]
        attention_mask = full_tokens["attention_mask"][0]

        labels = input_ids.clone()
        prompt_len = prompt_tokens["input_ids"].shape[1]

        # Safety: in case truncation makes prompt_len exceed full length
        prompt_len = min(prompt_len, labels.shape[0])

        # Mask everything up to the generation start; loss only on assistant content tokens
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ------------- MODEL / QLOLA SETUP ------------- #

def get_bnb_config():
    return bnb.nn.Linear4bit(
        16, 16, bias=False,
        compute_dtype=torch.bfloat16,
    )  # placeholder per forzare l'import; usato solo per assicurare bitsandbytes


def load_model_and_tokenizer(base_model: str):
    """
    Carica BioMistral in 4-bit NF4 + prepara QLoRA.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Config 4-bit corretta con BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Step standard per QLoRA
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    return model, tokenizer


# ------------- TRAINER (LOSS STANDARD, METRIC SET-BASED POSSIBILE A PARTE) ------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True,
                        help="BioMistral base model, es. BioMistral/BioMistral-7B")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.base_model)

    train_data = read_jsonl(args.train_file)
    eval_data = read_jsonl(args.eval_file)

    train_dataset = EventsChatDataset(train_data, tokenizer, max_length=args.max_length)
    eval_dataset = EventsChatDataset(eval_data, tokenizer, max_length=args.max_length)

    data_collator = PadWithLabels(tokenizer)
    #DataCollatorForLanguageModeling(
    #    tokenizer=tokenizer,
    #    mlm=False,
    #)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=True,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

'''
python src/train.py \
  --base_model BioMistral/BioMistral-7B \
  --train_file train/train.jsonl \
  --eval_file train/test.jsonl \
  --output_dir outputs/biomistral_mie_ke_ao_qlora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --max_length 2048
'''

'''
If you see the model start outputting garbage formats after a few hundred steps, LR is usually the first suspect. Common stable range is 1e-4 to 3e-5 for instruction-ish extraction tasks, especially with noisy supervision.

Use early stopping behavior without adding more machinery:
- run 3 epochs, check set-F1 on test (your eval script),
- if it’s still improving and not getting more hallucination-y, go to 5–7 epochs,
- if test set-F1 peaks early and then drops, you were already overtraining.
Practical tweaks I’d actually recommend
If your dataset is < ~5k examples:
- try LR = 1e-4 first (or 5e-5 if it’s really small / very noisy)
- consider 5 epochs max unless test metrics keep improving
If your dataset is > ~10k examples:
- 2e-4 can be fine, and 3 epochs may even be enough.
And regardless: don’t judge by loss alone. Judge by set-based F1, because that’s your real target behavior.
'''
