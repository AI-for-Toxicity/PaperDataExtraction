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
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb


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

    Ogni esempio è:
    {
      "messages": [
        {"role": "system", "content": ...},
        {"role": "user", "content": ...},
        {"role": "assistant", "content": ...}  # lista eventi CSV
      ]
    }

    Viene applicata la chat_template del tokenizer.
    La loss viene calcolata SOLO sui token della risposta dell'assistant.
    """

    def __init__(self, data, tokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        # sanity check: il modello deve avere una chat_template
        if self.tokenizer.chat_template is None:
            raise ValueError("Tokenizer has no chat_template. Set tokenizer.chat_template manually.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        messages = example["messages"]

        # canonicalizza l'output dell'assistant
        # (così la loss non dipende dall'ordine “sporco” salvato nel file)
        last = messages[-1]
        if last["role"] != "assistant":
            raise ValueError("Last message must be assistant.")
        canonical_assistant_text = canonicalize_events_text(last["content"])
        messages = messages[:-1] + [
            {
                "role": "assistant",
                "content": canonical_assistant_text,
            }
        ]

        # input con risposta
        full_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # input SENZA risposta (solo system+user), serve per mascherare la loss
        user_only_messages = [m for m in messages if m["role"] != "assistant"]
        user_only_str = self.tokenizer.apply_chat_template(
            user_only_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        full_tokens = self.tokenizer(
            full_str,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        user_only_tokens = self.tokenizer(
            user_only_str,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        input_ids = full_tokens["input_ids"][0]
        attention_mask = full_tokens["attention_mask"][0]

        # labels = input_ids con i token "pre-assistant" mascherati a -100
        labels = input_ids.clone()
        user_len = user_only_tokens["input_ids"].shape[1]

        # tutto ciò che è prima della risposta non contribuisce alla loss
        labels[:user_len] = -100

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

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

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
