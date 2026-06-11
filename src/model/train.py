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
    Normalizes assistant output to order-invariant form: strip empty lines, sort, rejoin.
    Applied during training so the model always learns toward a sorted canonical form.
    """
    if text is None:
        return ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    lines = sorted(lines)
    return "\n".join(lines)


class EventsChatDataset(Dataset):
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

        canonical_assistant_text = canonicalize_events_text(assistant_msg["content"])

        merged_user_content_parts = []
        if system_msg is not None and system_msg.get("content"):
            merged_user_content_parts.append(system_msg["content"])
        if user_msg.get("content"):
            merged_user_content_parts.append(user_msg["content"])

        merged_user_content = "\n\n".join(merged_user_content_parts).strip()

        conv_messages = [
            {"role": "user", "content": merged_user_content},
            {"role": "assistant", "content": canonical_assistant_text},
        ]

        full_str = self.tokenizer.apply_chat_template(
            conv_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Build the exact prompt boundary including assistant-start tokens
        prompt_str = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": merged_user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )

        full_tokens = self.tokenizer(
            full_str,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        # Do NOT truncate the prompt: we need the true prompt token count to correctly
        # place the label mask boundary. The min() below handles the case where the prompt
        # alone exceeds max_length (all labels become -100 and that example contributes no
        # gradient — acceptable). Using truncation=True here would corrupt the mask.
        prompt_tokens = self.tokenizer(
            prompt_str,
            truncation=False,
            padding=False,
            return_tensors="pt",
        )

        input_ids = full_tokens["input_ids"][0]
        attention_mask = full_tokens["attention_mask"][0]

        labels = input_ids.clone()
        prompt_len = prompt_tokens["input_ids"].shape[1]

        # Safety: in case truncation makes prompt_len exceed full length
        prompt_len = min(prompt_len, labels.shape[0])

        # Mask prompt tokens; loss computed only on assistant content tokens
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class PadWithLabels:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        # tokenizer.pad pads "labels" with pad_token_id, not -100.
        # Use attention_mask to identify padding positions — NOT the token value,
        # because pad_token_id == eos_token_id, so replacing by value would also mask
        # the real EOS at the end of the assistant response.
        if "labels" in batch and "attention_mask" in batch:
            labels = batch["labels"]
            labels[batch["attention_mask"] == 0] = -100
            batch["labels"] = labels
        return batch


# ------------- MODEL / QLORA SETUP ------------- #

def load_model_and_tokenizer(base_model: str, load_in_4bit: bool = False, lora_r: int = 32):
    """
    Loads the model with LoRA.

    load_in_4bit=False (default): loads in bf16 — recommended when VRAM >= 20GB.
    load_in_4bit=True: QLoRA (4-bit NF4) for low-VRAM environments.
    lora_r: LoRA rank. Higher = more capacity.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
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
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.15,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    return model, tokenizer


# ------------- TRAINING ------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Use QLoRA (4-bit NF4). Omit for bf16 LoRA (recommended if VRAM >= 20GB).")
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA rank. lora_alpha is set to 2*lora_r automatically.")
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
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        args.base_model,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
    )

    train_data = read_jsonl(args.train_file)
    eval_data = read_jsonl(args.eval_file)

    train_dataset = EventsChatDataset(train_data, tokenizer, max_length=args.max_length)
    eval_dataset = EventsChatDataset(eval_data, tokenizer, max_length=args.max_length)

    data_collator = PadWithLabels(tokenizer)

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
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        save_total_limit=3,
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
