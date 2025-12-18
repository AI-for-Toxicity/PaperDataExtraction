#!/usr/bin/env python3
"""
Evaluate a BioMistral-7B (+ LoRA adapter) event extractor.

Metrics:
A) Token-level masked loss (normal LM loss) on the gold assistant text.
B) Generation metrics:
   - Exact string match (after canonicalization)
   - Ordered line match (order-sensitive)
   - Set-based micro Precision/Recall/F1 (order + duplicates invariant)

Expected assistant format (one per line):
  "chemical","event_type","short_description"

If no events: ideally empty output (no lines). If you used a sentinel like NO_EVENTS,
this script treats "NO_EVENTS" / "NO EVENT IDENTIFIED" / "NO AOP IDENTIFIED" as empty.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ----------------- I/O -----------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


# ----------------- Prompt + target handling -----------------

def merge_system_user(messages: List[Dict[str, str]]) -> Tuple[str, str]:
    """
    Returns (merged_user_content, assistant_content).
    Supports either:
      - messages: [system?, user, assistant]
      - messages already merged user->assistant
    """
    system_msg = None
    user_msg = None
    assistant_msg = None

    for m in messages:
        role = m.get("role")
        if role == "system":
            system_msg = m
        elif role == "user":
            user_msg = m
        elif role == "assistant":
            assistant_msg = m

    merged_parts = []
    if system_msg and system_msg.get("content"):
        merged_parts.append(system_msg["content"])
    if user_msg and user_msg.get("content"):
        merged_parts.append(user_msg["content"])
    merged_user = "\n\n".join(merged_parts).strip()

    assistant = (assistant_msg.get("content") if assistant_msg else "") or ""
    return merged_user, assistant


def canonicalize_events_text(text: str) -> str:
    """
    Canonical order: strip empty lines, normalize whitespace minimally, sort lines.
    This matches what you do during training.
    """
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Drop sentinel-y outputs if present
    sentinels = {"no_events", "no event identified", "no aop identified"}
    if len(lines) == 1 and lines[0].strip().lower() in sentinels:
        return ""
    lines.sort(key=lambda s: s.lower())
    return "\n".join(lines)


# ----------------- Parsing predicted/gold events -----------------

_SENTINELS_RE = re.compile(r"^\s*(NO_EVENTS|NO\s+EVENT\s+IDENTIFIED|NO\s+AOP\s+IDENTIFIED)\s*$", re.IGNORECASE)

def normalize_line(line: str) -> str:
    # Keep it strict-ish but not insane: trim and collapse internal spaces
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    return line


def parse_event_lines(text: str) -> List[str]:
    """
    Returns normalized lines (order-preserving), ignoring empties and sentinels.
    Does not enforce CSV correctness here; we treat each non-empty line as a "line item".
    """
    if not text:
        return []
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if _SENTINELS_RE.match(ln):
            continue
        out.append(normalize_line(ln))
    return out


def parse_csv_triplets(text: str) -> Set[Tuple[str, str, str]]:
    """
    Robust-ish parse: for each line, try csv.reader. If it fails, fall back to raw line set.
    Returns a set of (chemical, event_type, short_desc).
    """
    triples = set()
    lines = parse_event_lines(text)
    for ln in lines:
        try:
            row = next(csv.reader([ln], skipinitialspace=True))
            if len(row) < 3:
                # Not a valid triplet line, skip (or treat as raw)
                continue
            chem = row[0].strip()
            etype = row[1].strip()
            desc = row[2].strip()
            triples.add((chem, etype, desc))
        except Exception:
            continue
    return triples


# ----------------- Metrics -----------------

def micro_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def ordered_line_metrics(pred_lines: List[str], gold_lines: List[str]) -> Dict[str, float]:
    """
    Order-sensitive: same length + same lines in same positions.
    """
    if not gold_lines and not pred_lines:
        return {"ordered_exact": 1.0, "ordered_line_acc": 1.0}
    if not gold_lines:
        return {"ordered_exact": 0.0, "ordered_line_acc": 0.0}

    same_len = (len(pred_lines) == len(gold_lines))
    exact = 1.0 if same_len and all(p == g for p, g in zip(pred_lines, gold_lines)) else 0.0

    # line accuracy over gold positions (truncate to shortest)
    m = min(len(pred_lines), len(gold_lines))
    correct = sum(1 for i in range(m) if pred_lines[i] == gold_lines[i])
    acc = correct / len(gold_lines) if len(gold_lines) > 0 else 0.0
    return {"ordered_exact": exact, "ordered_line_acc": acc}


# ----------------- Model loading -----------------

def load_model(base_model: str, adapter_dir: str, load_in_4bit: bool, bf16: bool, device_map: str="auto"):
    print("[eval] loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    quant_cfg = None
    torch_dtype = torch.bfloat16 if bf16 else torch.float16

    if load_in_4bit:
        print("[eval] setting up 4-bit quant config...", flush=True)
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    print(f"[eval] loading base model (device_map={device_map})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        quantization_config=quant_cfg,
        torch_dtype=torch_dtype,
    )
    print("[eval] loading LoRA adapter...", flush=True)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    print("[eval] model ready.", flush=True)
    return model, tokenizer


# ----------------- Main eval loop -----------------

@torch.no_grad()
def masked_ce_loss(model, tokenizer, prompt_str: str, full_str: str, max_length: int) -> float:
    """
    Compute CE loss only on assistant tokens by masking prompt tokens.
    """
    full = tokenizer(full_str, truncation=True, max_length=max_length, return_tensors="pt")
    prompt = tokenizer(prompt_str, truncation=True, max_length=max_length, return_tensors="pt")

    input_ids = full["input_ids"].to(model.device)
    attn = full["attention_mask"].to(model.device)

    labels = input_ids.clone()
    prompt_len = prompt["input_ids"].shape[1]
    prompt_len = min(prompt_len, labels.shape[1])
    labels[:, :prompt_len] = -100

    out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
    return float(out.loss.detach().cpu().item())


@torch.no_grad()
def generate_answer(model, tokenizer, prompt_str: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    toks = tokenizer(prompt_str, return_tensors="pt")
    input_ids = toks["input_ids"].to(model.device)
    attn = toks["attention_mask"].to(model.device)

    do_sample = temperature > 0.0

    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_tokens = gen[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, type=str)
    ap.add_argument("--adapter_dir", required=True, type=str)
    ap.add_argument("--eval_file", required=True, type=str)

    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)  # 0 = greedy
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--bf16", action="store_true")

    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--save_preds", type=str, default="", help="optional path to write JSONL with per-example outputs")

    args = ap.parse_args()

    model, tokenizer = load_model(args.base_model, args.adapter_dir, args.load_in_4bit, args.bf16)

    data = read_jsonl(args.eval_file)
    if args.limit and args.limit > 0:
        data = data[:args.limit]

    print(f"[eval] loaded {len(data)} examples", flush=True)

    # Aggregate counters
    n = 0
    loss_sum = 0.0

    exact_match = 0

    ordered_exact_sum = 0.0
    ordered_line_acc_sum = 0.0

    # set-based micro
    tp = fp = fn = 0

    # optional output
    out_f = open(args.save_preds, "w", encoding="utf-8") if args.save_preds else None

    for ex in data:
        if (n % 10) == 0:
            print(f"[eval] example {n}/{len(data)}", flush=True)

        messages = ex.get("messages", ex.get("conversation", ex.get("data", None)))
        if messages is None:
            # If your jsonl format differs, you're allowed to be consistent for once
            raise RuntimeError("Example has no 'messages' field (or compatible alias).")

        merged_user, gold_assistant = merge_system_user(messages)

        # Prompt boundary (IMPORTANT): include assistant-start tokens here
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": merged_user}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Full string for loss: user + assistant content (no gen prompt)
        gold_assistant_canon = canonicalize_events_text(gold_assistant)
        full_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": merged_user},
             {"role": "assistant", "content": gold_assistant_canon}],
            tokenize=False,
            add_generation_prompt=False,
        )

        # 1) masked CE loss
        ce = masked_ce_loss(model, tokenizer, prompt_str, full_str, args.max_length)
        loss_sum += ce

        # 2) generation
        pred_text = generate_answer(
            model, tokenizer, prompt_str,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        pred_canon = canonicalize_events_text(pred_text)

        if pred_canon.strip() == gold_assistant_canon.strip():
            exact_match += 1

        # Ordered-line metrics (order sensitive)
        pred_lines = parse_event_lines(pred_canon)
        gold_lines = parse_event_lines(gold_assistant_canon)
        om = ordered_line_metrics(pred_lines, gold_lines)
        ordered_exact_sum += om["ordered_exact"]
        ordered_line_acc_sum += om["ordered_line_acc"]

        # Set-based metrics (order + duplicates invariant)
        pred_set = set(pred_lines)
        gold_set = set(gold_lines)
        tp_i = len(pred_set & gold_set)
        fp_i = len(pred_set - gold_set)
        fn_i = len(gold_set - pred_set)
        tp += tp_i
        fp += fp_i
        fn += fn_i

        if out_f:
            out_f.write(json.dumps({
                "id": ex.get("id", None),
                "loss": ce,
                "prompt": merged_user,
                "gold": gold_assistant_canon,
                "pred": pred_canon,
                "exact_match": (pred_canon.strip() == gold_assistant_canon.strip()),
                "ordered": om,
                "set_counts": {"tp": tp_i, "fp": fp_i, "fn": fn_i},
            }, ensure_ascii=False) + "\n")

        n += 1

    if out_f:
        out_f.close()

    avg_loss = loss_sum / max(1, n)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")

    set_prf = micro_prf(tp, fp, fn)

    print(json.dumps({
        "n_examples": n,
        "masked_ce_loss": avg_loss,
        "masked_ppl": ppl,
        "exact_match_rate": exact_match / max(1, n),
        "ordered_exact_rate": ordered_exact_sum / max(1, n),
        "ordered_line_acc": ordered_line_acc_sum / max(1, n),
        "set_micro_precision": set_prf["precision"],
        "set_micro_recall": set_prf["recall"],
        "set_micro_f1": set_prf["f1"],
        "set_micro_counts": {"tp": tp, "fp": fp, "fn": fn},
    }, indent=2))

    print("[eval] done.", flush=True)


if __name__ == "__main__":
    main()


'''
python eval.py \
  --base_model BioMistral/BioMistral-7B \
  --adapter_dir outputs/biomistral_mie_ke_ao_qlora \
  --eval_file train/test.jsonl \
  --load_in_4bit --bf16 \
  --max_new_tokens 256 \
  --save_preds eval_preds.jsonl
'''

'''
Notes (so you don’t accidentally gaslight yourself with metrics)
- Masked CE loss tells you: “How well can it reproduce the canonical target text?” It’s order-sensitive because text is order-sensitive.
- Set micro F1 tells you what you actually care about: “Did it extract the right event lines, regardless of order and duplicates?”
If you want, next step is adding a line-normalization mode for set matching (lowercase event_type, normalize quotes, etc.) so you can choose strict vs forgiving matching without lying to yourself.
'''