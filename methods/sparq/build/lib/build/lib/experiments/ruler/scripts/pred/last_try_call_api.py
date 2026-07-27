#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare prediction jsonl with field `pred`.

dataset jsonl (per line):
{
  "index": int,
  "input": str,
  "outputs": [str],
  ... (optional: others/truncation/length)
}

prediction jsonl (per line):
{
  "index": int,
  "input": str,
  "outputs": [str],
  "pred": str,
  ... (optional: others/truncation/length)
}
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from llminference.myexperiments import Sparsity,SparsityMethods


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def read_existing_pred_indices(path: Path) -> set:
    if not path.exists():
        return set()
    indices = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "index" in obj:
                    indices.add(obj["index"])
            except Exception:
                # ignore bad lines
                pass
    return indices


def extract_answer(text: str) -> str:
    """
    “”

    
    1) 
    2) 
    """
    if not text:
        return ""
    text = text.strip()

    m = re.search(r"-?\d+", text)
    if m:
        return m.group(0)

    first_line = text.splitlines()[0].strip()
    parts = first_line.split()
    if parts:
        return parts[0]
    return text


def build_inputs(
    tokenizer,
    prompts: List[str],
    use_chat_template: bool,
    device: torch.device,
):
    if use_chat_template and getattr(tokenizer, "chat_template", None):

        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
    else:
        texts = prompts

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True,
    )
    
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    stop_words: List[str],
    use_chat_template: bool,
):

    device = next(model.parameters()).device

    inputs = build_inputs(tokenizer, prompts, use_chat_template, device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    #do_sample = temperature is not None and temperature > 0
    do_sample = 0
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_k=top_k if do_sample else None,
        top_p=top_p if do_sample else None,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
    )


    gen_texts = []
    prompt_len = input_ids.shape[1]

    for i in range(outputs.size(0)):
        #prompt_len = int(attention_mask[i].sum().item()) if attention_mask is not None else input_ids.size(1)
        gen_ids = outputs[i, prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        #prompt_len = input_ids.shape[1]
        #gen_ids = outputs[i, prompt_len:]

        if stop_words:
            for s in stop_words:
                if s:
                    text = text.split(s)[0]

        gen_texts.append(text)

    return gen_texts


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--subset", type=str, default="validation")  # validation/test
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--chunk_amount", type=int, default=1)

    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True)

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--stop_words", type=str, default="")  # comma separated
    parser.add_argument("--batch_size", type=int, default=1)

    # Prompt format
    parser.add_argument("--no_chat_template", action="store_true", default=False)

    parser.add_argument("--name", type=str, default="ann")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--local_k", type=int, default=32)
    parser.add_argument("--reallocate_to_mean_value", type=bool, default=True)
    parser.add_argument("--score", type=str, default="sparse_q")
    parser.add_argument("--rank", type=int, default=64)

    args = parser.parse_args()
    stop_words = list(filter(None, args.stop_words.split(",")))
    use_chat_template = not args.no_chat_template

    task_file = args.data_dir / args.task / f"{args.subset}.jsonl"
    args.save_dir.mkdir(parents=True, exist_ok=True)

    pred_file = (
        args.save_dir / f"{args.task}-{args.chunk_idx}.jsonl"
        if args.chunk_amount > 1
        else args.save_dir / f"{args.task}.jsonl"
    )

    print(f"Predict {args.task}\nfrom {task_file}\nto {pred_file}")

    data_all = read_jsonl(task_file)

    # chunk split
    if args.chunk_amount > 1:
        n = len(data_all)
        chunk_size = (n + args.chunk_amount - 1) // args.chunk_amount
        start = args.chunk_idx * chunk_size
        end = min(start + chunk_size, n)
        data_all = data_all[start:end]

    # skip already predicted indices
    done_indices = read_existing_pred_indices(pred_file)
    data = [x for x in data_all if x.get("index") not in done_indices]

    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto",
        attn_implementation="flash_attention_2",

    )

    print("attn impl =", getattr(model.config, "_attn_implementation", None))
    print("attn class =", type(model.model.layers[0].self_attn))

    
    sparq = Sparsity(
                name=args.name,
                k=args.k,
                local_k=args.local_k,
                rank=args.rank,
                score=args.score,
                reallocate_to_mean_value=args.reallocate_to_mean_value,
            )
    model = SparsityMethods.apply(sparq,model)

    print("attn impl =", getattr(model.config, "_attn_implementation", None))
    print("attn class =", type(model.model.layers[0].self_attn))

    model.eval()
    t0 = time.time()

    with open(pred_file, "a", encoding="utf-8", buffering=1) as fout:
        batch = []
        for dp in tqdm(data, total=len(data)):
            batch.append(dp)
            if len(batch) < args.batch_size:
                continue

            prompts = [x["input"] for x in batch]
            gen_texts = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop_words=stop_words,
                use_chat_template=use_chat_template,
            )

            for x, raw in zip(batch, gen_texts):
                #pred = extract_answer(raw)
                pred = raw
                out = {
                    "index": x["index"],
                    "pred": pred,
                    "input": x["input"],
                    "outputs": x["outputs"],
                }

                if "others" in x:
                    out["others"] = x["others"]
                if "truncation" in x:
                    out["truncation"] = x["truncation"]
                if "length" in x:
                    out["length"] = x["length"]

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")

            batch = []

        # flush last batch
        if batch:
            prompts = [x["input"] for x in batch]
            gen_texts = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop_words=stop_words,
                use_chat_template=use_chat_template,
            )
            for x, raw in zip(batch, gen_texts):
                pred = raw
                #pred = extract_answer(raw)
                out = {
                    "index": x["index"],
                    "pred": pred,
                    "input": x["input"],
                    "outputs": x["outputs"],
                }
                if "others" in x:
                    out["others"] = x["others"]
                if "truncation" in x:
                    out["truncation"] = x["truncation"]
                if "length" in x:
                    out["length"] = x["length"]

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Used time: {round((time.time() - t0) / 60, 2)} minutes")


if __name__ == "__main__":
    main()
