# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import os
import sys
import re
import json
import argparse

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CODE_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from minference import MInference
from utils import load_data

_CONFIG_DIR = os.path.join(_CODE_DIR, "config")
_PROMPTS_DIR = os.path.join(_CODE_DIR, "prompts")

model2path = json.load(open(os.path.join(_CONFIG_DIR, "model2path.json")))
model2maxlen = json.load(open(os.path.join(_CONFIG_DIR, "model2maxlen.json")))

# Default dataset path
_DEFAULT_DATA = "../../../../benchmarks/longbenchv2/filtered_longbench_v2_64k-192k.jsonl"


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen2.5-7B-Instruct-1M",
                        choices=list(model2path.keys()),
                        help="Model short name")
    parser.add_argument("--data_path", type=str, default=_DEFAULT_DATA,
                        help="Path to LongBenchV2 data.jsonl")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Output directory (default: results/<model>)")
    parser.add_argument("--cot", action="store_true", default=True,
                        help="Use chain-of-thought inference")
    parser.add_argument("--no_context", action="store_true", default=False,
                        help="Skip document context (measure memorization)")
    parser.add_argument("--max_len", type=int, default=None,
                        help="Max input tokens (default from model2maxlen)")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["fp16", "bf16"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cot_max_new_tokens", type=int, default=1024,
                        help="Max new tokens for CoT response")
    parser.add_argument("--ans_max_new_tokens", type=int, default=128,
                        help="Max new tokens for final answer")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Number of samples to evaluate (-1 for all)")
    return parser.parse_args(args)


def load_model_and_tokenizer(model_name, dtype, device):
    # Monkey-patch GLM config_class mismatch (same as AccuracyOverview)
    from transformers.models.auto.auto_factory import _BaseAutoModelClass
    _orig_register = _BaseAutoModelClass.register.__func__

    def _patched_register(cls, config_class, model_class, exist_ok=False):
        if (
            hasattr(model_class, "config_class")
            and str(model_class.config_class) != str(config_class)
        ):
            model_class.config_class = config_class
        return _orig_register(cls, config_class, model_class, exist_ok=exist_ok)

    _BaseAutoModelClass.register = classmethod(_patched_register)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    finally:
        _BaseAutoModelClass.register = classmethod(_orig_register)

    model.eval()

    # Apply MInference sparse attention patch
    minference_patch = MInference(model_name)
    model = minference_patch(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_prompt(item, cot=False, no_context=False):
    """Build the prompt for a data item using the appropriate template."""
    if no_context:
        template_file = "0shot_no_context.txt"
    else:
        template_file = "0shot_cot.txt" if cot else "0shot.txt"

    with open(os.path.join(_PROMPTS_DIR, template_file), "r") as f:
        template = f.read()

    prompt = template.replace("$DOC$", item.get("context", ""))
    prompt = prompt.replace("$Q$", item.get("question", ""))
    prompt = prompt.replace("$C_A$", item.get("choice_A", ""))
    prompt = prompt.replace("$C_B$", item.get("choice_B", ""))
    prompt = prompt.replace("$C_C$", item.get("choice_C", ""))
    prompt = prompt.replace("$C_D$", item.get("choice_D", ""))
    return prompt


def build_cot_ans_prompt(item, cot_response):
    """Build the answer extraction prompt after CoT."""
    with open(os.path.join(_PROMPTS_DIR, "0shot_cot_ans.txt"), "r") as f:
        template = f.read()

    prompt = template.replace("$Q$", item.get("question", ""))
    prompt = prompt.replace("$COT$", cot_response)
    prompt = prompt.replace("$C_A$", item.get("choice_A", ""))
    prompt = prompt.replace("$C_B$", item.get("choice_B", ""))
    prompt = prompt.replace("$C_C$", item.get("choice_C", ""))
    prompt = prompt.replace("$C_D$", item.get("choice_D", ""))
    return prompt


def extract_answer(response):
    """Extract the predicted answer letter from model response."""
    # Try with parentheses first
    m = re.search(r"correct answer is \(([A-D])\)", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Try without parentheses
    m = re.search(r"correct answer is ([A-D])", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: find any single A/B/C/D
    m = re.findall(r"\b([A-D])\b", response)
    if m:
        return m[-1].upper()
    return None


def truncate_input(input_ids, max_len):
    """Truncate to max_len by keeping first and last halves."""
    if input_ids.shape[1] <= max_len:
        return input_ids
    half = max_len // 2
    return torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)


def generate_response(model, tokenizer, prompt, max_new_tokens, max_len):
    """Generate text response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    # Truncate if needed
    input_ids = truncate_input(input_ids, max_len)
    attention_mask = truncate_input(attention_mask, max_len)

    input_len = input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=1.0,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = out[0, input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


def get_pred(model, tokenizer, item, args, max_len, out_path):
    """Run prediction on a single data item."""
    # Phase 1: Generate CoT response (or direct answer if not --cot)
    prompt = build_prompt(item, cot=args.cot, no_context=args.no_context)
    response_cot = generate_response(
        model, tokenizer, prompt,
        max_new_tokens=args.cot_max_new_tokens if args.cot else args.ans_max_new_tokens,
        max_len=max_len,
    )

    if args.cot:
        # Phase 2: Extract answer from CoT
        ans_prompt = build_cot_ans_prompt(item, response_cot)
        response = generate_response(
            model, tokenizer, ans_prompt,
            max_new_tokens=args.ans_max_new_tokens,
            max_len=max_len,
        )
    else:
        response = response_cot

    pred = extract_answer(response)

    # Build output record
    record = {
        "_id": item["_id"],
        "domain": item["domain"],
        "sub_domain": item["sub_domain"],
        "difficulty": item["difficulty"],
        "length": item["length"],
        "question": item["question"],
        "choice_A": item["choice_A"],
        "choice_B": item["choice_B"],
        "choice_C": item["choice_C"],
        "choice_D": item["choice_D"],
        "answer": item["answer"],
        "context": item["context"][:1000],
        "response": response,
        "pred": pred,
        "judge": pred == item["answer"] if pred is not None else False,
    }
    if args.cot:
        record["response_cot"] = response_cot

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    torch.cuda.empty_cache()


def main():
    args = parse_args()

    # Resolve model path
    model_path = model2path[args.model]
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    max_len = args.max_len or model2maxlen[args.model]

    # Default output path
    save_dir = args.save_dir or os.path.join(_CODE_DIR, "results", args.model)
    os.makedirs(save_dir, exist_ok=True)
    cot_suffix = "_cot" if args.cot else ""
    nc_suffix = "_no_context" if args.no_context else ""
    out_path = os.path.join(save_dir, f"pred{cot_suffix}{nc_suffix}.jsonl")

    # Load data
    print(f"Loading data from {args.data_path}")
    data = load_data(args.data_path)
    if args.num_samples > 0:
        data = data[: args.num_samples]

    # Resume support
    done_ids = set()
    if os.path.exists(out_path):
        for sample in load_data(out_path):
            done_ids.add(sample.get("_id", ""))
    remaining = [d for d in data if d["_id"] not in done_ids]
    print(f"Total: {len(data)}, Done: {len(done_ids)}, Remaining: {len(remaining)}")

    if len(remaining) == 0:
        print("All samples already predicted, exiting.")
        return

    # Load model
    print(f"Loading model: {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path, dtype, args.device)
    print(f"Model loaded. Device: {model.device}, Max input len: {max_len}")

    # Predict
    for item in tqdm(remaining, desc="LongBenchV2"):
        get_pred(model, tokenizer, item, args, max_len, out_path)

    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
