import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Peak memory test with full attention")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-family", type=str, choices=["llama", "qwen"], required=True)

    parser.add_argument("--input-file", type=str, default="./myinput.txt")
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)

    parser.add_argument("--csv", type=str, default="./log_mem_full/mem_full_results.csv")
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()


def build_input(tokenizer, input_file, input_len, device):
    with open(input_file, "r", encoding="utf-8") as f:
        input_string = f.read()

    encoded = tokenizer(
        input_string,
        truncation=False,
        return_tensors="pt",
    )

    input_ids = encoded.input_ids

    # repeat tokens if input file is too short, to ensure actual input_len
    if input_ids.shape[1] < input_len:
        repeat_times = (input_len + input_ids.shape[1] - 1) // input_ids.shape[1]
        input_ids = input_ids.repeat(1, repeat_times)

    input_ids = input_ids[:, :input_len].to(device)
    return input_ids


def append_csv(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "time",
        "model_family",
        "model",
        "input_len",
        "max_new_tokens",
        "peak_alloc_mib",
        "peak_alloc_bytes",
        "elapsed_sec",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()

    print("=" * 80, flush=True)
    print(f"Model        : {args.model}", flush=True)
    print(f"Model family : {args.model_family}", flush=True)
    print(f"Input len    : {args.input_len}", flush=True)
    print(f"Output len   : {args.max_new_tokens}", flush=True)
    print("=" * 80, flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=True,
    )
    print("Tokenizer loaded.", flush=True)

    print("Loading model weights on CPU...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
        attn_implementation="flash_attention_2",
    )
    print(f"Model weights loaded. Moving model to {args.device}...", flush=True)
    model = model.to(args.device)
    print(f"Model moved to {args.device}.", flush=True)

    model.eval()
    print("attn_impl =", model.config._attn_implementation, flush=True)
    print("attn_cls  =", model.model.layers[0].self_attn.__class__.__name__, flush=True)
    input_ids = build_input(
        tokenizer=tokenizer,
        input_file=args.input_file,
        input_len=args.input_len,
        device=args.device,
    )

    print(f"Actual input_ids shape: {tuple(input_ids.shape)}", flush=True)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # only track peak memory during generate phase
    torch.cuda.memory.reset_peak_memory_stats()

    try:
        torch.cuda.memory._record_memory_history()
    except TypeError:
        torch.cuda.memory._record_memory_history(enabled=True)

    torch.cuda.synchronize()
    begin = time.perf_counter()

    with torch.no_grad():
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=None,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.max_new_tokens,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
        )

    torch.cuda.synchronize()
    end = time.perf_counter()

    peak_alloc = torch.cuda.memory.max_memory_allocated()
    elapsed = end - begin

    print(f"elapsed       = {elapsed:.4f} s", flush=True)
    print(f"peak_alloc    = {peak_alloc / 1024**2:.2f} MiB", flush=True)

    append_csv(
        args.csv,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_family": args.model_family,
            "model": args.model,
            "input_len": args.input_len,
            "max_new_tokens": args.max_new_tokens,
            "peak_alloc_mib": f"{peak_alloc / 1024**2:.2f}",
            "peak_alloc_bytes": peak_alloc,
            "elapsed_sec": f"{elapsed:.4f}",
        },
    )

    try:
        torch.cuda.memory._record_memory_history(enabled=None)
    except Exception:
        pass

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
