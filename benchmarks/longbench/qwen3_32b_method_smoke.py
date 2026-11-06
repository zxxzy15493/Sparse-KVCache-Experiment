#!/usr/bin/env python3
"""Load one Qwen3-32B LongBench method and generate a short answer.

This complements ``qwen3_32b_all_methods.sh``: it validates a method patch
without downloading or executing a full LongBench dataset split.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.longbench.longbench_pred import parse_args
from loaders import load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--prompt-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.prompt_tokens <= 0:
        raise ValueError("--prompt-tokens must be positive")

    longbench_args = parse_args(
        [
            "--method", args.method,
            "--model", "Qwen/Qwen3-32B",
            "--budget", str(args.budget),
            "--datasets", "qasper",
            *sum((["--set", item] for item in args.set), []),
        ]
    )
    # MagicPig reads this during loader construction. Other methods ignore it.
    longbench_args.dataset = "qasper"
    started = time.monotonic()
    model = tokenizer = None
    try:
        model, tokenizer = load_model_and_tokenizer(longbench_args)
        prompt = "Answer briefly and factually: " + "The quick brown fox jumps over the lazy dog. " * args.prompt_tokens
        input_device = getattr(model, "device", None)
        if input_device is None:
            input_device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=True)
        assert output.shape[-1] == inputs.input_ids.shape[-1] + 1
        print(json.dumps({
            "status": "pass", "method": args.method, "budget": args.budget,
            "set": args.set, "input_tokens": int(inputs.input_ids.shape[-1]),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }))
    finally:
        cleanup = getattr(longbench_args, "_cleanup", None)
        if callable(cleanup):
            cleanup()
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
