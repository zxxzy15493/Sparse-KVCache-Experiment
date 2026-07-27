from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch

from benchmarks.common import METHODS, MODELS, method_defaults
from loaders import load_model_and_tokenizer


CHAT_DATASETS = {
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa",
    "2wikimqa", "musique", "dureader", "gov_report", "qmsum", "multi_news",
    "vcsum", "passage_count", "passage_retrieval_en", "passage_retrieval_zh",
}


def build_chat(tokenizer, prompt: str, dataset: str) -> str:
    if dataset not in CHAT_DATASETS or not getattr(tokenizer, "chat_template", None):
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-input inference")
    parser.add_argument("--input", default=None)
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=METHODS, default="pqcache")
    parser.add_argument("--model", choices=MODELS, default="llama-3.1-8b")
    parser.add_argument("--dataset", default=None, help="Optional LongBench dataset for its chat prompt rule")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=3000)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    if args.budget is None:
        args.budget = method_defaults(args.method)["budget"]
    if bool(args.input) == bool(args.input_file):
        parser.error("provide exactly one of --input or --input-file")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    prompt = args.input or Path(args.input_file).read_text(encoding="utf-8")
    model, tokenizer = load_model_and_tokenizer(args)
    try:
        inputs = tokenizer(build_chat(tokenizer, prompt, args.dataset or ""), truncation=False, return_tensors="pt").to(model.device)
        if args.method == "clusterkv":
            cluster_args = args._method_args
            cluster_args.nlist = max(args._base_nlist, math.ceil(inputs.input_ids.shape[-1] / 80))
            for module in model.modules():
                if hasattr(module, "flash_forward"):
                    args._cluster_attention.apply_cluster_config(module, cluster_args)
            args._cluster_attention.cluster_reset(model)
        context_length = inputs.input_ids.shape[-1]
        try:
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)[0]
        finally:
            if args.method == "clusterkv":
                args._cluster_attention.cluster_reset(model)
            torch.cuda.empty_cache()
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"input": prompt, "output": tokenizer.decode(output[context_length:], skip_special_tokens=True), "method": args.method, "model": args.model, "dataset": args.dataset}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(destination)
    finally:
        if getattr(args, "_cleanup", None):
            args._cleanup()
        del model, tokenizer
        torch.cuda.empty_cache()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    
if __name__ == "__main__":
    seed_everything(42)
    main()
