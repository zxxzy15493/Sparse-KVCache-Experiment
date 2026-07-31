from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from benchmarks.common import (
    METHODS, MODELS,
    model_context_length,
    model_family,
    method_defaults,
    parse_set,
    run_name,
)
from loaders import load_model_and_tokenizer


LONG_BENCH_CONFIG = Path(__file__).parent / "config"
DEFAULT_DATASETS = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa",
    "2wikimqa", "musique", "dureader", "gov_report", "qmsum", "multi_news",
    "vcsum", "trec", "triviaqa", "samsum", "lsht", "passage_count",
    "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p",
]
CHAT_DATASETS = set(DEFAULT_DATASETS) - {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified LongBench predictor")
    parser.add_argument("--method", choices=METHODS, default="pqcache")
    parser.add_argument("--model", choices=MODELS, default="llama-3.1-8b")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--dataset", nargs="+", default=None)
    parser.add_argument("--experiment", default="default")
    parser.add_argument("--output-root", default=str(Path(__file__).parent / "outputs"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--max-context-length", type=int, default=None)
    parser.add_argument("--runtest", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    if args.budget is None:
        args.budget = method_defaults(args.method)["budget"]
    return args


def prepare_sample(args: argparse.Namespace, model, context_length: int) -> None:
    if args.method != "clusterkv":
        return
    cluster_args = args._method_args
    cluster_args.nlist = max(args._base_nlist, math.ceil(context_length / 80))
    for module in model.modules():
        if hasattr(module, "flash_forward"):
            args._cluster_attention.apply_cluster_config(module, cluster_args)
            module.token_budget = cluster_args.token_budget
            module.cluster_cache = None
    args._cluster_attention.cluster_reset(model)


def finish_sample(args: argparse.Namespace, model) -> None:
    if args.method == "clusterkv":
        args._cluster_attention.cluster_reset(model)
    torch.cuda.empty_cache()


def load_configs() -> tuple[dict[str, str], dict[str, int]]:
    prompts = json.load(open(LONG_BENCH_CONFIG / "dataset2prompt.json", "r"))
    limits = json.load(open(LONG_BENCH_CONFIG / "dataset2maxlen.json", "r"))
    return prompts, limits


def load_data(dataset: str, extended: bool):
    subset = f"{dataset}_e" if extended else dataset
    return load_dataset("THUDM/LongBench", subset, split="test")


def truncate_prompt(tokenizer, prompt: str, max_context_length: int, family: str) -> str:
    tokens = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=family != "glm4").input_ids[0]
    if len(tokens) <= max_context_length:
        return prompt
    first = max_context_length // 2
    return tokenizer.decode(tokens[:first], skip_special_tokens=True) + tokenizer.decode(
        tokens[-(max_context_length - first):], skip_special_tokens=True
    )


def build_chat(tokenizer, prompt: str, dataset: str) -> str:
    if dataset not in CHAT_DATASETS or not getattr(tokenizer, "chat_template", None):
        return prompt
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)


def post_process(prediction: str, model_name: str) -> str:
    if model_name.startswith("glm-"):
        return prediction.split("Assistant:")[-1]
    if "qwen" in model_name:
        return prediction.split("<|im_end|>")[0].split("<|im_start|>")[-1].strip()
    return prediction


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def run(args: argparse.Namespace) -> list[Path]:
    prompts, max_generations = load_configs()
    datasets = args.datasets or DEFAULT_DATASETS
    unknown = sorted(set(datasets) - set(prompts))
    if unknown:
        raise ValueError(f"Unsupported LongBench datasets: {', '.join(unknown)}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    
    model, tokenizer = load_model_and_tokenizer(args)
    
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print(f"Loaded model: {args.model} on device {args.device}")
    print(f"Using method: {args.method} with budget: {args.budget}")
    print(f"Datasets to evaluate: {', '.join(datasets)}")

    max_context = args.max_context_length or model_context_length(args.model)
    file_signature = run_name(args.method, args.budget, parse_set(args.set, args.method))
    paths: list[Path] = []
    try:
        for dataset in datasets:
            if args.method == "magicpig":
                args.dataset = dataset
                model, tokenizer = load_model_and_tokenizer(args)
            print(f"\n===== LongBench dataset: {dataset} =====", flush=True)
            data = load_data(dataset, args.extended)
            if args.runtest:
                data = data.select(range(min(2, len(data))))
            if args.limit is not None:
                data = data.select(range(min(args.limit, len(data))))
            path = Path(args.output_root) / args.model / args.experiment / args.method / dataset / f"{file_signature}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            if args.overwrite and path.exists():
                path.unlink()
            completed = sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0
            if completed >= len(data):
                paths.append(path)
                continue
            with path.open("a", encoding="utf-8") as handle:
                for item in tqdm(data.select(range(completed, len(data))), desc=dataset):
                    prompt = truncate_prompt(tokenizer, prompts[dataset].format(**item), max_context, model_family(args.model))
                    inputs = tokenizer(build_chat(tokenizer, prompt, dataset), truncation=False, return_tensors="pt").to(model.device)
                    generation_kwargs = {}
                    if dataset == "samsum" or dataset == "trec":
                        generation_kwargs = {
                            "min_length": inputs.input_ids.shape[-1] + 1,
                            "eos_token_id": [tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                        }
                    context_length = inputs.input_ids.shape[-1]
                    prepare_sample(args, model, context_length)
                    try:
                        with torch.no_grad():
                            output = model.generate(
                                **inputs, max_new_tokens=max_generations[dataset], num_beams=1,
                                do_sample=False, temperature=1.0, **generation_kwargs,
                            )[0]
                    finally:
                        finish_sample(args, model)
                    json.dump({
                        "pred": post_process(tokenizer.decode(output[context_length:], skip_special_tokens=True), args.model),
                        "answers": item["answers"], "all_classes": item.get("all_classes", []), "length": item["length"],
                    }, handle, ensure_ascii=False)
                    handle.write("\n")
                    handle.flush()
            paths.append(path)
            if args.method == "magicpig":
                del model, tokenizer
    finally:
        if getattr(args, "_cleanup", None):
            args._cleanup()
        del model, tokenizer
        torch.cuda.empty_cache()
    return paths


def main(argv: list[str] | None = None) -> None:
    for path in run(parse_args(argv)):
        print(path)


if __name__ == "__main__":
    seed_everything(42)
    main()
