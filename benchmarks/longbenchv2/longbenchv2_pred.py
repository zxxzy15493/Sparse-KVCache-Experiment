from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from benchmarks.common import (
    METHODS, MODELS, ROOT, build_chat_prompt,
    method_defaults, middle_truncate, model_context_length, parse_set, output_path, run_name,
)
from loaders import load_model_and_tokenizer


PROMPTS = ROOT / "methods" / "ClusterKV" / "LongBenchV2" / "prompts"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified LongBenchV2 predictor")
    parser.add_argument("--method", choices=METHODS, default="pqcache")
    parser.add_argument("--model", choices=MODELS, default="llama-3.1-8b")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--dataset", nargs="+", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default=str(Path(__file__).parent / "outputs"))
    parser.add_argument("--max-context-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--cot-max-new-tokens", type=int, default=1024)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--rag", type=int, default=0)
    parser.add_argument("--runtest", action="store_true")
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
            module.token_budget, module.cluster_cache = cluster_args.token_budget, None
    args._cluster_attention.cluster_reset(model)


def finish_sample(args: argparse.Namespace, model) -> None:
    if args.method == "clusterkv":
        args._cluster_attention.cluster_reset(model)
    torch.cuda.empty_cache()


def read_templates() -> dict[str, str]:
    return {name: (PROMPTS / f"{name}.txt").read_text(encoding="utf-8") for name in ("0shot", "0shot_cot", "0shot_cot_ans", "0shot_no_context", "0shot_rag")}


def load_data() -> list[dict]:
    data_path = Path(__file__).parent / "filtered_longbench_v2_64k-192k.jsonl"
    return [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_answer(response: str) -> str | None:
    match = re.search(r"The correct answer is \(?([A-D])\)?", response.replace("*", ""))
    return match.group(1) if match else None


def fill_template(template: str, item: dict, context: str, cot: str = "") -> str:
    return (template.replace("$DOC$", context.strip()).replace("$Q$", item["question"].strip())
            .replace("$C_A$", item["choice_A"].strip()).replace("$C_B$", item["choice_B"].strip())
            .replace("$C_C$", item["choice_C"].strip()).replace("$C_D$", item["choice_D"].strip()).replace("$COT$", cot))


def base_prompt(templates: dict[str, str], item: dict, args: argparse.Namespace) -> str:
    context = item["context"]
    if args.rag:
        context = "\n\n".join(f"Retrieved chunk {index + 1}: {row['content']}" for index, row in enumerate(sorted(item["retrieved_context"][:args.rag], key=lambda row: row["c_idx"])))
        return fill_template(templates["0shot_rag"], item, context)
    return fill_template(templates["0shot_no_context" if args.no_context else "0shot_cot" if args.cot else "0shot"], item, context)


def generate(model, tokenizer, args: argparse.Namespace, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(build_chat_prompt(tokenizer, prompt), truncation=False, return_tensors="pt").to(model.device)
    context_length = inputs.input_ids.shape[-1]
    prepare_sample(args, model, context_length)
    try:
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, temperature=1.0)[0]
    finally:
        finish_sample(args, model)
    return tokenizer.decode(output[context_length:], skip_special_tokens=True).strip()


def record(item: dict, response: str, response_cot: str | None = None) -> dict:
    output = {"response": response, "pred": extract_answer(response), "answer": item["answer"], "_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "question": item["question"], "context": item["context"][:1000]}
    output["judge"] = output["pred"] == output["answer"]
    if response_cot is not None:
        output["response_cot"] = response_cot
    return output


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def run(args: argparse.Namespace) -> Path:
    data = load_data()
    if args.runtest:
        data = data[:2]
    destination = output_path(args.output_root, args.model, args.method, "longbenchv2", run_name(args.method, args.budget, parse_set(args.set, args.method)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and destination.exists():
        destination.unlink()
    completed = {json.loads(line)["_id"] for line in destination.open(encoding="utf-8") if line.strip()} if destination.exists() else set()

    args.dataset = 'longbenchv2'     
    model, tokenizer = load_model_and_tokenizer(args)
    templates, max_context = read_templates(), args.max_context_length or model_context_length(args.model)
    try:
        with destination.open("a", encoding="utf-8") as handle:
            for item in tqdm(data, desc="longbenchv2"):
                if item["_id"] in completed:
                    continue
                prompt = middle_truncate(tokenizer, base_prompt(templates, item, args), max_context)
                response_cot = None
                if args.cot:
                    response_cot = generate(model, tokenizer, args, prompt, args.cot_max_new_tokens)
                    prompt = middle_truncate(tokenizer, fill_template(templates["0shot_cot_ans"], item, item["context"], response_cot), max_context)
                json.dump(record(item, generate(model, tokenizer, args, prompt, args.max_new_tokens), response_cot), handle, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
    finally:
        if getattr(args, "_cleanup", None):
            args._cleanup()
        del model, tokenizer
        torch.cuda.empty_cache()
    return destination


def main(argv: list[str] | None = None) -> None:
    print(run(parse_args(argv)))


if __name__ == "__main__":
    seed_everything(42)
    main()
