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
from tqdm import tqdm

from benchmarks.common import METHODS, MODELS, build_chat_prompt, method_defaults, parse_set, output_path, run_name
from loaders import load_model_and_tokenizer


DATA_FILE = Path(__file__).resolve().parent / "data" / "gsm8k_test.jsonl"
EXAMPLES = [
    ("question: There are 15 trees in the grove. Grove workers will plant trees in thegrove today. After they are done, there will be 21 trees. How many trees didthe grove workers plant today?", "target: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 15 = 6. The answer is 6."),
    ("question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?", "target: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?", "target: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12lollipops. How many lollipops did Jason give to Denny?", "target: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."),
    ("question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?", "target: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."),
    ("question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?", "target: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."),
    ("question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?", "target: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."),
    ("question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?", "target: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. The answer is 8."),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified GSM8K predictor")
    parser.add_argument("--method", choices=METHODS, default="pqcache")
    parser.add_argument("--model", choices=MODELS, default="llama-3.1-8b")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--dataset", nargs="+", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default=str(Path(__file__).parent / "outputs"))
    parser.add_argument("--max-new-tokens", type=int, default=3000)
    parser.add_argument("--runtest", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
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


def build_prompt(question: str) -> str:
    return "".join(f"{prompt}\n{answer}" for prompt, answer in EXAMPLES) + f"\nQuestion: {question}\n"


def extract_final_answer(answer: str) -> str | None:
    match = re.search(r"#### (\d{1,3}(?:,\d{3})*(?:\.?\d+)?)", answer)
    return match.group(1) if match else None


def load_data() -> list[dict]:
    with DATA_FILE.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for index, row in enumerate(rows):
        row.setdefault("index", index)
    return rows


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
    if args.limit is not None:
        data = data[:args.limit]
    destination = output_path(args.output_root, args.model, args.method, "gsm8k", run_name(args.method, args.budget, parse_set(args.set, args.method)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and destination.exists():
        destination.unlink()
    completed = {json.loads(line)["index"] for line in destination.open(encoding="utf-8")} if destination.exists() else set()
    args.dataset = "gsm8k"
    model, tokenizer = load_model_and_tokenizer(args)
    try:
        with destination.open("a", encoding="utf-8") as handle:
            for item in tqdm(data, desc="gsm8k"):
                if item["index"] in completed:
                    continue
                inputs = tokenizer(build_chat_prompt(tokenizer, build_prompt(item["question"])), truncation=False, return_tensors="pt").to(model.device)
                context_length = inputs.input_ids.shape[-1]
                prepare_sample(args, model, context_length)
                try:
                    with torch.no_grad():
                        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, num_beams=1, do_sample=False, temperature=1.0)[0]
                finally:
                    finish_sample(args, model)
                json.dump(
                    {
                        "index": item["index"],
                        "match_result": "null",
                        "final_answer": extract_final_answer(item["answer"]),
                        "pred": tokenizer.decode(output[context_length:], skip_special_tokens=True).strip(),
                    },
                    handle,
                    ensure_ascii=False,
                )
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
