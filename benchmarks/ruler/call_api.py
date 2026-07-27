from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml
from tqdm import tqdm

from benchmarks.common import METHODS, MODELS, method_defaults
from model_wrappers import RulerModel


RULER_ROOT = Path(__file__).resolve().parent


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def task_config(task: str) -> dict:
    module = importlib.import_module("data.synthetic.constants")
    customized = yaml.safe_load((RULER_ROOT / "synthetic.yaml").read_text(encoding="utf-8"))
    if task not in customized:
        raise ValueError(f"Unknown RULER task: {task}")
    config = dict(customized[task])
    config.update(module.TASKS[config["task"]])
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local unified models on RULER data")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--save-dir",
        type=Path,
        required=True,
        help="Prediction directory: benchmark_root_pred/synthetic/<model>/<method>/<parameter-signature>/<length>",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", choices=METHODS, default="pqcache")
    parser.add_argument("--model", choices=MODELS, default="llama-3.1-8b")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--dataset", nargs="+", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    if args.budget is None:
        args.budget = method_defaults(args.method)["budget"]
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def run(args: argparse.Namespace) -> Path:
    config = task_config(args.task)
    task_file = args.data_dir / args.task / "validation.jsonl"
    pred_file = args.save_dir / f"{args.task}.jsonl"
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and pred_file.exists():
        pred_file.unlink()

    data = read_manifest(task_file)
    if args.limit is not None:
        data = data[:args.limit]
    completed = {row["index"] for row in read_manifest(pred_file)} if pred_file.exists() else set()
    pending = [row for row in data if row["index"] not in completed]
    if not pending:
        print(f"All samples already processed: {pred_file}")
        return pred_file
    print("Call_api.py =================================== ================================")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    args.dataset = "ruler"
    
    wrapper = RulerModel(args.method, args.model, args.budget, args.device, args.set, args.max_seq_length)
    max_new_tokens = args.max_new_tokens or config["tokens_to_generate"]
    try:
        with pred_file.open("a", encoding="utf-8", buffering=1) as handle:
            for sample in tqdm(pending, desc=args.task):
                record = {
                    "index": sample["index"],
                    "pred": wrapper.generate(sample["input"], max_new_tokens),
                    "outputs": sample.get("outputs", []),
                    "others": sample.get("others", {}),
                    "truncation": sample.get("truncation"),
                    "length": sample.get("length"),
                }
                json.dump(record, handle, ensure_ascii=False)
                handle.write("\n")
    finally:
        wrapper.close()
    print(pred_file)
    return pred_file


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    seed_everything(42)
    main()
