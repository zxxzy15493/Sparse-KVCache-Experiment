from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common import ROOT, method_defaults, parse_set, run_name


def load_evaluator():
    path = ROOT / "methods" / "ClusterKV" / "accuracy" / "LongBench" / "eval.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("clusterkv_longbench_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate unified LongBench prediction files")
    parser.add_argument("--output-root", default=str(Path(__file__).parent / "outputs"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--n2eos", action="store_true")
    args = parser.parse_args(argv)
    if args.budget is None:
        if args.method not in {"topp", "topp32","flexprefill","xattention"}:
            parser.error("--budget is required except for topp, topp32, xattention, and flexprefill")
        args.budget = method_defaults(args.method)["budget"]
    return args


def evaluate(args: argparse.Namespace) -> Path:
    evaluator = load_evaluator()
    root = Path(args.output_root) / args.model / args.experiment / args.method
    if not root.is_dir():
        raise FileNotFoundError(root)
    scores = {}
    prediction_name = run_name(args.method, args.budget, parse_set(args.set, args.method))
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        dataset = dataset_dir.name
        if args.datasets and dataset not in args.datasets:
            continue
        predictions, answers, lengths, all_classes = [], [], [], []
        path = dataset_dir / f"{prediction_name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            prediction = record["pred"]
            if args.n2eos and evaluator.dataset2metric[dataset] != evaluator.rouge_score:
                prediction = prediction.split("\n\n")[0]
            predictions.append(prediction)
            answers.append(record["answers"])
            lengths.append(record.get("length", 0))
            all_classes.append(record.get("all_classes", []))
        if not predictions:
            continue
        if args.extended:
            scores[dataset] = evaluator.scorer_e(dataset, predictions, answers, lengths, all_classes[-1])
        else:
            scores[dataset] = evaluator.scorer(dataset, predictions, answers, all_classes[-1])
    output = root / (f"result-{prediction_name}-n2eos.json" if args.n2eos else f"result-{prediction_name}.json")
    output.write_text(json.dumps(scores, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> None:
    print(evaluate(parse_args(argv)))


if __name__ == "__main__":
    main()
