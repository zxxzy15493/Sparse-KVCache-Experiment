"""Self-contained RULER evaluation for BatchExperiment prediction JSONL files.

Uses Adakv's ``RULER/eval/synthetic/constants.py`` metric functions together with
``benchmarks/ruler/synthetic.yaml`` task definitions. The legacy
``RULER/eval/evaluate.py`` is intentionally not reused (it hardcodes a
``cakekv.experiments...`` import and would be broken here).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
# REPO_ROOT is Sparse-KVCache-Experiment/ (holds benchmarks/ruler).
REPO_ROOT = EXPERIMENT_ROOT.parents[3]
RULER_ROOT = REPO_ROOT / "benchmarks" / "ruler"
RULER_CONSTANTS = EXPERIMENT_ROOT.parent / "RULER" / "eval" / "synthetic" / "constants.py"


def load_constants() -> dict:
    spec = importlib.util.spec_from_file_location("adakv_ruler_constants", RULER_CONSTANTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TASKS


def load_task_configs() -> dict:
    import yaml

    customized = yaml.safe_load((RULER_ROOT / "synthetic.yaml").read_text(encoding="utf-8"))
    constants = load_constants()
    configs = {}
    for name, value in customized.items():
        config = dict(value)
        config.update(constants[config["task"]])
        configs[name] = config
    return configs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RULER prediction JSONL files")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing BatchExperiment task prediction JSONL files")
    parser.add_argument("--tasks", nargs="+", default=None)
    return parser.parse_args(argv)


def evaluate(args: argparse.Namespace) -> Path:
    configs = load_task_configs()
    tasks = args.tasks or sorted(path.stem for path in args.data_dir.glob("*.jsonl")
                                 if not path.name.endswith(".batch_stats.jsonl"))
    evaluated_tasks = []
    scores = []
    nulls = []
    for task in tasks:
        predictions = args.data_dir / f"{task}.jsonl"
        if not predictions.exists():
            raise FileNotFoundError(f"Prediction file not found: {predictions}")
        if task not in configs:
            raise ValueError(f"Unknown RULER task: {task}")
        records = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        predicts = [record.get("pred", "").strip() for record in records]
        answers = [record.get("outputs", [""]) for record in records]
        score = configs[task]["metric_fn"](predicts, answers) \
            if answers and answers[0] and answers[0][0] is not None else 0.0
        evaluated_tasks.append(task)
        scores.append(score)
        nulls.append(f"{sum(not value for value in predicts)}/{len(predicts)}")

    output = args.data_dir / "summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(range(len(evaluated_tasks) + 1))
        writer.writerow(["Tasks", *evaluated_tasks])
        writer.writerow(["Score", *scores])
        writer.writerow(["Nulls", *nulls])
    print(output)
    return output


def main(argv: list[str] | None = None) -> None:
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
