from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

# REPO_ROOT is the project root: Sparse-KVCache-Experiment/
REPO_ROOT = Path(__file__).resolve().parents[4]

# Point to the actual RULER benchmark directory for synthetic.yaml and data.synthetic.constants
RULER_ROOT = REPO_ROOT / "benchmarks" / "ruler"
sys.path.insert(0, str(RULER_ROOT))

import yaml


def string_match_part(predictions: list[str], references: list[list[str]]) -> float:
    score = sum(max(1.0 if reference.lower() in prediction.lower() else 0.0 for reference in answers)
                for prediction, answers in zip(predictions, references))
    return round(100 * score / len(predictions), 2) if predictions else 0.0


def string_match_all(predictions: list[str], references: list[list[str]]) -> float:
    score = sum(
        sum(1.0 if reference.lower() in prediction.lower() else 0.0 for reference in answers) / len(answers)
        for prediction, answers in zip(predictions, references)
    )
    return round(100 * score / len(predictions), 2) if predictions else 0.0


METRICS = {
    "niah": string_match_all,
    "variable_tracking": string_match_all,
    "common_words_extraction": string_match_all,
    "freq_words_extraction": string_match_all,
    "qa": string_match_part,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RULER prediction JSONL files")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Prediction directory: benchmark_root_pred/synthetic/<model>/<method>/<parameter-signature>/<length>",
    )
    parser.add_argument("--tasks", nargs="+", default=None)
    return parser.parse_args(argv)


def load_task_configs() -> dict:
    module = importlib.import_module("data.synthetic.constants")
    customized = yaml.safe_load((RULER_ROOT / "synthetic.yaml").read_text(encoding="utf-8"))
    configs = {}
    for name, value in customized.items():
        config = dict(value)
        config.update(module.TASKS[config["task"]])
        config["metric_fn"] = METRICS[config["task"]]
        configs[name] = config
    return configs


def evaluate(args: argparse.Namespace) -> Path:
    configs = load_task_configs()
    tasks = args.tasks or sorted(path.stem for path in args.data_dir.glob("*.jsonl"))
    evaluated_tasks = []
    scores = []
    nulls = []
    for task in tasks:
        predictions = args.data_dir / f"{task}.jsonl"
        if not predictions.exists():
            continue
        records = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
        predicts = [record.get("pred", "").strip() for record in records]
        answers = [record.get("outputs", [""]) for record in records]
        score = configs[task]["metric_fn"](predicts, answers) if answers and answers[0] and answers[0][0] is not None else 0.0
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
