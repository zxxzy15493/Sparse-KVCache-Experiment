"""Evaluate LongBench predictions with LonBench/metrics.py and scoring rules."""

import argparse
import json
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
LONGBENCH_DIR = EXPERIMENT_ROOT.parent / "LonBench"
if str(LONGBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(LONGBENCH_DIR))

from metrics import (
    qa_f1_score, qa_f1_zh_score, rouge_score, rouge_zh_score,
    classification_score, retrieval_score, retrieval_zh_score, count_score, code_sim_score,
)
from utils import load_data

dataset2metric = {
    "narrativeqa": qa_f1_score, "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score, "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score, "2wikimqa": qa_f1_score, "musique": qa_f1_score,
    "dureader": rouge_zh_score, "gov_report": rouge_score, "qmsum": rouge_score,
    "multi_news": rouge_score, "vcsum": rouge_zh_score,
    "trec": classification_score, "triviaqa": qa_f1_score, "samsum": rouge_score,
    "lsht": classification_score, "passage_retrieval_en": retrieval_score,
    "passage_count": count_score, "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score, "repobench-p": code_sim_score,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", default="LongBench", choices=["LongBench"])
    parser.add_argument("--task", required=True, choices=sorted(dataset2metric))
    parser.add_argument("--save_dir", required=True, type=Path)
    args = parser.parse_args(argv)
    records = load_data(args.save_dir / f"{args.task}.jsonl")
    if not records:
        raise ValueError(f"Prediction file is empty: {args.task}.jsonl")
    total_score = 0.0
    for record in records:
        prediction = record["pred"]
        if args.task in {"trec", "triviaqa", "samsum", "lsht"}:
            prediction = prediction.lstrip("\n").split("\n")[0]
        total_score += max((dataset2metric[args.task](
            prediction, answer, all_classes=record.get("all_classes"),
        ) for answer in record["answers"]), default=0.0)
    score = round(100 * total_score / len(records), 2)
    result_path = args.save_dir / "result.json"
    scores = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    scores[args.task] = score
    result_path.write_text(json.dumps(scores, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    print(f"[{args.task}] score={score}, samples={len(records)}; results: {result_path}")


if __name__ == "__main__":
    main()
