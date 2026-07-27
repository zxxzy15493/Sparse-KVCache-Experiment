# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import os
import re
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)
from utils import load_data

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

# Synthetic task groups for Ruler
SYNTHETIC_TASK_GROUPS = {
    'niah': 128,
    'vt': 30,
    'cwe': 120,
    'fwe': 50,
    'qa': 32
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Model name or path (as used in result directory)")
    parser.add_argument('--benchmark', type=str, required=True, choices=["LongBench", "Synthetic"],
                        help="Benchmark type")
    parser.add_argument('--task', type=str, required=True, help="Task name")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory containing prediction jsonl files")
    parser.add_argument('--data_dir', type=str, default="../../../../benchmarks",
                        help="Directory containing the task data files (constructed by Accuracy.sh)")
    return parser.parse_args(args)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.strip().split('\n')[0]  # take first line
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    s = ' '.join(s.split())
    return s


def exact_match_score(prediction, ground_truths):
    """Exact match scoring for Synthetic tasks. Check if prediction contains any ground truth."""
    pred = prediction.strip().lower()
    for gt in ground_truths:
        gt = gt.strip().lower()
        if gt in pred:
            return 1.0
    return 0.0


def score_longbench(dataset, predictions, answers, all_classes):
    """Score LongBench predictions."""
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def score_synthetic(predictions, ground_truths_list):
    """Score Synthetic/Ruler predictions using exact match."""
    total_score = 0.
    for pred, gts in zip(predictions, ground_truths_list):
        total_score += exact_match_score(pred, gts)
    return round(100 * total_score / len(predictions), 2)


if __name__ == '__main__':
    args = parse_args()

    path = args.save_dir
    if not path.endswith('/'):
        path += '/'

    scores = dict()

    if args.benchmark == "LongBench":
        # LongBench evaluation - same as before
        all_files = os.listdir(path)
        print("Evaluating on:", all_files)
        for filename in all_files:
            if not filename.endswith("jsonl"):
                continue
            predictions, answers = [], []
            dataset = filename.replace('.jsonl', '')
            with open(f"{path}{filename}", "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    predictions.append(data["pred"])
                    answers.append(data["answers"])
                    all_classes = data.get("all_classes", None)
            score = score_longbench(dataset, predictions, answers, all_classes)
            scores[dataset] = score

    elif args.benchmark == "Synthetic":
        # Synthetic/Ruler evaluation
        task = args.task
        pred_file = f"{path}{task}.jsonl"
        if not os.path.exists(pred_file):
            print(f"Prediction file not found: {pred_file}")
            exit(1)

        # Load predictions
        pred_data = load_data(pred_file)
        pred_map = {sample["index"]: sample["pred"] for sample in pred_data}

        # Load original data to get ground truth
        task_file = f'{args.data_dir}/validation.jsonl'
        if not os.path.exists(task_file):
            raise FileNotFoundError(f"Original task file not found: {task_file}")

        orig_data = load_data(task_file)

        # Match predictions with ground truth
        predictions, ground_truths_list = [], []
        for sample in orig_data:
            idx = sample["index"]
            if idx in pred_map:
                predictions.append(pred_map[idx])
                ground_truths_list.append(sample["outputs"])

        if len(predictions) == 0:
            print(f"No matching predictions found for task {task}")
            exit(1)

        score = score_synthetic(predictions, ground_truths_list)

        # Get task group for display
        task_key = next((key for key in SYNTHETIC_TASK_GROUPS.keys() if key in task), task)
        scores[task] = score
        print(f"Task: {task}, Exact Match: {score}%, Samples: {len(predictions)}")

    # Save results
    out_path = f"{path}result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    print(f"\nResults saved to {out_path}")
    print(json.dumps(scores, ensure_ascii=False, indent=4))
