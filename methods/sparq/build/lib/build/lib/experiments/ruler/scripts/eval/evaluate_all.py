# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
Evaluate ALL jsonl files under data_dir using ONE fixed metric task config,
and write scores to summary.csv (one row per file).

Running:
python evaluate.py \
  --data_dir /path/to/preds \
  --benchmark synthetic \
  --metric_task niah_single_1
"""

import re
import os
import argparse
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

import pandas as pd
import importlib
import yaml
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="path to the prediction jsonl files")
parser.add_argument("--benchmark", type=str, default="synthetic", help="used to load metrics (constants/yaml)")
parser.add_argument("--metric_task", type=str, default="niah_single_1", help="use this task's metric_fn to score ALL files")
parser.add_argument("--verbose", type=int, default=0, help="how many lines you want to display.")
args = parser.parse_args()


def postprocess_pred(predict_str: str, task_config: dict):
    predict_str = (predict_str or "").strip()
    np_pattern = re.compile(r"[\x00-\x1f]")
    predict_str = np_pattern.sub("\n", predict_str).strip()
    return predict_str


def get_pred_and_ref(
    predictions_file: str,
    task_config: dict,
    input_field: str = "input",
    references_field: str = "outputs",
    prediction_field: str = "pred",
    metadata_field: str = "others",
):
    lines = read_manifest(predictions_file)

    inputs = []
    predicts = []
    references = []
    indices = []

    for line in tqdm(lines, desc=f"Read {os.path.basename(predictions_file)}"):
        input_ = line.get(input_field, "")
        predict = line.get(prediction_field, "")
        predict = postprocess_pred(predict, task_config)

        reference = line.get(references_field, None)
        if reference is None:
            reference = [line.get("output", "")]
        if not isinstance(reference, list):
            reference = [reference]

        others = line.get(metadata_field, {}) or {}
        index = others.get("id", line.get("index", None))
        if index is None:
            index = str(len(indices))

        inputs.append(input_)
        predicts.append(predict)
        references.append(reference)
        indices.append(index)

    return inputs, predicts, references, indices


def run_evaluation_per_task(task_config: dict, predictions_file: str, verbose: int = 0):
    inputs, predicts, references, indices = get_pred_and_ref(
        predictions_file=predictions_file,
        task_config=task_config,
    )

    task_nulls = f"{sum([len(x) == 0 for x in predicts])}/{len(predicts)}"

    metric_fn = task_config.get("metric_fn", None)
    if (
        metric_fn is not None
        and len(references) > 0
        and len(references[0]) > 0
        and references[0][0] is not None
    ):
        task_score = metric_fn(predicts, references)
    else:
        task_score = 0.0

    if verbose != 0:
        print("=" * 40)
        for i, (input_, reference, predict) in enumerate(zip(inputs, references, predicts)):
            print(f"Input     : {input_}")
            print(f"Reference : {reference}")
            print(f"Prediction: {predict}")
            print("=" * 40)
            if i >= verbose:
                break

    return task_score, task_nulls, predicts, indices


def write_evaluation_by_filename(results: dict, out_dir: str):
    """
    results: { filename: {'score': float, 'nulls': 'x/y'} }
    """
    rows = []
    for filename, r in results.items():
        rows.append(
            {
                "File": filename,
                "Score": r["score"],
                "Nulls": r["nulls"],
            }
        )

    df = pd.DataFrame(rows).sort_values("File")
    output_file = os.path.join(out_dir, "summary.csv")
    df.to_csv(output_file, index=False)

    print("\n=============================================\n")
    print(df)
    print(f"\nSaved eval results to {output_file}")


def write_submission(results: dict, out_dir: str):

    COLUMNS = ["Task", "ID", "Prediction"]
    dfs = pd.DataFrame(columns=COLUMNS, data=[])

    for task, result in results.items():
        df = pd.DataFrame(
            {
                "Task": task,
                "ID": result["indices"],
                "Prediction": result["predicts"],
            }
        )
        dfs = pd.concat((dfs, df[COLUMNS]))

    output_file = os.path.join(out_dir, "submission.csv")
    dfs = dfs.reset_index(drop=True)
    dfs.to_csv(output_file, index=False)
    print(f"\nSaved submission results to {output_file}")


def aggregate_chunk(folder):
    jsonl_files = [file for file in os.listdir(folder) if Path(file).suffix == ".jsonl"]
    chunk_files = sorted([file for file in jsonl_files if re.match(r".*[^_]+-\d+\.jsonl", file)])
    chunk_files_dict = defaultdict(list)

    for file in chunk_files:
        task = "-".join(file.split("-")[:-1])
        chunk_files_dict[task].append(file)

    for task, files in chunk_files_dict.items():
        lines = []
        for file in sorted(files):
            file_path = os.path.join(folder, file)
            lines += read_manifest(file_path)
            os.remove(file_path)
        write_manifest(os.path.join(folder, f"{task}.jsonl"), lines)


def _load_tasks_config():
    """
    Load TASKS from benchmark.constants + benchmark.yaml (same as your current logic),
    return merged TASKS dict and tasks_base.
    """
    curr_folder = os.path.dirname(os.path.abspath(__file__))

    # constants
    try:
        module = importlib.import_module(f"{args.benchmark}.constants")
        tasks_base = module.TASKS
    except Exception as e:
        print(f"[WARN] Cannot import {args.benchmark}.constants, metrics may be missing. Error: {e}")
        tasks_base = {}

    # yaml
    tasks_customized = {}
    yaml_path = os.path.join(curr_folder, f"../{args.benchmark}.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            tasks_customized = yaml.safe_load(f) or {}
    else:
        print(f"[WARN] YAML not found: {yaml_path}.")

    TASKS = tasks_customized
    for _, config in TASKS.items():
        base_task_name = config.get("task", None)
        if base_task_name in tasks_base:
            config.update(tasks_base[base_task_name])

    return TASKS, tasks_base


def main():
    # 1) Load config
    TASKS, tasks_base = _load_tasks_config()
    print(f"Loaded task configs (from yaml): {list(TASKS.keys())}")

    # 2) Pick ONE fixed metric task config
    fixed_task = args.metric_task
    if fixed_task in TASKS:
        fixed_config = TASKS[fixed_task]
    elif fixed_task in tasks_base:
        fixed_config = tasks_base[fixed_task]
    else:
        raise ValueError(
            f"Cannot find metric task '{fixed_task}' in TASKS (yaml) or tasks_base (constants). "
            f"Available keys in yaml: {list(TASKS.keys())[:20]}..."
        )

    if fixed_config.get("metric_fn", None) is None:
        raise ValueError(f"Task '{fixed_task}' has no metric_fn. Cannot score files.")

    print(f"[INFO] Use fixed metric_task='{fixed_task}' to score ALL jsonl files.")

    # 3) Aggregate chunk files
    aggregate_chunk(args.data_dir)

    # 4) Evaluate all jsonl files using fixed_config
    jsonl_files = sorted([f for f in os.listdir(args.data_dir) if Path(f).suffix == ".jsonl"])
    if not jsonl_files:
        print(f"No .jsonl files found under: {args.data_dir}")
        return

    eval_results = {}   # key = filename
    subm_results = {}

    for filename in jsonl_files:
        pred_path = os.path.join(args.data_dir, filename)
        print(f"\nEvaluate file: {filename} (fixed_metric={fixed_task})")

        task_score, task_nulls, predicts, indices = run_evaluation_per_task(
            task_config=fixed_config,
            predictions_file=pred_path,
            verbose=args.verbose,
        )

        eval_results[filename] = {"score": task_score, "nulls": task_nulls}
        subm_results[Path(filename).stem] = {"predicts": predicts, "indices": indices}

    # 5) Write outputs
    write_evaluation_by_filename(eval_results, args.data_dir)
    write_submission(subm_results, args.data_dir)


if __name__ == "__main__":
    main()
