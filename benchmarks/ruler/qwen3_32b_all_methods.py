#!/usr/bin/env python3
"""Worker for the Qwen3-32B RULER shell runner.

The companion ``qwen3_32b_all_methods.sh`` prepares data and starts one worker
per method/length. Each worker evaluates all 13 tasks in one model process.
YaRN is applied in-process without editing model files.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm


RULER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RULER_ROOT.parents[1]
DATA_ROOT = RULER_ROOT / "benchmark_root" / "qwen3-32b" / "synthetic"
PRED_ROOT = RULER_ROOT / "benchmark_root_pred" / "synthetic" / "qwen3-32b-yarn2"
STATUS_DEFAULT = RULER_ROOT / "qwen3_32b_yarn2_status.tsv"
REPORT_DEFAULT = RULER_ROOT / "qwen3_32b_yarn2_report.md"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RULER_ROOT))

from benchmarks.common import METHODS, method_defaults, run_name  # noqa: E402


TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe",
    "qa_1", "qa_2",
)
LENGTHS = (4096, 8192, 16384, 32768, 65536)
DEFERRED_METHODS = (
    "retroinfer", "magicpig", "duo-attention", "headkv", "xattention", "minference",
)
DEFAULT_METHODS = tuple(method for method in METHODS if method not in DEFERRED_METHODS) + DEFERRED_METHODS
MODEL_ID = "Qwen/Qwen3-32B"
MODEL_ALIAS = "qwen3-32b"
YARN_CONFIG = {
    "rope_type": "yarn",
    "factor": 2.0,
    "original_max_position_embeddings": 32768,
}
SLOW_SECONDS_DEFAULT = 3600.0
LOW_SCORE_DEFAULT = 50.0


def install_qwen3_yarn_patch() -> None:
    """Set the official 64k YaRN config on Qwen3 configs before model init.

    Different method loaders use AutoConfig, AutoModel, or a concrete
    Qwen3ForCausalLM class. Patching both config loading paths covers all of
    those loaders while keeping the change local to this process.
    """
    from transformers import AutoConfig
    from transformers.configuration_utils import PretrainedConfig

    def apply(config):
        # 模型加载使用 return_unused_kwargs=True，需保留返回的二元组。
        if isinstance(config, tuple):
            loaded_config, unused_kwargs = config
            return apply(loaded_config), unused_kwargs
        if getattr(config, "model_type", None) == "qwen3":
            config.rope_scaling = dict(YARN_CONFIG)
        return config

    original_pretrained = PretrainedConfig.from_pretrained.__func__

    def pretrained_from_pretrained(cls, *args, **kwargs):
        return apply(original_pretrained(cls, *args, **kwargs))

    PretrainedConfig.from_pretrained = classmethod(pretrained_from_pretrained)

    original_auto = AutoConfig.from_pretrained.__func__

    def auto_from_pretrained(cls, *args, **kwargs):
        return apply(original_auto(cls, *args, **kwargs))

    AutoConfig.from_pretrained = classmethod(auto_from_pretrained)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, choices=(MODEL_ID, MODEL_ALIAS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--lengths", nargs="+", type=int, default=list(LENGTHS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None, help="Limit inference examples per task")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=PRED_ROOT)
    parser.add_argument("--status-file", type=Path, default=STATUS_DEFAULT)
    parser.add_argument("--report-file", type=Path, default=REPORT_DEFAULT)
    parser.add_argument("--slow-seconds", type=float, default=SLOW_SECONDS_DEFAULT,
                        help="Flag a method/length result as slow at this wall time")
    parser.add_argument("--low-score", type=float, default=LOW_SCORE_DEFAULT,
                        help="Flag a task score below this percentage")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite prediction files and rerun")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.num_samples <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--num-samples and --limit must be positive")
    if any(length not in LENGTHS for length in args.lengths):
        parser.error(f"--lengths must be selected from {LENGTHS}")
    if any(method in DEFERRED_METHODS for method in args.methods):
        ordered = [m for m in args.methods if m not in DEFERRED_METHODS]
        ordered += [m for m in DEFERRED_METHODS if m in args.methods]
        args.methods = ordered
    return args


def load_task_config(task: str) -> dict[str, Any]:
    module = importlib.import_module("data.synthetic.constants")
    customized = yaml.safe_load((RULER_ROOT / "synthetic.yaml").read_text(encoding="utf-8"))
    config = dict(customized[task])
    config.update(module.TASKS[config["task"]])
    return config


def prepare_task(args: argparse.Namespace, length: int, task: str) -> Path:
    data_dir = args.data_root / str(length) / "data"
    data_file = data_dir / task / "validation.jsonl"
    if data_file.exists() and not args.no_resume:
        try:
            if len([line for line in data_file.read_text(encoding="utf-8").splitlines() if line.strip()]) == args.num_samples:
                return data_file
        except OSError:
            pass
    if args.no_resume and data_file.exists():
        data_file.unlink()

    command = [
        sys.executable, str(RULER_ROOT / "data" / "prepare.py"),
        "--save_dir", str(data_dir),
        "--benchmark", "synthetic",
        "--task", task,
        "--tokenizer_path", args.model,
        "--tokenizer_type", "hf",
        "--max_seq_length", str(length),
        "--model_template_type", "qwen3-chat",
        "--num_samples", str(args.num_samples),
    ]
    print(f"[prepare] length={length} task={task} samples={args.num_samples}", flush=True)
    subprocess.run(command, cwd=RULER_ROOT, check=True)
    if not data_file.exists():
        raise FileNotFoundError(f"RULER prepare did not create {data_file}")
    actual_samples = len(read_jsonl(data_file))
    if actual_samples != args.num_samples:
        raise RuntimeError(
            f"RULER prepare created {actual_samples} samples for {task}/{length}; "
            f"expected {args.num_samples}"
        )
    return data_file


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sample_match_key(row: dict[str, Any]) -> tuple[int, str, str, str]:
    """Identify a prepared sample when migrating predictions without sample_id."""
    return (
        int(row["index"]),
        json.dumps(row.get("outputs", []), sort_keys=True, ensure_ascii=False),
        json.dumps(row.get("others", {}), sort_keys=True, ensure_ascii=False),
        json.dumps(row.get("length"), sort_keys=True, ensure_ascii=False),
    )


def read_completed(path: Path, data: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Return completed predictions keyed by validation row, not RULER index.

    Synthetic NIAH data can reuse ``index`` for different generated examples.
    Older prediction files do not have a row identifier, so match them to the
    validation data by index and expected answer metadata before resuming.
    """
    if not path.exists():
        return {}
    samples_by_key: dict[tuple[int, str, str, str], list[int]] = {}
    samples_by_index: dict[int, list[int]] = {}
    for sample_id, sample in enumerate(data):
        samples_by_key.setdefault(sample_match_key(sample), []).append(sample_id)
        samples_by_index.setdefault(int(sample["index"]), []).append(sample_id)

    available = set(range(len(data)))
    completed: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if "index" not in row:
            continue
        # Error/empty rows are checkpoints of failed attempts, not completed
        # samples. Drop them when normalizing the file so a rerun retries them.
        if row.get("error") or not str(row.get("pred", "")).strip():
            continue
        sample_id = row.get("sample_id")
        if sample_id is not None:
            try:
                sample_id = int(sample_id)
            except (TypeError, ValueError):
                sample_id = None
            if (sample_id not in available or sample_id is None
                    or int(data[sample_id]["index"]) != int(row["index"])):
                sample_id = None

        if sample_id is None:
            sample_id = next(
                (candidate for candidate in samples_by_key.get(sample_match_key(row), [])
                 if candidate in available),
                None,
            )
        if sample_id is None:
            sample_id = next(
                (candidate for candidate in samples_by_index.get(int(row["index"]), [])
                 if candidate in available),
                None,
            )
        if sample_id is None:
            continue

        normalized = dict(row)
        normalized["sample_id"] = sample_id
        completed[sample_id] = normalized
        available.remove(sample_id)
    return completed


def write_completed(path: Path, completed: dict[int, dict[str, Any]]) -> None:
    """Keep prediction rows in validation order for the line-based evaluator."""
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample_id in sorted(completed):
            json.dump(completed[sample_id], handle, ensure_ascii=False)
            handle.write("\n")
    os.replace(temporary, path)


def prediction_dir(args: argparse.Namespace, method: str, length: int) -> Path:
    overrides = default_overrides(method)
    signature = run_name(method, 1024, overrides)
    # Include YaRN in the model-root and signature so native-context runs can
    # never be mixed with these predictions.
    return args.output_root / method / f"{signature}__rope-yarn-2.0" / str(length)


def default_overrides(method: str) -> dict[str, Any]:
    try:
        config = method_defaults(method)
    except ValueError as exc:
        # MInference intentionally has an empty YAML file because its loader
        # does not expose cache-budget parameters.
        if "must be a YAML mapping" not in str(exc):
            raise
        config = {}
    overrides: dict[str, Any] = {}
    for key in ("fixthreshold", "sparsity"):
        if key in config:
            overrides[key] = config[key]
    # The unified loaders expose these framework parameters through --set.
    # method_defaults already supplies them for all methods that need one.
    return overrides


def model_config(model: Any) -> Any | None:
    candidates = [model]
    for attr in ("model", "base_model", "llm", "transformer"):
        value = getattr(model, attr, None)
        if value is not None:
            candidates.append(value)
            nested = getattr(value, "model", None)
            if nested is not None:
                candidates.append(nested)
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        if config is not None and getattr(config, "model_type", None) == "qwen3":
            return config
    return None


def verify_yarn(model: Any, method: str) -> None:
    config = model_config(model)
    if config is None:
        # Some wrappers (e.g. RetroInfer) do not expose their Hugging Face
        # config. Their loader still uses the config patch installed above.
        print(f"[yarn] {method}: wrapper exposes no Qwen3 config; load-time patch active", flush=True)
        return
    actual = getattr(config, "rope_scaling", None) or {}
    for key, expected in YARN_CONFIG.items():
        if actual.get(key) != expected:
            raise RuntimeError(f"{method} did not retain YaRN {key}={expected}: {actual}")
    print(f"[yarn] {method}: {actual}", flush=True)


def ensure_status_header(path: Path) -> None:
    if path.exists() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, delimiter="\t").writerow(
            ("method", "length", "status", "elapsed_seconds", "load_seconds", "inference_seconds",
             "examples", "score_mean", "low_score_tasks", "slow", "prediction_dir", "error")
        )


def append_status(path: Path, row: list[Any]) -> None:
    ensure_status_header(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        existing = [
            values for values in reader
            if len(values) < 2 or (values[0] != str(row[0]) or values[1] != str(row[1]))
        ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(existing)
        writer.writerow(row)


def evaluate_predictions(path: Path, tasks: list[str]) -> dict[str, float]:
    from evaluate import evaluate

    evaluate(Namespace(data_dir=path, tasks=tasks))
    summary = path / "summary.csv"
    if not summary.exists():
        return {}
    rows = list(csv.reader(summary.open(encoding="utf-8", newline="")))
    if len(rows) < 3 or rows[1][0] != "Tasks" or rows[2][0] != "Score":
        return {}
    return {task: float(score) for task, score in zip(rows[1][1:], rows[2][1:])}


def update_report(args: argparse.Namespace) -> None:
    if not args.status_file.exists():
        return
    with args.status_file.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    lines = [
        "# Qwen3-32B RULER YaRN 2.0 run",
        "",
        "Configuration: budget 1024, 4k/8k/16k/32k/64k, 13 synthetic tasks, 50 samples/task by default.",
        "Qwen3 prompts use the official non-thinking chat suffix; every loaded Qwen3 model receives YaRN factor 2.0 with original positions 32768.",
        "The existing RULER `qwen-chat` entry stays available for Qwen2.5; Qwen3 uses its own empty think-block suffix.",
        "Low-score flag: task mean below {:.1f}%. Slow flag: method/length wall time above {:.0f} seconds.".format(
            args.low_score, args.slow_seconds
        ),
        "",
        "| Method | Length | Status | Wall (s) | Load (s) | Inference (s) | Mean task score | Low tasks | Slow |",
        "|---|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        try:
            mean_score = float(row["score_mean"]) if row["score_mean"] else float("nan")
        except ValueError:
            mean_score = float("nan")
        elapsed = float(row["elapsed_seconds"] or 0)
        load_seconds = float(row.get("load_seconds") or 0)
        inference_seconds = float(row.get("inference_seconds") or 0)
        score_text = f"{mean_score:.2f}" if mean_score == mean_score else "unavailable"
        low_text = row.get("low_score_tasks") or "-"
        slow_text = "yes" if row["slow"] == "yes" or elapsed > args.slow_seconds else "no"
        lines.append(
            f"| {row['method']} | {int(row['length']) // 1024}k | {row['status']} | "
            f"{elapsed:.1f} | {load_seconds:.1f} | {inference_seconds:.1f} | "
            f"{score_text} | {low_text} | {slow_text} |"
        )
        if row.get("error"):
            lines.extend(("", f"- `{row['method']}` {row['length']} error: `{row['error']}`"))
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_method_length(args: argparse.Namespace, method: str, length: int) -> None:
    print(f"\n===== method={method} length={length} budget=1024 yarn=2.0 =====", flush=True)
    started = time.monotonic()
    out_dir = prediction_dir(args, method, length)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_files = {task: args.data_root / str(length) / "data" / task / "validation.jsonl" for task in args.tasks}
    pending_by_task: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    completed_by_task: dict[str, dict[int, dict[str, Any]]] = {}
    total_by_task: dict[str, int] = {}
    completed_all = 0
    for task, data_file in task_files.items():
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        data = read_jsonl(data_file)
        if args.limit is not None:
            data = data[:args.limit]
        pred_file = out_dir / f"{task}.jsonl"
        if args.no_resume and pred_file.exists():
            pred_file.unlink()
        done = read_completed(pred_file, data)
        pending = [
            (sample_id, sample)
            for sample_id, sample in enumerate(data)
            if sample_id not in done
        ]
        completed_all += len(done)
        completed_by_task[task] = done
        if pred_file.exists():
            write_completed(pred_file, done)
        pending_by_task[task] = pending
        total_by_task[task] = len(data)

    pending_count = sum(map(len, pending_by_task.values()))
    load_elapsed = 0.0
    inference_elapsed = 0.0
    if pending_count == 0:
        scores = evaluate_predictions(out_dir, list(args.tasks))
        elapsed = time.monotonic() - started
        examples = completed_all
        status = "complete"
        error = ""
    else:
        from model_wrappers import RulerModel

        overrides = [f"{key}={json.dumps(value)}" for key, value in default_overrides(method).items()]
        wrapper = None
        examples = completed_all
        errors = []
        try:
            load_started = time.monotonic()
            wrapper = RulerModel(method, args.model, 1024, args.device, overrides, max_seq_length=length)
            load_elapsed = time.monotonic() - load_started
            verify_yarn(wrapper.model, method)
            max_new_tokens = {task: int(load_task_config(task)["tokens_to_generate"]) for task in args.tasks}
            inference_started = time.monotonic()
            for task in args.tasks:
                pending = pending_by_task[task]
                if not pending:
                    continue
                done = completed_by_task[task]
                pred_file = out_dir / f"{task}.jsonl"
                print(f"[infer] length={length} task={task} pending={len(pending)}; model remains loaded", flush=True)
                progress_start = total_by_task[task] - len(pending)
                with pred_file.open("a", encoding="utf-8", buffering=1) as handle:
                    for sample_id, sample in tqdm(
                        pending,
                        total=total_by_task[task],
                        initial=progress_start,
                        desc=f"{task} ({length // 1024}k)",
                        unit="sample",
                    ):
                        index = int(sample["index"])
                        try:
                            prediction = wrapper.generate(sample["input"], max_new_tokens[task])
                            if not prediction.strip():
                                raise ValueError("Model generated an empty prediction")
                            record = {
                                "sample_id": sample_id,
                                "index": index,
                                "pred": prediction,
                                "outputs": sample.get("outputs", []),
                                "others": sample.get("others", {}),
                                "truncation": sample.get("truncation"),
                                "length": sample.get("length"),
                            }
                        except Exception as exc:  # preserve progress and score failed samples as empty
                            if (type(exc).__name__ == "AcceleratorError"
                                    or "launch timed out" in str(exc).lower()):
                                print(
                                    f"[fatal-cuda-error] {task}[{index}] {exc}; stopping this worker. "
                                    "Successful samples are saved and failed samples will be retried.",
                                    flush=True,
                                )
                                raise
                            message = f"{type(exc).__name__}: {exc}"
                            errors.append(f"{task}[{index}] {message}")
                            record = {
                                "sample_id": sample_id, "index": index, "pred": "",
                                "outputs": sample.get("outputs", []),
                                "others": sample.get("others", {}), "length": sample.get("length"),
                                "error": message,
                            }
                            print(f"[sample-error] {task}[{index}] {message}", flush=True)
                        json.dump(record, handle, ensure_ascii=False)
                        handle.write("\n")
                        examples += 1
                        done[sample_id] = record
                write_completed(pred_file, done)
                successful = sum(
                    not record.get("error") and bool(str(record.get("pred", "")).strip())
                    for record in done.values()
                )
                task_status = "task-complete" if successful == total_by_task[task] else "task-incomplete"
                print(
                    f"[{task_status}] method={method} model={args.model} length={length} "
                    f"task={task} successful={successful}/{total_by_task[task]}",
                    flush=True,
                )
            inference_elapsed = time.monotonic() - inference_started
            scores = evaluate_predictions(out_dir, list(args.tasks))
            status = "complete" if not errors else "partial-sample-errors"
            error = "; ".join(errors[:5])
        finally:
            try:
                if wrapper is not None:
                    wrapper.close()
            finally:
                # Qwen3 attention patches bind methods back onto their modules,
                # creating reference cycles. Collect them before loading the
                # next context length so stale CUDA tensors do not cause OOM.
                wrapper = None
                gc.collect()
                import torch

                torch.cuda.empty_cache()
        elapsed = time.monotonic() - started

    score_mean = sum(scores.values()) / len(scores) if scores else None
    low_tasks = [task for task, score in scores.items() if score < args.low_score]
    slow = elapsed > args.slow_seconds
    append_status(args.status_file, [
        method, length, status, f"{elapsed:.2f}", f"{load_elapsed:.2f}",
        f"{inference_elapsed:.2f}", examples,
        "" if score_mean is None else f"{score_mean:.2f}", ",".join(low_tasks),
        "yes" if slow else "no", str(out_dir), error,
    ])
    update_report(args)
    print(
        f"[done] method={method} length={length} status={status} elapsed={elapsed:.1f}s "
        f"examples={examples} mean_score={score_mean if score_mean is not None else 'unavailable'} "
        f"low_tasks={low_tasks} slow={slow}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.model = MODEL_ID
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    args.status_file = args.status_file.resolve()
    args.report_file = args.report_file.resolve()
    if args.limit is not None:
        args.num_samples = min(args.num_samples, args.limit)

    if args.worker and len(args.methods) != 1:
        raise ValueError("A worker process must receive exactly one method")

    if not args.worker:
        if not args.prepare_only:
            raise SystemExit(
                "Use benchmarks/ruler/qwen3_32b_all_methods.sh to run the full grid. "
                "This Python file only prepares data or runs one internal worker."
            )
        for length in args.lengths:
            print(f"\n===== prepare length={length} tasks={len(args.tasks)} =====", flush=True)
            for task in args.tasks:
                prepare_task(args, length, task)
        return 0

    # RetroInfer derives a maximum generation/cache length from this shared
    # model table; extend it for the requested factor-2 64k run.
    from benchmarks.common import MODEL_CONTEXT_LENGTHS

    MODEL_CONTEXT_LENGTHS[MODEL_ALIAS] = 65536
    install_qwen3_yarn_patch()
    ensure_status_header(args.status_file)
    failed = False
    for method in args.methods:
        for length in args.lengths:
            block_started = time.monotonic()
            try:
                run_method_length(args, method, length)
            except Exception as exc:
                failed = True
                message = f"{type(exc).__name__}: {exc}"
                block_elapsed = time.monotonic() - block_started
                print(f"[method-length-error] method={method} length={length}: {message}", flush=True)
                traceback.print_exc()
                append_status(args.status_file, [
                    method, length, "failed", f"{block_elapsed:.2f}", "", "", 0, "", "", "no",
                    str(prediction_dir(args, method, length)), message,
                ])
                update_report(args)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
