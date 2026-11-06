#!/usr/bin/env python3
"""在一个模型/方法/batch 进程中复用权重，逐个测试输入长度。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import signal
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common import model_alias, parse_set
from benchmarks.efficency_vram import pred as benchmark
from benchmarks.efficency_vram.tools.result_store import append_result
from benchmarks.efficency_vram.tools.run_grid import (
    append_status,
    cell_key,
    dominated_failure,
    error_from_log,
    log_contains_oom,
    metrics_from_summary,
    read_existing_status,
    threshold_failures,
)
from loaders import load_model_and_tokenizer


class CellTimeout(Exception):
    """单个组合超过配置的最大运行时间。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--input-lengths", nargs="+", type=int, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--task", default="niah_single_1")
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measured-rounds", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", default="1")
    return parser.parse_args()


def write_current_cell(path: Path, values: dict) -> None:
    lines = [f"{key}={value}" for key, value in values.items()]
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def make_base_args(args: argparse.Namespace) -> argparse.Namespace:
    base_length = min(args.input_lengths)
    argv = [
        "--model",
        args.model,
        "--method",
        args.method,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--input-length",
        str(base_length),
        "--context-length",
        str(base_length),
        "--fixed-output-length",
        str(args.output_length),
        "--measurement",
        "both",
        "--budget",
        str(args.budget),
        "--task",
        args.task,
        "--data-dir",
        str(args.data_file.resolve()),
        "--num-samples",
        str(args.batch_size * args.measured_rounds),
        "--warmup",
        str(args.warmup),
        "--seed",
        "42",
        "--experiment",
        args.experiment,
        "--output-root",
        str(args.output_root.resolve()),
        "--save-predictions",
        "--overwrite",
    ]
    options = benchmark.parse_args(argv)
    options._method_args = SimpleNamespace(**parse_set(options.set, options.method))
    return options


def prepare_cell_options(
    options: argparse.Namespace,
    batch_size: int,
    input_length: int,
    measured_rounds: int,
) -> None:
    options.batch_size = batch_size
    options.input_length = input_length
    options.context_length = input_length
    options.max_seq_length = max(input_length, options.context_length)
    options.num_samples = batch_size * measured_rounds
    options.num_batches = None
    options.measurement = "both"
    options.overwrite = True
    options.save_predictions = True


def make_metadata(
    options: argparse.Namespace,
    data_file: Path,
    samples: list[dict],
    batches: list[list[dict]],
    paths: dict,
    model: torch.nn.Module,
    devices: tuple[torch.device, ...],
) -> dict:
    arguments = {
        key: value
        for key, value in vars(options).items()
        if not key.startswith("_")
    }
    metadata = {
        "arguments": arguments,
        "data_file": str(data_file),
        "selected_samples": len(samples),
        "dropped_samples": len(samples) - sum(map(len, batches)),
        "num_batches": len(batches),
        "paths": paths,
        "status": "running",
        "backend": "native_generate" if options.method in benchmark.NATIVE_METHODS else "forward",
        "model_path": options.model_path,
        "effective_method_config": {
            **vars(options._method_args),
            "benchmark_budget": options.budget,
        },
        "torch_version": torch.__version__,
        "transformers_version": benchmark.version("transformers"),
        "cuda_visible_devices": args_physical_gpu(),
        "devices": {
            str(device): torch.cuda.get_device_name(device)
            for device in devices
        },
    }
    return metadata


def args_physical_gpu() -> str | None:
    return os.environ.get("CUDA_VISIBLE_DEVICES")


def write_cell_log(path: Path, row: dict, details: str = "") -> None:
    lines = [
        f"model={row.get('model')}",
        f"method={row.get('method')}",
        f"batch_size={row.get('batch_size')}",
        f"input_length={row.get('input_length')}",
        f"status={row.get('status')}",
        f"elapsed_s={row.get('elapsed_s')}",
        f"error={row.get('error', '')}",
    ]
    if details:
        lines.extend(["", details])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def timeout_handler(_signum, _frame) -> None:
    raise CellTimeout("单组合达到超时上限。")


def clear_cuda_state(devices: tuple[torch.device, ...]) -> None:
    gc.collect()
    for device in devices:
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        try:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        except Exception:
            pass


def run_cell(
    args: argparse.Namespace,
    options: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer,
    input_device: torch.device,
    devices: tuple[torch.device, ...],
    data_file: Path,
    input_length: int,
) -> dict:
    batch_size = args.batch_size
    sample_count = batch_size * args.measured_rounds
    prepare_cell_options(options, batch_size, input_length, args.measured_rounds)
    cell_log_path = (
        args.state_dir
        / "logs"
        / model_alias(args.model)
        / args.method
        / f"b{batch_size}_i{input_length}.log"
    )
    current_cell_path = args.state_dir / "current_cell.txt"
    started = time.monotonic()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    current_cell = {
        "model": args.model,
        "method": args.method,
        "batch_size": batch_size,
        "input_length": input_length,
        "started_at": started_at,
        "started_monotonic": started,
    }
    write_current_cell(current_cell_path, current_cell)
    print(f"CELL_START_JSON:{json.dumps(current_cell, ensure_ascii=False)}", flush=True)

    row = {
        "model": args.model,
        "model_alias": model_alias(args.model),
        "method": args.method,
        "batch_size": batch_size,
        "input_length": input_length,
        "output_length": args.output_length,
        "status": "failed",
        "phase": "preparing",
        "started_at": started_at,
    }
    detail = ""
    metadata = None
    pred_json_path = args.output_root / "pred.json"
    records = []
    predictions = []
    summary = None
    original_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)

    try:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        samples = benchmark.read_samples(data_file, sample_count)
        batches = list(benchmark.iter_batches(samples, batch_size))
        paths = {"predictions": pred_json_path}
        metadata = make_metadata(
            options,
            data_file,
            samples,
            batches,
            paths,
            model,
            devices,
        )
        for warmup_index in range(options.warmup):
            benchmark.predict_batch(
                options,
                model,
                tokenizer,
                batches[0],
                input_device,
                devices,
                measured=False,
            )
            print(
                f"Warmup {warmup_index + 1}/{options.warmup} "
                f"b{batch_size} i{input_length}",
                flush=True,
            )

        for batch_index, batch in enumerate(batches):
            tokens, record = benchmark.predict_batch(
                options,
                model,
                tokenizer,
                batch,
                input_device,
                devices,
                measured=True,
            )
            record.update(
                batch_index=batch_index,
                method=options.method,
                model=options.model,
                budget=options.budget,
                task=options.task,
                measurement=options.measurement,
                configured_batch_size=options.batch_size,
            )
            records.append(record)

            decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            for sample, token_ids, prediction in zip(batch, tokens.tolist(), decoded):
                predictions.append({
                    "index": sample["index"],
                    "pred": prediction,
                    "output_token_ids": token_ids,
                })

            print(
                f"Batch {batch_index + 1}/{len(batches)}: "
                f"ttft_s={record.get('ttft_s')}, "
                f"tpot_ms={record.get('tpot_ms')}, "
                f"peak_allocated_mib={record.get('peak_allocated_mib')}",
                flush=True,
            )

        summary = {
            **benchmark.summarize(records),
            "method": options.method,
            "model": options.model,
            "task": options.task,
            "measurement": options.measurement,
        }
        metadata["status"] = "complete"

        row.update(metrics_from_summary(
            summary,
            batch_size,
            args.measured_rounds,
            args.output_length,
            pred_json_path,
        ))
        row["status"] = "passed"
        row["phase"] = "complete"
        detail = f"pred_json_path={pred_json_path}"
    except CellTimeout as error:
        row["status"] = "timeout"
        row["phase"] = "generation"
        row["error"] = f"{error} 超时阈值为 {args.timeout_seconds} 秒。"
        detail = traceback.format_exc()
    except Exception as error:
        error_text = traceback.format_exc()
        row["status"] = "oom" if log_contains_oom(error_text) else "failed"
        row["phase"] = "generation"
        row["error"] = error_from_log(error_text)
        detail = error_text
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, original_handler)
        row["elapsed_s"] = round(time.monotonic() - started, 3)
        result_id = uuid.uuid4().hex
        row["pred_json_path"] = str(pred_json_path)
        row["pred_result_id"] = result_id
        row["summary_path"] = str(pred_json_path)
        if metadata is not None and row["status"] != "passed":
            metadata.update(status="failed", error=row.get("error"))
        result = {
            "result_id": result_id,
            "model": args.model,
            "model_alias": model_alias(args.model),
            "experiment": args.experiment,
            "method": args.method,
            "task": args.task,
            "batch_size": batch_size,
            "input_length": input_length,
            "output_length": args.output_length,
            "status": row["status"],
            "started_at": started_at,
            "elapsed_s": row["elapsed_s"],
            "error": row.get("error"),
            "skip_reason": row.get("skip_reason"),
            "metrics": row,
            "measurements": records,
            "summary": summary,
            "predictions": predictions,
            "metadata": metadata,
        }
        append_result(pred_json_path, result)
        write_cell_log(cell_log_path, row, detail)
        clear_cuda_state(devices)
        try:
            current_cell_path.unlink()
        except FileNotFoundError:
            pass

    return row


def mark_setup_failure(
    args: argparse.Namespace,
    input_lengths: list[int],
    error_text: str,
) -> None:
    status_file = args.state_dir / "status.csv"
    rows = read_existing_status(status_file)
    pending = []
    for input_length in input_lengths:
        key = cell_key(args.model, args.method, args.batch_size, input_length)
        if key not in rows:
            pending.append(input_length)

    for position, input_length in enumerate(pending):
        if position == 0:
            status = "failed"
            setup_failure = True
            error = error_from_log(error_text)
        else:
            status = "skipped_setup_failure"
            setup_failure = True
            error = "模型或方法初始化失败；见同一 batch 的首个失败组合。"
        row = {
            "model": args.model,
            "model_alias": model_alias(args.model),
            "method": args.method,
            "batch_size": args.batch_size,
            "input_length": input_length,
            "output_length": args.output_length,
            "status": status,
            "phase": "loading",
            "setup_failure": setup_failure,
            "error": error,
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        append_status(status_file, row)
        print(f"CELL_RESULT_JSON:{json.dumps(row, ensure_ascii=False)}", flush=True)


def main() -> None:
    args = parse_args()
    args.data_file = args.data_file.resolve()
    args.output_root = args.output_root.resolve()
    args.state_dir = args.state_dir.resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.physical_gpu = str(args.physical_gpu)

    options = make_base_args(args)
    data_file = benchmark.resolve_data_file(options)
    status_file = args.state_dir / "status.csv"
    current_cell_path = args.state_dir / "current_cell.txt"

    existing_rows = read_existing_status(status_file)
    pending = []
    for input_length in args.input_lengths:
        key = cell_key(args.model, args.method, args.batch_size, input_length)
        if key in existing_rows:
            continue
        if dominated_failure(
            threshold_failures(existing_rows, args.model, args.method),
            args.batch_size,
            input_length,
        ):
            continue
        pending.append(input_length)
    if not pending:
        print("这个 batch 的组合均已完成或被 OOM/timeout 阈值覆盖。", flush=True)
        return

    print(
        f"加载一次模型：{args.model} / {args.method} / batch={args.batch_size}；"
        f"待测长度={pending}",
        flush=True,
    )
    try:
        options._cleanup = None
        model, tokenizer = load_model_and_tokenizer(options)
        input_device, devices = benchmark.model_devices(model, args.method)
        print(
            f"模型已加载，input_device={input_device}，devices={[str(device) for device in devices]}",
            flush=True,
        )
    except Exception:
        mark_setup_failure(args, pending, traceback.format_exc())
        try:
            current_cell_path.unlink()
        except FileNotFoundError:
            pass
        return

    try:
        for input_length in args.input_lengths:
            current_rows = read_existing_status(status_file)
            key = cell_key(args.model, args.method, args.batch_size, input_length)
            if key in current_rows:
                continue

            failure = dominated_failure(
                threshold_failures(current_rows, args.model, args.method),
                args.batch_size,
                input_length,
            )
            if failure is not None:
                row = {
                    "model": args.model,
                    "model_alias": model_alias(args.model),
                    "method": args.method,
                    "batch_size": args.batch_size,
                    "input_length": input_length,
                    "output_length": args.output_length,
                    "status": f"skipped_{failure['status']}_threshold",
                    "phase": "not_started",
                    "skip_reason": (
                        f"{failure['status']} at batch={failure['batch_size']}, "
                        f"input={failure['input_length']}"
                    ),
                    "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                }
                append_status(status_file, row)
                print(f"CELL_RESULT_JSON:{json.dumps(row, ensure_ascii=False)}", flush=True)
                continue

            row = run_cell(
                args,
                options,
                model,
                tokenizer,
                input_device,
                devices,
                data_file,
                input_length,
            )
            append_status(status_file, row)
            print(f"CELL_RESULT_JSON:{json.dumps(row, ensure_ascii=False)}", flush=True)

    finally:
        if options._cleanup is not None:
            options._cleanup()
        del model
        del tokenizer
        clear_cuda_state(devices)


if __name__ == "__main__":
    main()
