#!/usr/bin/env python3
"""在一个模型/方法/batch 进程中复用权重，逐个测试输入长度。"""

from __future__ import annotations

import argparse
import gc
import json
import random
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common import model_alias
from benchmarks.efficency_vram import pred as benchmark
from benchmarks.efficency_vram.tools.reporting import (
    append_history_event,
    cell_key,
    format_number,
    method_history_path,
    metrics_from_summary,
    read_history_state,
    round_measurement_records,
    round_floats,
    summarize,
)
from benchmarks.efficency_vram.tools.result_store import append_predictions
from benchmarks.efficency_vram.tools.runtime import (
    dominated_failure,
    error_from_log,
    log_contains_oom,
    threshold_failures,
)
from benchmarks.efficency_vram.tools.unsupported_batch import is_unsupported_batch_error
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
    parser.add_argument("--efficiency-backend", action="store_true",
                        help="用 Quest/ClusterKV 自带的效率实现（仅效率脚本）。")
    return parser.parse_args()


def write_current_cell(path: Path, values: dict) -> None:
    lines = [f"{key}={value}" for key, value in values.items()]
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def make_base_args(args: argparse.Namespace) -> argparse.Namespace:
    max_length = max(args.input_lengths)
    max_seq_length = max_length
    if args.method == "pqcache":
        max_seq_length += max(0, args.output_length - 1024)
    return argparse.Namespace(
        model=args.model,
        method=args.method,
        device="cuda:0",
        batch_size=args.batch_size,
        input_length=max_length,
        context_length=max_length,
        max_seq_length=max_seq_length,
        fixed_output_length=args.output_length,
        task=args.task,
        warmup=args.warmup,
        budget=args.budget,
        set=[],
        experiment=args.experiment,
        output_root=args.output_root,
        efficiency_backend=args.efficiency_backend,
        dataset="ruler",
    )


def prepare_cell_options(
    options: argparse.Namespace,
    input_length: int,
) -> None:
    options.input_length = input_length
    options.context_length = input_length
    options.max_seq_length = input_length


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


def record_status(history_file: Path, row: dict) -> None:
    append_history_event(history_file, row)
    print(f"CELL_RESULT_JSON:{json.dumps(round_floats(row), ensure_ascii=False)}", flush=True)


def timeout_handler(_signum, _frame) -> None:
    raise CellTimeout("单组合达到超时上限。")


def clear_cuda_state(devices: tuple[torch.device, ...]) -> None:
    gc.collect()
    for device in devices:
        torch.cuda.synchronize(device)
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


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
    prepare_cell_options(options, input_length)
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
    print(f"CELL_START_JSON:{json.dumps(round_floats(current_cell), ensure_ascii=False)}", flush=True)

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
    pred_json_path = benchmark.result_paths(options)["predictions"]
    records = []
    predictions = []
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
            record["batch_index"] = batch_index
            records.append(record)

            decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            predictions.extend(
                benchmark.prediction_records(
                    batch,
                    decoded,
                    batch_index + 1,
                    input_length,
                    args.output_length,
                )
            )

            print(
                f"Batch {batch_index + 1}/{len(batches)}: "
                f"ttft_s={format_number(record.get('ttft_s'))}, "
                f"tpot_ms={format_number(record.get('tpot_ms'))}, "
                f"peak_allocated_mib={format_number(record.get('peak_allocated_mib'))}",
                flush=True,
            )

        summary = summarize(records)
        row.update(metrics_from_summary(
            summary,
            batch_size,
            args.measured_rounds,
            args.output_length,
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
        if log_contains_oom(error_text):
            row["status"] = "oom"
        elif batch_size > 1 and is_unsupported_batch_error(error_text):
            row["status"] = "unsupported_batch"
            row["skip_reason"] = "运行时明确报告此方法不支持多 batch。"
        else:
            row["status"] = "failed"
        row["phase"] = "generation"
        row["error"] = error_from_log(error_text)
        detail = error_text
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, original_handler)
        row["elapsed_s"] = round(time.monotonic() - started, 2)
        row.update(round_floats(row))
        row["round_measurements"] = round_floats(
            round_measurement_records(records)
        )
        append_predictions(pred_json_path, predictions)
        write_cell_log(cell_log_path, row, detail)
        clear_cuda_state(devices)
        try:
            current_cell_path.unlink()
        except FileNotFoundError:
            pass

    return row


def mark_unsupported_batch_cells(
    args: argparse.Namespace,
    input_lengths: list[int],
    reason: str,
) -> None:
    history_file = method_history_path(
        args.output_root,
        args.model,
        args.experiment,
        args.method,
        args.task,
    )
    rows = read_history_state(history_file, args.model, args.method)
    for input_length in input_lengths:
        key = cell_key(args.model, args.method, args.batch_size, input_length)
        if key in rows:
            continue
        row = {
            "model": args.model,
            "model_alias": model_alias(args.model),
            "method": args.method,
            "batch_size": args.batch_size,
            "input_length": input_length,
            "output_length": args.output_length,
            "status": "skipped_unsupported_batch",
            "phase": "not_started",
            "skip_reason": reason,
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        record_status(history_file, row)


def mark_setup_failure(
    args: argparse.Namespace,
    input_lengths: list[int],
    error_text: str,
) -> None:
    history_file = method_history_path(
        args.output_root,
        args.model,
        args.experiment,
        args.method,
        args.task,
    )
    rows = read_history_state(history_file, args.model, args.method)
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
        record_status(history_file, row)


def main() -> None:
    args = parse_args()
    args.data_file = args.data_file.resolve()
    args.output_root = args.output_root.resolve()
    args.state_dir = args.state_dir.resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    options = make_base_args(args)
    data_file = args.data_file
    history_file = method_history_path(
        args.output_root,
        args.model,
        args.experiment,
        args.method,
        args.task,
    )
    current_cell_path = args.state_dir / "current_cell.txt"

    existing_rows = read_history_state(history_file, args.model, args.method)
    existing_unsupported = next(
        (
            row
            for row in existing_rows.values()
            if row.get("model") == args.model
            and row.get("method") == args.method
            and row.get("batch_size", args.batch_size) <= args.batch_size
            and row.get("status") == "unsupported_batch"
        ),
        None,
    )
    if existing_unsupported is not None:
        reason = (
            f"此前 batch={args.batch_size}、input={existing_unsupported.get('input_length')} "
            f"检测到不支持多 batch：{existing_unsupported.get('error', '')}"
        )
        missing_lengths = [
            input_length
            for input_length in args.input_lengths
            if cell_key(args.model, args.method, args.batch_size, input_length) not in existing_rows
        ]
        mark_unsupported_batch_cells(args, missing_lengths, reason)
        return

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
        for input_index, input_length in enumerate(args.input_lengths):
            current_rows = read_history_state(history_file, args.model, args.method)
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
                record_status(history_file, row)
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
            record_status(history_file, row)

            if row.get("status") == "unsupported_batch":
                reason = (
                    f"batch={args.batch_size}、input={input_length} 检测到方法不支持多 batch："
                    f"{row.get('error', '')}"
                )
                mark_unsupported_batch_cells(
                    args,
                    args.input_lengths[input_index + 1:],
                    reason,
                )
                break

    finally:
        if options._cleanup is not None:
            options._cleanup()
        del model
        del tokenizer
        clear_cuda_state(devices)


if __name__ == "__main__":
    main()
