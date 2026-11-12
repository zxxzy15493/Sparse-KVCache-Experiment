#!/usr/bin/env python3
"""按模型、方法、batch、输入长度和 budget 执行单卡效率/显存矩阵。"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
METHOD_WORKER = SCRIPT_DIR / "run_method_batch.py"
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common import (
    EFFICIENCY_METHODS,
    METHODS,
    MODEL_CHOICES,
    model_alias as get_model_alias,
)
from benchmarks.efficency_vram.tools.reporting import (
    append_history_event,
    cell_key,
    format_number,
    method_history_path,
    read_history_state,
    save_method_averages,
)
from benchmarks.efficency_vram.tools.runtime import (
    dominated_failure,
    error_from_log,
    log_contains_oom,
    threshold_failures,
)

DEFAULT_MODELS = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct-1M",
)
DEFAULT_METHODS = EFFICIENCY_METHODS
DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16)
DEFAULT_INPUT_LENGTHS = (
    4096,
    8192,
    16384,
    32768,
    49152,
    65536,
    81920,
    98304,
    114688,
    131072,
)


def data_file_for_model(data_file: Path, model: str) -> Path:
    model_name = get_model_alias(model)
    path = Path(str(data_file).format(model=model_name, model_alias=model_name))
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--input-lengths", nargs="+", type=int, default=list(DEFAULT_INPUT_LENGTHS))
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--experiment", default="efficiency_vram_matrix")
    parser.add_argument("--task", default="niah_single_1")
    parser.add_argument("--budgets", nargs="+", type=int)
    parser.add_argument("--budget", type=int,
                        help="单个 budget 的兼容参数；新脚本请使用 --budgets。")
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measured-rounds", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--efficiency-backend", action="store_true",
                        help="用 Quest/ClusterKV 自带的效率实现（仅效率脚本）。")
    parser.add_argument("--serial-prefill", choices=("auto", "on", "off"), default="auto",
                        help="batch 测试时逐条串行 prefill（decode 仍按 batch）："
                             "auto 只对已知安全的方法启用，on 额外启用需合并模块状态的方法，off 保持批量 prefill。")
    parser.add_argument("--serial-prefill-verify", action="store_true",
                        help="每个长度首次测量时用批量 prefill 做对照，核对串行 prefill 的 token 是否一致。")
    parser.add_argument(
        "--export-csv-only",
        action="store_true",
        help="从各方法 history.csv 重新生成 average.csv，不运行测试。",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    unknown_methods = sorted(set(args.methods) - set(METHODS))
    unknown_models = sorted(set(args.models) - set(MODEL_CHOICES))
    if unknown_methods:
        raise ValueError(f"未知方法：{', '.join(unknown_methods)}")
    if unknown_models:
        raise ValueError(f"未知模型：{', '.join(unknown_models)}")
    if "{model}" not in str(args.data_file) and "{model_alias}" not in str(args.data_file):
        data_file = data_file_for_model(args.data_file, args.models[0])
        if not data_file.is_file():
            raise FileNotFoundError(f"RULER 输入文件不存在：{data_file}")
    if not args.budgets or any(budget < 1 for budget in args.budgets):
        raise ValueError("所有 budget 必须为正数。")
    if args.output_length < 2:
        raise ValueError("输出长度至少为 2，才能同时测量 TTFT 和 TPOT。")
    if args.warmup < 0 or args.measured_rounds < 1:
        raise ValueError("warmup 不能小于 0，measured-rounds 必须至少为 1。")
    if args.timeout_seconds < 1:
        raise ValueError("timeout-seconds 必须为正数。")
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        raise ValueError("所有 batch-size 必须为正数。")
    if any(input_length < 1 for input_length in args.input_lengths):
        raise ValueError("所有 input-length 必须为正数。")


def count_jsonl_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def worker_command(
    args: argparse.Namespace,
    model: str,
    method: str,
    batch_size: int,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(METHOD_WORKER),
        "--model",
        model,
        "--method",
        method,
        "--batch-size",
        str(batch_size),
        "--input-lengths",
        *[str(length) for length in args.input_lengths],
        "--data-file",
        str(data_file_for_model(args.data_file, model)),
        "--output-root",
        str(args.output_root),
        "--state-dir",
        str(args.state_dir),
        "--experiment",
        args.experiment,
        "--task",
        args.task,
        "--budget",
        str(args.budget),
        "--output-length",
        str(args.output_length),
        "--warmup",
        str(args.warmup),
        "--measured-rounds",
        str(args.measured_rounds),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if args.efficiency_backend:
        command.append("--efficiency-backend")
    if args.serial_prefill != "auto":
        command.extend(["--serial-prefill", args.serial_prefill])
    if args.serial_prefill_verify:
        command.append("--serial-prefill-verify")
    return command


def history_file_mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return 0


def read_current_cell(path: Path) -> dict | None:
    if not path.is_file():
        return None

    values = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value

        for name in ("budget", "batch_size", "input_length"):
            if name in values:
                values[name] = int(values[name])
        if "started_monotonic" in values:
            values["started_monotonic"] = float(values["started_monotonic"])
    except (OSError, ValueError):
        return None

    return values


def append_timeout_and_pruned_rows(
    args: argparse.Namespace,
    model: str,
    method: str,
    model_alias: str,
    batch_size: int,
    input_length: int,
    elapsed_s: float,
    history_file: Path,
) -> dict[str, dict]:
    rows = read_history_state(history_file, model, method, args.budget)
    failed_key = cell_key(model, method, args.budget, batch_size, input_length)
    if failed_key not in rows:
        timeout_row = {
            "model": model,
            "model_alias": model_alias,
            "method": method,
            "budget": args.budget,
            "batch_size": batch_size,
            "input_length": input_length,
            "output_length": args.output_length,
            "status": "timeout",
            "phase": "generation",
            "elapsed_s": round(elapsed_s, 2),
            "error": f"单组合超过 {args.timeout_seconds} 秒；由父进程 watchdog 终止 worker。",
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        append_history_event(history_file, timeout_row)
        rows[failed_key] = timeout_row

    for later_length in args.input_lengths:
        if later_length < input_length:
            continue
        key = cell_key(model, method, args.budget, batch_size, later_length)
        if key in rows:
            continue
        skipped_row = {
            "model": model,
            "model_alias": model_alias,
            "method": method,
            "budget": args.budget,
            "batch_size": batch_size,
            "input_length": later_length,
            "output_length": args.output_length,
            "status": "skipped_timeout_threshold",
            "phase": "not_started",
            "skip_reason": f"timeout at batch={batch_size}, input={input_length}",
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        append_history_event(history_file, skipped_row)
        rows[key] = skipped_row
    return rows


def append_oom_and_pruned_rows(
    args: argparse.Namespace,
    model: str,
    method: str,
    model_alias: str,
    batch_size: int,
    input_length: int,
    error: str,
    history_file: Path,
    phase: str = "generation",
    return_code: int | None = None,
) -> dict[str, dict]:
    rows = read_history_state(history_file, model, method, args.budget)
    failed_key = cell_key(model, method, args.budget, batch_size, input_length)
    existing_failure = rows.get(failed_key, {})

    if existing_failure.get("status") not in {"oom", "passed", "timeout"}:
        oom_row = {
            "model": model,
            "model_alias": model_alias,
            "method": method,
            "budget": args.budget,
            "batch_size": batch_size,
            "input_length": input_length,
            "output_length": args.output_length,
            "status": "oom",
            "phase": phase,
            "error": error,
            "started_at": existing_failure.get(
                "started_at",
                datetime.now().astimezone().isoformat(timespec="seconds"),
            ),
        }
        if return_code is not None:
            oom_row["return_code"] = return_code
        append_history_event(history_file, oom_row)
        rows[failed_key] = oom_row

    for later_length in args.input_lengths:
        if later_length < input_length:
            continue
        key = cell_key(model, method, args.budget, batch_size, later_length)
        if key == failed_key:
            continue

        existing_row = rows.get(key)
        if existing_row is not None and existing_row.get("status") in {
            "passed",
            "oom",
            "timeout",
        }:
            continue

        skipped_row = {
            "model": model,
            "model_alias": model_alias,
            "method": method,
            "budget": args.budget,
            "batch_size": batch_size,
            "input_length": later_length,
            "output_length": args.output_length,
            "status": "skipped_oom_threshold",
            "phase": "not_started",
            "skip_reason": f"oom at batch={batch_size}, input={input_length}",
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        append_history_event(history_file, skipped_row)
        rows[key] = skipped_row

    return rows


def append_worker_exit_rows(
    args: argparse.Namespace,
    model: str,
    method: str,
    model_alias: str,
    batch_size: int,
    return_code: int | None,
    history_file: Path,
) -> dict[str, dict]:
    rows = read_history_state(history_file, model, method, args.budget)
    for input_length in args.input_lengths:
        key = cell_key(model, method, args.budget, batch_size, input_length)
        if key in rows:
            continue
        row = {
            "model": model,
            "model_alias": model_alias,
            "method": method,
            "budget": args.budget,
            "batch_size": batch_size,
            "input_length": input_length,
            "output_length": args.output_length,
            "status": "failed",
            "phase": "worker",
            "return_code": return_code,
            "error": "batch worker 非预期退出；详见 worker 日志。",
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        append_history_event(history_file, row)
        rows[key] = row
    return rows


def append_unsupported_batch_rows(
    args: argparse.Namespace,
    model: str,
    method: str,
    model_alias: str,
    batch_size: int,
    reason: str,
    history_file: Path,
) -> dict[str, dict]:
    rows = read_history_state(history_file, model, method, args.budget)
    for input_length in args.input_lengths:
        key = cell_key(model, method, args.budget, batch_size, input_length)
        if key in rows:
            continue

        row = {
            "model": model,
            "model_alias": model_alias,
            "method": method,
            "budget": args.budget,
            "batch_size": batch_size,
            "input_length": input_length,
            "output_length": args.output_length,
            "status": "skipped_unsupported_batch",
            "phase": "not_started",
            "skip_reason": reason,
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        append_history_event(history_file, row)
        rows[key] = row
    return rows


def run_batch_worker(
    args: argparse.Namespace,
    model: str,
    model_alias: str,
    method: str,
    batch_size: int,
    history_file: Path,
) -> dict[str, dict]:
    log_path = (
        args.state_dir
        / "logs"
        / model_alias
        / method
        / (
            f"budget_{args.budget}_batch_{batch_size}_"
            f"input_{args.input_lengths[0]}_worker.log"
        )
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    current_cell_path = args.state_dir / "current_cell.txt"
    try:
        current_cell_path.unlink()
    except FileNotFoundError:
        pass

    command = worker_command(args, model, method, batch_size)
    worker_started = time.monotonic()
    external_timeout = False
    timed_out_cell = None
    last_history_mtime = history_file_mtime(history_file)
    last_progress = worker_started
    last_heartbeat = worker_started

    print(
        f"启动 worker：{model_alias}/{method} k{args.budget} b{batch_size} "
        f"i{args.input_lengths[0]}；日志：{log_path}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                time.sleep(2)
                now = time.monotonic()
                current = read_current_cell(current_cell_path)

                current_mtime = history_file_mtime(history_file)
                if current_mtime != last_history_mtime:
                    rows = read_history_state(history_file, model, method, args.budget)
                    last_history_mtime = current_mtime
                    last_progress = now
                    latest = next(reversed(rows.values()), None) if rows else None
                    if latest is not None:
                        print(
                            f"完成 {model_alias}/{method} k{args.budget} b{batch_size} "
                            f"i{latest.get('input_length')}: {latest.get('status')} "
                            f"ttft_ms={format_number(latest.get('ttft_ms'))} "
                            f"tpot_ms={format_number(latest.get('tpot_ms'))} "
                            f"peak_mib={format_number(latest.get('peak_allocated_mib'))}",
                            flush=True,
                        )

                if current is not None:
                    cell_elapsed = now - current.get("started_monotonic", now)
                    if cell_elapsed >= args.timeout_seconds + 2:
                        external_timeout = True
                        timed_out_cell = current
                        stop_process_group(process)
                        break
                elif now - last_progress >= args.timeout_seconds + 2:
                    # 初始化卡住时，将该 batch 最早的未完成点记为超时。
                    rows = read_history_state(history_file, model, method, args.budget)
                    pending_lengths = [
                        length
                        for length in args.input_lengths
                        if cell_key(model, method, args.budget, batch_size, length) not in rows
                    ]
                    if pending_lengths:
                        timed_out_cell = {
                            "model": model,
                            "method": method,
                            "budget": args.budget,
                            "batch_size": batch_size,
                            "input_length": pending_lengths[0],
                            "started_monotonic": last_progress,
                        }
                        external_timeout = True
                        stop_process_group(process)
                        break

                if now - last_heartbeat >= 60:
                    if current is None:
                        detail = "模型初始化或组合切换仍在运行"
                    else:
                        detail = (
                            f"当前 i{current.get('input_length')} 已运行 "
                            f"{cell_elapsed:.0f}s"
                        )
                    print(
                        f"等待 {model_alias}/{method} k{args.budget} b{batch_size}：{detail}",
                        flush=True,
                    )
                    last_heartbeat = now
        except BaseException:
            stop_process_group(process)
            raise

        if process.poll() is None:
            stop_process_group(process)
        return_code = process.wait()

    if external_timeout and timed_out_cell is not None:
        args.budget = timed_out_cell["budget"]
        timeout_elapsed = time.monotonic() - timed_out_cell.get(
            "started_monotonic",
            worker_started,
        )
        rows = append_timeout_and_pruned_rows(
            args,
            model,
            method,
            model_alias,
            batch_size,
            timed_out_cell["input_length"],
            timeout_elapsed,
            history_file,
        )
        try:
            current_cell_path.unlink()
        except FileNotFoundError:
            pass
    elif return_code != 0:
        current_cell = read_current_cell(current_cell_path)

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        oom_from_log = log_contains_oom(log_text)
        oom_kill_exit = return_code in {-9, -137, 137}

        if oom_from_log or oom_kill_exit:
            rows_before_worker = read_history_state(history_file, model, method, args.budget)
            if current_cell is not None:
                current_key = cell_key(
                    model,
                    method,
                    args.budget,
                    batch_size,
                    current_cell["input_length"],
                )
                completed_row = rows_before_worker.get(current_key)
                if completed_row is not None and completed_row.get("status") in {
                    "passed",
                    "oom",
                    "timeout",
                }:
                    current_cell = None

            pending_lengths = [
                length
                for length in args.input_lengths
                if cell_key(model, method, args.budget, batch_size, length) not in rows_before_worker
            ]

            if current_cell is None and pending_lengths:
                current_cell = {
                    "budget": args.budget,
                    "batch_size": batch_size,
                    "input_length": pending_lengths[0],
                    "started_monotonic": worker_started,
                }

            if current_cell is not None:
                args.budget = current_cell.get("budget", args.budget)
                if oom_from_log:
                    oom_error = f"worker 日志包含 OOM：{error_from_log(log_text)}"
                else:
                    oom_error = (
                        f"worker 以 return code {return_code} 被 SIGKILL 终止；"
                        "按宿主内存 OOM 处理。详见 worker 日志。"
                    )

                rows = append_oom_and_pruned_rows(
                    args,
                    model,
                    method,
                    model_alias,
                    batch_size,
                    current_cell["input_length"],
                    oom_error,
                    history_file,
                    phase="worker",
                    return_code=return_code,
                )
            else:
                rows = append_worker_exit_rows(
                    args,
                    model,
                    method,
                    model_alias,
                    batch_size,
                    return_code,
                    history_file,
                )
        else:
            rows = append_worker_exit_rows(
                args,
                model,
                method,
                model_alias,
                batch_size,
                return_code,
                history_file,
            )

        try:
            current_cell_path.unlink()
        except FileNotFoundError:
            pass
    else:
        rows = read_history_state(history_file, model, method, args.budget)

    return rows


def main() -> None:
    args = parse_args()
    if args.budgets is None:
        args.budgets = [args.budget if args.budget is not None else 1024]
    elif args.budget is not None:
        raise ValueError("--budgets 和 --budget 不能同时使用。")
    args.budget = args.budgets[0]
    if args.state_dir is None:
        args.state_dir = args.output_root
    args.data_file = args.data_file.resolve()
    args.output_root = args.output_root.resolve()
    args.state_dir = args.state_dir.resolve()
    validate_args(args)

    maximum_samples = max(args.batch_sizes) * args.measured_rounds
    data_files = {}
    source_rows = {}
    if not args.export_csv_only:
        for model in args.models:
            data_file = data_file_for_model(args.data_file, model)
            if not data_file.is_file():
                raise FileNotFoundError(f"RULER 输入文件不存在：{data_file}")
            row_count = count_jsonl_rows(data_file)
            if row_count < maximum_samples:
                raise ValueError(
                    f"RULER 文件只有 {row_count} 条输入：{data_file}；"
                    f"最大 batch 需要 {maximum_samples} 条，以保证每组有指定数量的完整测量轮次。"
                )
            data_files[model] = data_file
            source_rows[model] = row_count

    configuration = {
        "models": args.models,
        "methods": args.methods,
        "batch_sizes": args.batch_sizes,
        "input_lengths": args.input_lengths,
        "data_file_template": str(args.data_file),
        "data_files": {model: str(path) for model, path in data_files.items()},
        "source_rows": source_rows,
        "output_root": str(args.output_root),
        "experiment": args.experiment,
        "task": args.task,
        "budgets": args.budgets,
        "output_length": args.output_length,
        "warmup": args.warmup,
        "measured_rounds": args.measured_rounds,
        "timeout_seconds": args.timeout_seconds,
        "efficiency_backend": args.efficiency_backend,
        "serial_prefill": args.serial_prefill,
        "serial_prefill_verify": args.serial_prefill_verify,
        "python": sys.executable,
    }
    total = (
        len(args.models)
        * len(args.methods)
        * len(args.budgets)
        * len(args.batch_sizes)
        * len(args.input_lengths)
    )
    if args.dry_run:
        print(json.dumps({"planned_combinations_before_pruning": total, **configuration}, ensure_ascii=False, indent=2))
        return

    args.state_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.export_csv_only:
        save_method_averages(args)
        print(
            f"只导出 CSV，未启动模型：{args.output_root}",
            flush=True,
        )
        return

    save_method_averages(args)

    planned_position = 0
    batch_positions = {
        batch_size: position
        for position, batch_size in enumerate(args.batch_sizes)
    }
    input_lengths = args.input_lengths
    for model in args.models:
        model_alias = get_model_alias(model)
        for method in args.methods:
            history_file = method_history_path(
                args.output_root,
                model,
                method,
            )
            for batch_size in args.batch_sizes:
                args.input_lengths = input_lengths
                current_batch_position = batch_positions[batch_size]
                for budget in args.budgets:
                    args.budget = budget
                    pending_lengths = []

                    for input_length in input_lengths:
                        rows = read_history_state(history_file, model, method, budget)
                        planned_position += 1
                        key = cell_key(model, method, budget, batch_size, input_length)

                        if key in rows:
                            print(
                                f"[{planned_position}/{total}] RESUME {model_alias}/{method} "
                                f"b{batch_size} i{input_length} k{budget}: {rows[key].get('status')}",
                                flush=True,
                            )
                            continue

                        unsupported_sources = [
                            row
                            for row in rows.values()
                            if row.get("model") == model
                            and row.get("method") == method
                            and row.get("status") == "unsupported_batch"
                            and (
                                batch_positions.get(
                                    row.get("batch_size"),
                                    current_batch_position + 1,
                                )
                                <= current_batch_position
                                or (
                                    row.get("batch_size") not in batch_positions
                                    and row.get("batch_size", 1) < batch_size
                                )
                            )
                        ]
                        if unsupported_sources:
                            source = unsupported_sources[-1]
                            reason = (
                                f"budget={budget}、batch={source.get('batch_size')}、"
                                f"input={source.get('input_length')} 检测到方法不支持多 batch："
                                f"{source.get('error', '')}"
                            )
                            row = {
                                "model": model,
                                "model_alias": model_alias,
                                "method": method,
                                "budget": budget,
                                "batch_size": batch_size,
                                "input_length": input_length,
                                "output_length": args.output_length,
                                "status": "skipped_unsupported_batch",
                                "phase": "not_started",
                                "skip_reason": reason,
                                "started_at": datetime.now().astimezone().isoformat(
                                    timespec="seconds"
                                ),
                            }
                            append_history_event(history_file, row)
                            print(
                                f"[{planned_position}/{total}] SKIP {model_alias}/{method} "
                                f"b{batch_size} i{input_length} k{budget}: {reason}",
                                flush=True,
                            )
                            continue

                        setup_failures = [
                            row
                            for row in rows.values()
                            if row.get("model") == model
                            and row.get("method") == method
                            and row.get("setup_failure")
                        ]
                        if setup_failures:
                            source = setup_failures[-1]
                            row = {
                                "model": model,
                                "model_alias": model_alias,
                                "method": method,
                                "budget": budget,
                                "batch_size": batch_size,
                                "input_length": input_length,
                                "output_length": args.output_length,
                                "status": "skipped_setup_failure",
                                "phase": "loading",
                                "skip_reason": source.get("error"),
                                "started_at": datetime.now().astimezone().isoformat(
                                    timespec="seconds"
                                ),
                            }
                            append_history_event(history_file, row)
                            print(
                                f"[{planned_position}/{total}] SKIP {model_alias}/{method} "
                                f"b{batch_size} i{input_length} k{budget}: prior loader failure",
                                flush=True,
                            )
                            continue

                        failure = dominated_failure(
                            threshold_failures(rows, model, method),
                            batch_size,
                            input_length,
                        )
                        if failure is not None:
                            row = {
                                "model": model,
                                "model_alias": model_alias,
                                "method": method,
                                "budget": budget,
                                "batch_size": batch_size,
                                "input_length": input_length,
                                "output_length": args.output_length,
                                "status": f"skipped_{failure['status']}_threshold",
                                "phase": "not_started",
                                "skip_reason": (
                                    f"{failure['status']} at batch={failure['batch_size']}, "
                                    f"input={failure['input_length']}"
                                ),
                                "started_at": datetime.now().astimezone().isoformat(
                                    timespec="seconds"
                                ),
                            }
                            append_history_event(history_file, row)
                            print(
                                f"[{planned_position}/{total}] SKIP {model_alias}/{method} "
                                f"b{batch_size} i{input_length} k{budget}: "
                                f"dominated by {failure['status']} at "
                                f"b{failure['batch_size']} i{failure['input_length']}",
                                flush=True,
                            )
                            continue

                        pending_lengths.append((planned_position, input_length))

                    if not pending_lengths:
                        continue

                    print(
                        f"[worker] {model_alias}/{method} b{batch_size} k{budget} 将加载一次模型，"
                        f"待测长度={[length for _, length in pending_lengths]}",
                        flush=True,
                    )
                    rows = run_batch_worker(
                        args,
                        model,
                        model_alias,
                        method,
                        batch_size,
                        history_file,
                    )
                    for position, input_length in pending_lengths:
                        key = cell_key(model, method, budget, batch_size, input_length)
                        row = rows.get(key, {})
                        print(
                            f"[{position}/{total}] "
                            f"{row.get('status', 'UNKNOWN').upper()} "
                            f"{model_alias}/{method} b{batch_size} i{input_length} k{budget} "
                            f"ttft_ms={format_number(row.get('ttft_ms'))} "
                            f"tpot_ms={format_number(row.get('tpot_ms'))} "
                            f"peak_mib={format_number(row.get('peak_allocated_mib'))} "
                            f"{row.get('error', row.get('skip_reason', ''))}",
                            flush=True,
                        )

    print("矩阵运行结束。", flush=True)


if __name__ == "__main__":
    main()
