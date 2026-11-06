#!/usr/bin/env python3
"""按模型、方法、batch 和输入长度执行单卡效率/显存矩阵。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
METHOD_WORKER = SCRIPT_DIR / "run_method_batch.py"

DEFAULT_MODELS = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct-1M",
)
DEFAULT_METHODS = (
    "full_attention",
    "h2o",
    "keyformer",
    "snapkv",
    "streaming",
    "quest",
    "sparq",
    "xattention",
    "flexprefill",
    "retroinfer",
    "headkv",
    "adakv",
    "cakekv",
    "pyramidkv",
)
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
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measured-rounds", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument(
        "--export-csv-only",
        action="store_true",
        help="从已有状态文件重新生成各方法 CSV，不运行任何测试。",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_benchmark_choices() -> tuple[set[str], set[str], dict[str, str]]:
    sys.path.insert(0, str(REPO_ROOT))

    from benchmarks.common import METHODS, MODEL_CHOICES, model_alias

    return set(METHODS), set(MODEL_CHOICES), {model: model_alias(model) for model in MODEL_CHOICES}


def validate_args(args: argparse.Namespace) -> None:
    methods, models, _ = load_benchmark_choices()
    unknown_methods = sorted(set(args.methods) - methods)
    unknown_models = sorted(set(args.models) - models)
    if unknown_methods:
        raise ValueError(f"未知方法：{', '.join(unknown_methods)}")
    if unknown_models:
        raise ValueError(f"未知模型：{', '.join(unknown_models)}")
    if not args.data_file.is_file():
        raise FileNotFoundError(f"RULER 输入文件不存在：{args.data_file}")
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


def cell_key(model: str, method: str, batch_size: int, input_length: int) -> str:
    return f"{model}\t{method}\t{batch_size}\t{input_length}"


def read_existing_status(path: Path) -> dict[str, dict]:
    results = {}
    if not path.is_file():
        return results

    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("model"):
                continue
            row = decode_status_row(row)
            key = cell_key(row["model"], row["method"], row["batch_size"], row["input_length"])
            results[key] = row
    return results


STATUS_FIELDS = [
    "model",
    "model_alias",
    "method",
    "batch_size",
    "input_length",
    "output_length",
    "status",
    "phase",
    "started_at",
    "elapsed_s",
    "return_code",
    "setup_failure",
    "backend",
    "measured_rounds",
    "measured_samples",
    "ttft_ms",
    "tpot_ms",
    "generation_latency_s",
    "output_throughput_tokens_per_s",
    "decode_throughput_tokens_per_s",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "incremental_peak_allocated_mib",
    "decode_peak_allocated_mib",
    "decode_peak_reserved_mib",
    "error",
    "skip_reason",
    "log_path",
    "command",
    "pred_json_path",
    "pred_result_id",
    "summary_path",
]

STATUS_INTEGER_FIELDS = {
    "batch_size",
    "input_length",
    "output_length",
    "return_code",
    "measured_rounds",
    "measured_samples",
}
STATUS_FLOAT_FIELDS = {
    "elapsed_s",
    "ttft_ms",
    "tpot_ms",
    "generation_latency_s",
    "output_throughput_tokens_per_s",
    "decode_throughput_tokens_per_s",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "incremental_peak_allocated_mib",
    "decode_peak_allocated_mib",
    "decode_peak_reserved_mib",
}


def decode_status_row(row: dict) -> dict:
    decoded = dict(row)
    for field in STATUS_INTEGER_FIELDS:
        value = decoded.get(field)
        decoded[field] = int(value) if value not in (None, "") else None
    for field in STATUS_FLOAT_FIELDS:
        value = decoded.get(field)
        decoded[field] = float(value) if value not in (None, "") else None

    setup_failure = decoded.get("setup_failure")
    if setup_failure not in (None, ""):
        decoded["setup_failure"] = setup_failure.lower() == "true"
    if decoded.get("command"):
        try:
            decoded["command"] = json.loads(decoded["command"])
        except json.JSONDecodeError:
            pass
    return decoded


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


def log_contains_oom(log_text: str) -> bool:
    lowered = log_text.lower()
    oom_markers = (
        "out of memory",
        "cuda_error_out_of_memory",
        "cublas_status_alloc_failed",
        "cuda error: memory allocation",
        "cuda error: out of memory",
    )
    return any(marker in lowered for marker in oom_markers)


def error_from_log(log_text: str) -> str:
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    if not lines:
        return "子进程没有输出。"
    markers = (
        "outofmemoryerror",
        "out of memory",
        "traceback",
        "runtimeerror",
        "valueerror",
        "assertionerror",
        "exception",
        "error:",
    )
    matching = [line for line in lines if any(marker in line.lower() for marker in markers)]
    selected = matching[-6:] if matching else lines[-6:]
    return "\n".join(selected)[-1800:]


def metrics_from_summary(
    summary: dict,
    batch_size: int,
    rounds: int,
    output_length: int,
    pred_json_path: Path,
) -> dict:
    groups = summary.get("groups", [])
    if len(groups) != 1:
        raise ValueError(f"应当只汇总一个 batch/长度组合，实际 groups={len(groups)}")
    group = groups[0]
    if group.get("num_batches") != rounds or group.get("num_samples") != batch_size * rounds:
        raise ValueError(
            f"测量批次数不符：batches={group.get('num_batches')}，"
            f"samples={group.get('num_samples')}，预期 {rounds} 批、{batch_size * rounds} 条。"
        )

    mean_decode_latency = group.get("mean_decode_latency_s")
    decode_throughput = None
    if mean_decode_latency:
        decode_throughput = batch_size * (output_length - 1) / mean_decode_latency

    mean_generation_latency = group.get("mean_generation_latency_s")
    return {
        "backend": group.get("backend"),
        "measured_rounds": group.get("num_batches"),
        "measured_samples": group.get("num_samples"),
        "ttft_ms": group.get("mean_ttft_s") * 1000 if group.get("mean_ttft_s") is not None else None,
        "tpot_ms": group.get("mean_tpot_ms"),
        "generation_latency_s": mean_generation_latency,
        "output_throughput_tokens_per_s": group.get("output_throughput_tokens_per_s"),
        "decode_throughput_tokens_per_s": decode_throughput,
        "peak_allocated_mib": group.get("peak_allocated_mib"),
        "peak_reserved_mib": group.get("peak_reserved_mib"),
        "decode_peak_allocated_mib": group.get("mean_decode_peak_allocated_mib"),
        "decode_peak_reserved_mib": group.get("mean_decode_peak_reserved_mib"),
        "incremental_peak_allocated_mib": (
            group.get("memory_by_device", {}).get("cuda:0", {}).get("incremental_peak_allocated_mib")
        ),
        "pred_json_path": str(pred_json_path),
        "summary_path": str(pred_json_path),
    }


def threshold_failures(rows: dict[str, dict], model: str, method: str) -> list[dict]:
    return [
        row
        for row in rows.values()
        if row.get("model") == model
        and row.get("method") == method
        and row.get("status") in {"oom", "timeout"}
    ]


def dominated_failure(
    failures: list[dict],
    batch_size: int,
    input_length: int,
) -> dict | None:
    for failure in failures:
        if batch_size >= failure["batch_size"] and input_length >= failure["input_length"]:
            return failure
    return None


def append_status(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.is_file() or path.stat().st_size == 0
    row_to_write = dict(row)
    if isinstance(row_to_write.get("command"), (list, dict)):
        row_to_write["command"] = json.dumps(row_to_write["command"], ensure_ascii=False)

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row_to_write)
        handle.flush()


def ordered_rows(rows: dict[str, dict], args: argparse.Namespace) -> list[dict]:
    model_order = {name: position for position, name in enumerate(args.models)}
    method_order = {name: position for position, name in enumerate(args.methods)}
    batch_order = {name: position for position, name in enumerate(args.batch_sizes)}
    length_order = {name: position for position, name in enumerate(args.input_lengths)}
    return sorted(
        rows.values(),
        key=lambda row: (
            model_order.get(row["model"], len(model_order)),
            method_order.get(row["method"], len(method_order)),
            batch_order.get(row["batch_size"], len(batch_order)),
            length_order.get(row["input_length"], len(length_order)),
        ),
    )


def save_reports(
    args: argparse.Namespace,
    rows: dict[str, dict],
    configuration: dict,
    finished: bool,
) -> None:
    report_rows = ordered_rows(rows, args)
    save_method_csvs(args, report_rows)
    counts = Counter(row.get("status", "unknown") for row in report_rows)
    csv_fields = [
        "model",
        "model_alias",
        "method",
        "batch_size",
        "input_length",
        "output_length",
        "status",
        "phase",
        "measured_rounds",
        "measured_samples",
        "backend",
        "ttft_ms",
        "tpot_ms",
        "generation_latency_s",
        "output_throughput_tokens_per_s",
        "decode_throughput_tokens_per_s",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "incremental_peak_allocated_mib",
        "decode_peak_allocated_mib",
        "decode_peak_reserved_mib",
        "elapsed_s",
        "return_code",
        "error",
        "log_path",
        "summary_path",
    ]
    with (args.state_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report_rows)

    report_lines = [
        "# 长上下文效率与显存测试",
        "",
        f"更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"运行状态：{'完成' if finished else '进行中'}",
        f"已记录组合：{len(report_rows)}；状态计数：`{dict(counts)}`",
        "",
        "## 测试配置",
        "",
        f"- Hugging Face 模型 ID：`{', '.join(args.models)}`",
        f"- 方法：`{', '.join(args.methods)}`",
        f"- Batch size：`{', '.join(map(str, args.batch_sizes))}`",
        f"- 输入长度：`{', '.join(map(str, args.input_lengths))}` tokens",
        f"- 固定输出：{args.output_length} tokens；TTFT 计首 token，TPOT 统计后续 {args.output_length - 1} 个 decode step。",
        f"- Warmup：{args.warmup} 轮；测量：每组合 {args.measured_rounds} 个完整 batch。",
        f"- KV budget：{args.budget}；单组合超时：{args.timeout_seconds} 秒。",
        f"- 设备：物理 GPU {args.physical_gpu}（进程内 `{args.device}`）。",
        f"- RULER 输入：`{args.data_file}`。输入会由 benchmark tokenizer 调整为表中精确长度；短输入按原入口的 repeat 规则补足，长输入从中间截断。",
        "- 预测、逐轮指标、运行状态、CSV 汇总、报告和日志都保存在本次输出目录。",
        "",
        "## 各模型/方法状态",
        "",
        "| 模型 | 方法 | Passed | OOM | Timeout | Failed | Skipped | 最大成功 batch/长度 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in report_rows:
        grouped[(row["model"], row["method"])].append(row)
    for model in args.models:
        for method in args.methods:
            group = grouped.get((model, method), [])
            group_counts = Counter(row.get("status", "unknown") for row in group)
            passed_rows = [row for row in group if row.get("status") == "passed"]
            if passed_rows:
                largest_context_row = max(
                    passed_rows,
                    key=lambda row: (row["input_length"], row["batch_size"]),
                )
                maximum = (
                    f"b{largest_context_row['batch_size']} / "
                    f"{largest_context_row['input_length']} tokens"
                )
            else:
                maximum = "—"
            skipped_count = sum(
                count
                for status, count in group_counts.items()
                if status.startswith("skipped")
            )
            report_lines.append(
                f"| {model} | {method} | {group_counts.get('passed', 0)} | "
                f"{group_counts.get('oom', 0)} | {group_counts.get('timeout', 0)} | "
                f"{group_counts.get('failed', 0)} | {skipped_count} | {maximum} |"
            )

    report_lines.extend([
        "",
        "OOM 或超时后，后续只有在 batch size 和输入长度都大于等于失败点时才跳过；其他组合继续执行。",
        "TTFT、TPOT 和 throughput 沿用 `benchmarks/efficency_vram/` 的计时器口径；显存记录 PyTorch allocator 峰值。",
        "RetroInfer 使用原生 generate，没有统一逐 token observer，因此 TTFT/TPOT 为空。",
        "每个模型/方法的 `history.csv` 和 `average.csv` 位于对应测试输出目录。",
        "逐组合结果位于输出根目录唯一的 `pred.json`；运行状态保存在 `status.csv`，汇总表为 `summary.csv`。每次子进程日志位于 `logs/`。",
        "",
    ])
    (args.state_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


METHOD_CSV_FIELDS = [
    "record_type",
    "round_index",
    "method",
    "batch_size",
    "input_length",
    "status",
    "phase",
    "ttft_ms",
    "tpot_ms",
    "generation_latency_s",
    "output_throughput_tokens_per_s",
    "decode_throughput_tokens_per_s",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "incremental_peak_allocated_mib",
    "decode_peak_allocated_mib",
    "decode_peak_reserved_mib",
    "elapsed_s",
    "started_at",
    "return_code",
    "error",
    "skip_reason",
    "log_path",
    "summary_path",
]

AVERAGE_CSV_FIELDS = [
    "method",
    "batch_size",
    "input_length",
    "latest_status",
    "average_ttft_ms",
    "average_tpot_ms",
    "average_generation_latency_s",
    "average_output_throughput_tokens_per_s",
    "average_decode_throughput_tokens_per_s",
    "average_peak_allocated_mib",
    "average_peak_reserved_mib",
    "average_decode_peak_allocated_mib",
    "average_decode_peak_reserved_mib",
    "latest_error",
    "latest_skip_reason",
]

HISTORY_EVENT_CACHE: dict[tuple, list[dict]] = {}
PRED_RESULT_CACHE: dict[tuple, dict[str, dict]] = {}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            rounded_row = {
                key: f"{value:.2f}" if isinstance(value, float) else value
                for key, value in row.items()
            }
            writer.writerow(rounded_row)


def read_status_events(path: Path, allowed_methods: set[str]) -> list[dict]:
    events = []
    if not path.is_file():
        return events

    with path.open(encoding="utf-8", newline="") as handle:
        for event_index, row in enumerate(csv.DictReader(handle), start=1):
            if row.get("method") not in allowed_methods:
                continue
            row = decode_status_row(row)
            events.append({"event_index": event_index, **row})

    return events


def pred_result_for_status(row: dict) -> dict | None:
    result_id = row.get("pred_result_id")
    pred_json_path = row.get("pred_json_path") or row.get("summary_path")
    if not result_id or not pred_json_path:
        return None

    path = Path(pred_json_path)
    if not path.is_file():
        return None

    cache_key = (str(path), path.stat().st_mtime_ns)
    results_by_id = PRED_RESULT_CACHE.get(cache_key)
    if results_by_id is None:
        document = json.loads(path.read_text(encoding="utf-8"))
        results_by_id = {
            result["result_id"]: result
            for result in document.get("results", [])
            if result.get("result_id")
        }
        PRED_RESULT_CACHE.clear()
        PRED_RESULT_CACHE[cache_key] = results_by_id

    return results_by_id.get(result_id)


def history_rows_for_event(event: dict) -> list[dict]:
    result = pred_result_for_status(event)
    result_id = event.get("pred_result_id")
    result_mtime = None
    pred_json_path = event.get("pred_json_path") or event.get("summary_path")
    if pred_json_path and Path(pred_json_path).is_file():
        result_mtime = Path(pred_json_path).stat().st_mtime_ns
    cache_key = (
        event.get("event_index"),
        event.get("status"),
        result_id,
        result_mtime,
    )
    cached_rows = HISTORY_EVENT_CACHE.get(cache_key)
    if cached_rows is not None:
        return cached_rows

    if event.get("status") != "passed":
        rows = [{"record_type": "status", **event}]
        HISTORY_EVENT_CACHE[cache_key] = rows
        return rows

    if result is None:
        rows = [{"record_type": "status", **event}]
        HISTORY_EVENT_CACHE[cache_key] = rows
        return rows

    measurement_rows = []
    for measurement in result.get("measurements", []):
        measurement_rows.append({
            "record_type": "measurement",
            "round_index": measurement.get("batch_index", 0) + 1,
            "method": event.get("method"),
            "batch_size": measurement.get("batch_size", event.get("batch_size")),
            "input_length": measurement.get("input_length", event.get("input_length")),
            "status": "passed",
            "phase": "complete",
            "ttft_ms": measurement.get("ttft_s") * 1000
            if measurement.get("ttft_s") is not None else None,
            "tpot_ms": measurement.get("tpot_ms"),
            "generation_latency_s": measurement.get("generation_latency_s"),
            "output_throughput_tokens_per_s": measurement.get(
                "output_throughput_tokens_per_s"
            ),
            "decode_throughput_tokens_per_s": measurement.get(
                "decode_throughput_tokens_per_s"
            ),
            "peak_allocated_mib": measurement.get("peak_allocated_mib"),
            "peak_reserved_mib": measurement.get("peak_reserved_mib"),
            "decode_peak_allocated_mib": measurement.get("decode_peak_allocated_mib"),
            "decode_peak_reserved_mib": measurement.get("decode_peak_reserved_mib"),
            "incremental_peak_allocated_mib": measurement.get(
                "memory_by_device", {}
            ).get("cuda:0", {}).get("incremental_peak_allocated_mib"),
            "elapsed_s": event.get("elapsed_s"),
            "started_at": event.get("started_at"),
            "return_code": event.get("return_code"),
            "error": event.get("error"),
            "skip_reason": event.get("skip_reason"),
            "log_path": event.get("log_path"),
            "summary_path": event.get("summary_path"),
        })

    rows = measurement_rows or [{"record_type": "status", **event}]
    HISTORY_EVENT_CACHE[cache_key] = rows
    return rows


def average_attempts(measurements: list[dict], latest: dict) -> dict:
    averaged_fields = {
        "ttft_ms": "average_ttft_ms",
        "tpot_ms": "average_tpot_ms",
        "generation_latency_s": "average_generation_latency_s",
        "output_throughput_tokens_per_s": "average_output_throughput_tokens_per_s",
        "decode_throughput_tokens_per_s": "average_decode_throughput_tokens_per_s",
        "peak_allocated_mib": "average_peak_allocated_mib",
        "peak_reserved_mib": "average_peak_reserved_mib",
        "decode_peak_allocated_mib": "average_decode_peak_allocated_mib",
        "decode_peak_reserved_mib": "average_decode_peak_reserved_mib",
    }
    result = {
        "method": latest.get("method"),
        "batch_size": latest.get("batch_size"),
        "input_length": latest.get("input_length"),
        "latest_status": latest.get("status"),
        "latest_error": latest.get("error"),
        "latest_skip_reason": latest.get("skip_reason"),
    }

    for source_field, output_field in averaged_fields.items():
        values = [row[source_field] for row in measurements if row.get(source_field) is not None]
        result[output_field] = statistics.mean(values) if values else None

    return result


def save_method_csvs(args: argparse.Namespace, latest_rows: list[dict]) -> None:
    """按模型和方法将历史、平均值 CSV 写入对应测试输出目录。"""
    allowed_methods = set(args.methods)
    history_events = read_status_events(args.state_dir / "status.csv", allowed_methods)
    measurements_by_key: dict[tuple, list[dict]] = defaultdict(list)

    for event in history_events:
        event_history_rows = history_rows_for_event(event)
        key = cell_key(
            event["model"],
            event["method"],
            event["batch_size"],
            event["input_length"],
        )
        measurements_by_key[key].extend(
            row for row in event_history_rows if row.get("record_type") == "measurement"
        )

    latest_by_key = {
        cell_key(row["model"], row["method"], row["batch_size"], row["input_length"]): row
        for row in latest_rows
        if row.get("method") in allowed_methods
    }
    model_alias_by_id = load_benchmark_choices()[2]

    for model in args.models:
        model_alias = model_alias_by_id[model]
        for method in args.methods:
            method_dir = (
                args.output_root
                / model_alias
                / args.experiment
                / method
                / args.task
            )
            method_events = [
                event
                for event in history_events
                if event.get("model") == model and event.get("method") == method
            ]
            history_rows = [
                history_row
                for event in method_events
                for history_row in history_rows_for_event(event)
            ]
            write_csv(method_dir / "history.csv", METHOD_CSV_FIELDS, history_rows)

            average_rows = []
            for key, latest in latest_by_key.items():
                if latest.get("model") != model or latest.get("method") != method:
                    continue
                measurements = measurements_by_key.get(key, [])
                average_rows.append(average_attempts(measurements, latest))

            write_csv(method_dir / "average.csv", AVERAGE_CSV_FIELDS, average_rows)


def worker_command(
    args: argparse.Namespace,
    model: str,
    method: str,
    batch_size: int,
) -> list[str]:
    return [
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
        str(args.data_file),
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
        "--device",
        args.device,
        "--physical-gpu",
        args.physical_gpu,
    ]


def status_file_mtime(path: Path) -> int:
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

        for name in ("batch_size", "input_length"):
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
    status_file: Path,
) -> dict[str, dict]:
    rows = read_existing_status(status_file)
    failed_key = cell_key(model, method, batch_size, input_length)
    if failed_key not in rows:
        timeout_row = {
            "model": model,
            "model_alias": model_alias,
            "method": method,
            "batch_size": batch_size,
            "input_length": input_length,
            "output_length": args.output_length,
            "status": "timeout",
            "phase": "generation",
            "elapsed_s": round(elapsed_s, 3),
            "error": f"单组合超过 {args.timeout_seconds} 秒；由父进程 watchdog 终止 worker。",
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        append_status(status_file, timeout_row)
        rows[failed_key] = timeout_row

    for later_length in args.input_lengths:
        if later_length < input_length:
            continue
        key = cell_key(model, method, batch_size, later_length)
        if key in rows:
            continue
        skipped_row = {
            "model": model,
            "model_alias": model_alias,
            "method": method,
            "batch_size": batch_size,
            "input_length": later_length,
            "output_length": args.output_length,
            "status": "skipped_timeout_threshold",
            "phase": "not_started",
            "skip_reason": f"timeout at batch={batch_size}, input={input_length}",
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        append_status(status_file, skipped_row)
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
    status_file: Path,
    phase: str = "generation",
    return_code: int | None = None,
) -> dict[str, dict]:
    rows = read_existing_status(status_file)
    failed_key = cell_key(model, method, batch_size, input_length)
    existing_failure = rows.get(failed_key, {})

    if existing_failure.get("status") not in {"oom", "passed", "timeout"}:
        oom_row = {
            "model": model,
            "model_alias": model_alias,
            "method": method,
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
        append_status(status_file, oom_row)
        rows[failed_key] = oom_row

    for later_length in args.input_lengths:
        if later_length < input_length:
            continue
        key = cell_key(model, method, batch_size, later_length)
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
            "batch_size": batch_size,
            "input_length": later_length,
            "output_length": args.output_length,
            "status": "skipped_oom_threshold",
            "phase": "not_started",
            "skip_reason": f"oom at batch={batch_size}, input={input_length}",
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        append_status(status_file, skipped_row)
        rows[key] = skipped_row

    return rows


def append_worker_exit_rows(
    args: argparse.Namespace,
    model: str,
    method: str,
    model_alias: str,
    batch_size: int,
    return_code: int | None,
    status_file: Path,
) -> dict[str, dict]:
    rows = read_existing_status(status_file)
    for input_length in args.input_lengths:
        key = cell_key(model, method, batch_size, input_length)
        if key in rows:
            continue
        row = {
            "model": model,
            "model_alias": model_alias,
            "method": method,
            "batch_size": batch_size,
            "input_length": input_length,
            "output_length": args.output_length,
            "status": "failed",
            "phase": "worker",
            "return_code": return_code,
            "error": "batch worker 非预期退出；详见 worker 日志。",
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        append_status(status_file, row)
        rows[key] = row
    return rows


def run_batch_worker(
    args: argparse.Namespace,
    configuration: dict,
    model: str,
    model_alias: str,
    method: str,
    batch_size: int,
    status_file: Path,
) -> dict[str, dict]:
    log_path = (
        args.state_dir
        / "logs"
        / model_alias
        / method
        / f"batch_{batch_size}_worker.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    current_cell_path = args.state_dir / "current_cell.txt"
    try:
        current_cell_path.unlink()
    except FileNotFoundError:
        pass

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.physical_gpu
    environment["PYTHONUNBUFFERED"] = "1"
    command = worker_command(args, model, method, batch_size)
    worker_started = time.monotonic()
    external_timeout = False
    timed_out_cell = None
    last_status_mtime = status_file_mtime(status_file)
    last_heartbeat = worker_started

    print(
        f"启动复用 worker：{model_alias}/{method} b{batch_size}；"
        f"模型每个 batch 只加载一次。日志：{log_path}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                time.sleep(2)
                now = time.monotonic()
                current = read_current_cell(current_cell_path)

                if current is not None:
                    cell_elapsed = now - current.get("started_monotonic", now)
                    if cell_elapsed >= args.timeout_seconds + 2:
                        external_timeout = True
                        timed_out_cell = current
                        stop_process_group(process)
                        break
                elif now - worker_started >= args.timeout_seconds + 2:
                    # 初始化卡住时，将该 batch 最早的未完成点记为超时。
                    rows = read_existing_status(status_file)
                    pending_lengths = [
                        length
                        for length in args.input_lengths
                        if cell_key(model, method, batch_size, length) not in rows
                    ]
                    if pending_lengths:
                        timed_out_cell = {
                            "model": model,
                            "method": method,
                            "batch_size": batch_size,
                            "input_length": pending_lengths[0],
                            "started_monotonic": worker_started,
                        }
                        external_timeout = True
                        stop_process_group(process)
                        break

                current_mtime = status_file_mtime(status_file)
                if current_mtime != last_status_mtime:
                    rows = read_existing_status(status_file)
                    save_reports(args, rows, configuration, finished=False)
                    last_status_mtime = current_mtime
                    latest = next(reversed(rows.values()), None) if rows else None
                    if latest is not None:
                        print(
                            f"完成 {model_alias}/{method} b{batch_size} "
                            f"i{latest.get('input_length')}: {latest.get('status')} "
                            f"ttft_ms={latest.get('ttft_ms')} "
                            f"tpot_ms={latest.get('tpot_ms')} "
                            f"peak_mib={latest.get('peak_allocated_mib')}",
                            flush=True,
                        )

                if now - last_heartbeat >= 60:
                    if current is None:
                        detail = "模型初始化仍在运行"
                    else:
                        detail = (
                            f"当前 i{current.get('input_length')} 已运行 "
                            f"{cell_elapsed:.0f}s"
                        )
                    print(
                        f"等待 {model_alias}/{method} b{batch_size}：{detail}",
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
            status_file,
        )
        try:
            current_cell_path.unlink()
        except FileNotFoundError:
            pass
    elif return_code != 0:
        current_cell = None
        if current_cell_path.is_file():
            try:
                current_cell = json.loads(current_cell_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current_cell = None

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        oom_from_log = log_contains_oom(log_text)
        oom_kill_exit = return_code in {-9, -137, 137}

        if oom_from_log or oom_kill_exit:
            rows_before_worker = read_existing_status(status_file)
            if current_cell is not None:
                current_key = cell_key(
                    model,
                    method,
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
                if cell_key(model, method, batch_size, length) not in rows_before_worker
            ]

            if current_cell is None and pending_lengths:
                current_cell = {
                    "batch_size": batch_size,
                    "input_length": pending_lengths[0],
                    "started_monotonic": worker_started,
                }

            if current_cell is not None:
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
                    status_file,
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
                    status_file,
                )
        else:
            rows = append_worker_exit_rows(
                args,
                model,
                method,
                model_alias,
                batch_size,
                return_code,
                status_file,
            )

        try:
            current_cell_path.unlink()
        except FileNotFoundError:
            pass
    else:
        rows = read_existing_status(status_file)

    save_reports(args, rows, configuration, finished=False)
    return rows


def main() -> None:
    args = parse_args()
    if args.state_dir is None:
        args.state_dir = args.output_root
    args.data_file = args.data_file.resolve()
    args.output_root = args.output_root.resolve()
    args.state_dir = args.state_dir.resolve()
    validate_args(args)

    source_rows = count_jsonl_rows(args.data_file)
    maximum_samples = max(args.batch_sizes) * args.measured_rounds
    if source_rows < maximum_samples:
        raise ValueError(
            f"RULER 文件只有 {source_rows} 条输入；最大 batch 需要 {maximum_samples} 条，"
            "以保证每组有指定数量的完整测量轮次。"
        )

    configuration = {
        "models": args.models,
        "methods": args.methods,
        "batch_sizes": args.batch_sizes,
        "input_lengths": args.input_lengths,
        "data_file": str(args.data_file),
        "source_rows": source_rows,
        "output_root": str(args.output_root),
        "experiment": args.experiment,
        "task": args.task,
        "budget": args.budget,
        "output_length": args.output_length,
        "warmup": args.warmup,
        "measured_rounds": args.measured_rounds,
        "timeout_seconds": args.timeout_seconds,
        "physical_gpu": args.physical_gpu,
        "device": args.device,
        "python": sys.executable,
    }
    total = len(args.models) * len(args.methods) * len(args.batch_sizes) * len(args.input_lengths)
    if args.dry_run:
        print(json.dumps({"planned_combinations_before_pruning": total, **configuration}, ensure_ascii=False, indent=2))
        return

    args.state_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_file = args.state_dir / "status.csv"
    rows = read_existing_status(status_file)

    if args.export_csv_only:
        save_reports(args, rows, configuration, finished=True)
        print(
            f"只导出 CSV，未启动模型：{args.output_root}",
            flush=True,
        )
        return

    save_reports(args, rows, configuration, finished=False)

    planned_position = 0
    for model in args.models:
        model_alias = load_benchmark_choices()[2][model]
        for method in args.methods:
            for batch_size in args.batch_sizes:
                rows = read_existing_status(status_file)
                setup_failures = [
                    row
                    for row in rows.values()
                    if row.get("model") == model
                    and row.get("method") == method
                    and row.get("setup_failure")
                ]
                pending_lengths = []
                for input_length in args.input_lengths:
                    planned_position += 1
                    key = cell_key(model, method, batch_size, input_length)
                    if key in rows:
                        print(
                            f"[{planned_position}/{total}] RESUME {model_alias}/{method} "
                            f"b{batch_size} i{input_length}: {rows[key].get('status')}",
                            flush=True,
                        )
                        continue

                    if setup_failures:
                        source = setup_failures[-1]
                        row = {
                            "model": model,
                            "model_alias": model_alias,
                            "method": method,
                            "batch_size": batch_size,
                            "input_length": input_length,
                            "output_length": args.output_length,
                            "status": "skipped_setup_failure",
                            "phase": "loading",
                            "skip_reason": source.get("error"),
                            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                        }
                        rows[key] = row
                        append_status(status_file, row)
                        save_reports(args, rows, configuration, finished=False)
                        print(
                            f"[{planned_position}/{total}] SKIP {model_alias}/{method} "
                            f"b{batch_size} i{input_length}: prior loader failure",
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
                            "batch_size": batch_size,
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
                        rows[key] = row
                        append_status(status_file, row)
                        save_reports(args, rows, configuration, finished=False)
                        print(
                            f"[{planned_position}/{total}] SKIP {model_alias}/{method} "
                            f"b{batch_size} i{input_length}: dominated by {failure['status']} "
                            f"at b{failure['batch_size']} i{failure['input_length']}",
                            flush=True,
                        )
                        continue

                    pending_lengths.append(input_length)

                if pending_lengths:
                    print(
                        f"[{planned_position}/{total}] START {model_alias}/{method} "
                        f"b{batch_size} lengths={pending_lengths}; worker 将复用已加载模型",
                        flush=True,
                    )
                    rows = run_batch_worker(
                        args,
                        configuration,
                        model,
                        model_alias,
                        method,
                        batch_size,
                        status_file,
                    )
                    for input_length in pending_lengths:
                        key = cell_key(model, method, batch_size, input_length)
                        row = rows.get(key, {})
                        print(
                            f"[{planned_position}/{total}] {row.get('status', 'UNKNOWN').upper()} "
                            f"{model_alias}/{method} b{batch_size} i{input_length} "
                            f"ttft_ms={row.get('ttft_ms')} "
                            f"tpot_ms={row.get('tpot_ms')} "
                            f"peak_mib={row.get('peak_allocated_mib')} "
                            f"{row.get('error', row.get('skip_reason', ''))}",
                            flush=True,
                        )

    save_reports(args, rows, configuration, finished=True)
    print(f"矩阵运行结束：{args.state_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
