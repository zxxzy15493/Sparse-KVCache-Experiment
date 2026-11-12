"""整理效率测量记录并维护方法目录下的统计 CSV。"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from benchmarks.common import model_alias


def round_floats(value):
    """将嵌套结构中的浮点数统一保留两位。"""
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, dict):
        return {key: round_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(item) for item in value]
    return value


def format_number(value) -> str:
    """格式化控制台数值，统一保留两位。"""
    return "—" if value is None else f"{value:.2f}"


def csv_row(row: dict) -> dict:
    """格式化 CSV 行中的浮点数。"""
    return {
        key: f"{value:.2f}" if isinstance(value, float) else value
        for key, value in row.items()
    }


def summarize(records: list[dict]) -> dict:
    """按实际 backend、batch 和长度分别汇总测量记录。"""
    groups = {}
    for record in records:
        key = (
            record["backend"],
            record["batch_size"],
            record["input_length"],
            record["fixed_output_length"],
        )
        groups.setdefault(key, []).append(record)

    summary = {
        "num_batches": len(records),
        "num_samples": sum(row["batch_size"] for row in records),
        "groups": [],
    }
    for (backend, batch_size, input_length, output_length), rows in sorted(groups.items()):
        group = {
            "backend": backend,
            "batch_size": batch_size,
            "input_length": input_length,
            "fixed_output_length": output_length,
            "num_batches": len(rows),
            "num_samples": len(rows) * batch_size,
        }
        for field in (
            "ttft_s",
            "tpot_ms",
            "generation_latency_s",
            "decode_latency_s",
            "decode_peak_allocated_mib",
            "decode_peak_reserved_mib",
        ):
            if any(field in row for row in rows):
                values = [row[field] for row in rows if row.get(field) is not None]
                group[f"mean_{field}"] = statistics.mean(values) if values else None
                group[f"median_{field}"] = statistics.median(values) if values else None

        if any("memory_by_device" in row for row in rows):
            for field in (
                "peak_allocated_mib",
                "peak_reserved_mib",
                "sum_device_peak_allocated_mib",
                "sum_device_peak_reserved_mib",
            ):
                group[field] = max(row[field] for row in rows)

            devices = {device for row in rows for device in row["memory_by_device"]}
            group["memory_by_device"] = {}
            for device in sorted(devices):
                snapshots = [
                    row["memory_by_device"][device]
                    for row in rows
                    if device in row["memory_by_device"]
                ]
                group["memory_by_device"][device] = {
                    field: max(snapshot[field] for snapshot in snapshots)
                    for field in snapshots[0]
                }

        summary["groups"].append(group)

    return summary


def metrics_from_summary(
    summary: dict,
    batch_size: int,
    rounds: int,
    output_length: int,
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
        "ttft_ms": (
            group.get("mean_ttft_s") * 1000
            if group.get("mean_ttft_s") is not None
            else None
        ),
        "tpot_ms": group.get("mean_tpot_ms"),
        "generation_latency_s": mean_generation_latency,
        "decode_throughput_tokens_per_s": decode_throughput,
        "peak_allocated_mib": group.get("peak_allocated_mib"),
        "peak_reserved_mib": group.get("peak_reserved_mib"),
        "decode_peak_allocated_mib": group.get("mean_decode_peak_allocated_mib"),
        "decode_peak_reserved_mib": group.get("mean_decode_peak_reserved_mib"),
        "incremental_peak_allocated_mib": (
            group.get("memory_by_device", {})
            .get("cuda:0", {})
            .get("incremental_peak_allocated_mib")
        ),
    }


ROUND_MEASUREMENT_FIELDS = (
    "ttft_s",
    "ttft_ms",
    "tpot_ms",
    "generation_latency_s",
    "decode_latency_s",
    "decode_throughput_tokens_per_s",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "decode_peak_allocated_mib",
    "decode_peak_reserved_mib",
    "memory_by_device",
)


def round_measurement_records(measurements: list[dict]) -> list[dict]:
    return [
        {
            "round_index": measurement.get("batch_index", position - 1) + 1,
            **{
                field: measurement[field]
                for field in ROUND_MEASUREMENT_FIELDS
                if field in measurement
            },
        }
        for position, measurement in enumerate(measurements, start=1)
    ]


def cell_key(
    model: str,
    method: str,
    budget: int,
    batch_size: int,
    input_length: int,
) -> str:
    return f"{model}\t{method}\t{budget}\t{batch_size}\t{input_length}"


def method_history_path(
    output_root: Path,
    model: str,
    method: str,
) -> Path:
    return (
        output_root
        / model_alias(model)
        / method
        / "history.csv"
    )


HISTORY_FIELDS = [
    "record_type",
    "round_index",
    "method",
    "budget",
    "batch_size",
    "input_length",
    "output_length",
    "status",
    "phase",
    "ttft_ms",
    "tpot_ms",
    "generation_latency_s",
    "decode_throughput_tokens_per_s",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "incremental_peak_allocated_mib",
    "decode_peak_allocated_mib",
    "decode_peak_reserved_mib",
    "setup_failure",
    "error",
    "skip_reason",
]

AVERAGE_FIELDS = [
    "method",
    "budget",
    "batch_size",
    "input_length",
    "latest_status",
    "average_ttft_ms",
    "average_tpot_ms",
    "average_generation_latency_s",
    "average_decode_throughput_tokens_per_s",
    "average_peak_allocated_mib",
    "average_peak_reserved_mib",
    "average_decode_peak_allocated_mib",
    "average_decode_peak_reserved_mib",
    "latest_error",
    "latest_skip_reason",
]


def measurement_history_row(event: dict, measurement: dict, round_index: int) -> dict:
    if measurement.get("ttft_ms") is not None:
        ttft_ms = measurement["ttft_ms"]
    elif measurement.get("ttft_s") is not None:
        ttft_ms = measurement["ttft_s"] * 1000
    else:
        ttft_ms = None

    device_memory = measurement.get("memory_by_device", {}) or {}
    status = event.get("status")
    return {
        "record_type": "measurement",
        "round_index": measurement.get("round_index", round_index),
        "method": event.get("method"),
        "budget": event.get("budget", 1024),
        "batch_size": measurement.get("batch_size", event.get("batch_size")),
        "input_length": measurement.get("input_length", event.get("input_length")),
        "output_length": event.get("output_length"),
        "status": status,
        "phase": "complete" if status == "passed" else event.get("phase"),
        "ttft_ms": ttft_ms,
        "tpot_ms": measurement.get("tpot_ms"),
        "generation_latency_s": measurement.get("generation_latency_s"),
        "decode_throughput_tokens_per_s": measurement.get(
            "decode_throughput_tokens_per_s"
        ),
        "peak_allocated_mib": measurement.get("peak_allocated_mib"),
        "peak_reserved_mib": measurement.get("peak_reserved_mib"),
        "decode_peak_allocated_mib": measurement.get("decode_peak_allocated_mib"),
        "decode_peak_reserved_mib": measurement.get("decode_peak_reserved_mib"),
        "incremental_peak_allocated_mib": device_memory.get(
            "cuda:0", {}
        ).get("incremental_peak_allocated_mib"),
        "setup_failure": event.get("setup_failure"),
        "error": event.get("error"),
        "skip_reason": event.get("skip_reason"),
    }


def history_rows_for_event(event: dict) -> list[dict]:
    round_measurements = event.get("round_measurements")
    if isinstance(round_measurements, str) and round_measurements:
        try:
            round_measurements = json.loads(round_measurements)
        except json.JSONDecodeError:
            round_measurements = None

    if isinstance(round_measurements, list) and round_measurements:
        return [
            measurement_history_row(event, measurement, position)
            for position, measurement in enumerate(round_measurements, start=1)
        ]

    if event.get("status") == "passed":
        measurement = {
            "ttft_ms": event.get("ttft_ms"),
            "tpot_ms": event.get("tpot_ms"),
            "generation_latency_s": event.get("generation_latency_s"),
            "decode_throughput_tokens_per_s": event.get(
                "decode_throughput_tokens_per_s"
            ),
            "peak_allocated_mib": event.get("peak_allocated_mib"),
            "peak_reserved_mib": event.get("peak_reserved_mib"),
            "decode_peak_allocated_mib": event.get("decode_peak_allocated_mib"),
            "decode_peak_reserved_mib": event.get("decode_peak_reserved_mib"),
        }
        return [measurement_history_row(event, measurement, 1)]

    return [{"record_type": "status", **event}]


def _read_history_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []

    rows = []
    integer_fields = {
        "round_index",
        "budget",
        "batch_size",
        "input_length",
        "output_length",
    }
    float_fields = {
        "ttft_ms",
        "tpot_ms",
        "generation_latency_s",
        "decode_throughput_tokens_per_s",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "incremental_peak_allocated_mib",
        "decode_peak_allocated_mib",
        "decode_peak_reserved_mib",
    }

    with path.open(encoding="utf-8-sig", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            row = dict(raw_row)
            for field in integer_fields:
                value = row.get(field)
                row[field] = int(value) if value not in (None, "") else None
            if row.get("budget") is None:
                row["budget"] = 1024
            for field in float_fields:
                value = row.get(field)
                row[field] = float(value) if value not in (None, "") else None
            setup_failure = row.get("setup_failure")
            row["setup_failure"] = (
                setup_failure.lower() == "true"
                if setup_failure not in (None, "")
                else row.get("phase") == "loading"
                and row.get("status") in {"failed", "skipped_setup_failure"}
            )
            rows.append(row)

    return rows


def read_history_state(
    path: Path,
    model: str,
    method: str,
    budget: int | None = None,
) -> dict[str, dict]:
    history_rows = _read_history_rows(path)
    unsupported_budgets = {
        row.get("budget", 1024)
        for row in history_rows
        if row.get("status") == "unsupported_batch"
    }
    latest_by_key = {}
    measurements_by_key: dict[str, list[dict]] = defaultdict(list)

    for row in history_rows:
        if row.get("method") not in (None, "", method):
            continue
        row_budget = row.get("budget", 1024)
        if budget is not None and row_budget != budget:
            continue
        if (
            row.get("status") == "skipped_unsupported_batch"
            and row_budget not in unsupported_budgets
        ):
            continue

        batch_size = row.get("batch_size")
        input_length = row.get("input_length")
        if batch_size is None or input_length is None:
            continue

        row["budget"] = row_budget
        key = cell_key(model, method, row_budget, batch_size, input_length)
        row["model"] = model
        row["method"] = method
        latest_by_key[key] = row
        if row.get("record_type") == "measurement":
            measurements_by_key[key].append(row)

    results = {}
    for key, latest in latest_by_key.items():
        result = dict(latest)
        if latest.get("record_type") == "measurement":
            measurements = measurements_by_key[key]
            for field in (
                "ttft_ms",
                "tpot_ms",
                "generation_latency_s",
                "decode_throughput_tokens_per_s",
                "peak_allocated_mib",
                "peak_reserved_mib",
                "decode_peak_allocated_mib",
                "decode_peak_reserved_mib",
            ):
                values = [row[field] for row in measurements if row.get(field) is not None]
                result[field] = statistics.mean(values) if values else None
        results[key] = result

    return results


def append_history_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    history_rows = history_rows_for_event(event)

    if path.is_file() and path.stat().st_size > 0:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)

        if existing_fields != HISTORY_FIELDS:
            write_csv(path, HISTORY_FIELDS, existing_rows)

    needs_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerows(csv_row(row) for row in history_rows)
        handle.flush()

    save_average_csv(path)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_row(row) for row in rows)


def average_attempts(measurements: list[dict], latest: dict) -> dict:
    averaged_fields = {
        "ttft_ms": "average_ttft_ms",
        "tpot_ms": "average_tpot_ms",
        "generation_latency_s": "average_generation_latency_s",
        "decode_throughput_tokens_per_s": "average_decode_throughput_tokens_per_s",
        "peak_allocated_mib": "average_peak_allocated_mib",
        "peak_reserved_mib": "average_peak_reserved_mib",
        "decode_peak_allocated_mib": "average_decode_peak_allocated_mib",
        "decode_peak_reserved_mib": "average_decode_peak_reserved_mib",
    }
    result = {
        "method": latest.get("method"),
        "budget": latest.get("budget", 1024),
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


def save_average_csv(history_path: Path) -> None:
    history_rows = _read_history_rows(history_path)
    unsupported_budgets = {
        row.get("budget", 1024)
        for row in history_rows
        if row.get("status") == "unsupported_batch"
    }
    latest_by_key = {}
    measurements_by_key: dict[tuple[int, int, int], list[dict]] = defaultdict(list)

    for row in history_rows:
        if (
            row.get("status") == "skipped_unsupported_batch"
            and row.get("budget", 1024) not in unsupported_budgets
        ):
            continue
        batch_size = row.get("batch_size")
        input_length = row.get("input_length")
        if batch_size is None or input_length is None:
            continue

        budget = row.get("budget", 1024)
        key = (budget, batch_size, input_length)
        latest_by_key[key] = row
        if row.get("record_type") == "measurement":
            measurements_by_key[key].append(row)

    average_rows = [
        average_attempts(measurements_by_key.get(key, []), latest)
        for key, latest in latest_by_key.items()
    ]
    write_csv(history_path.parent / "average.csv", AVERAGE_FIELDS, average_rows)


def save_method_averages(args, models: list[str] | None = None, methods: list[str] | None = None) -> None:
    """从各方法目录的 history.csv 刷新对应 average.csv。"""
    selected_models = models if models is not None else args.models
    selected_methods = methods if methods is not None else args.methods

    for model in selected_models:
        for method in selected_methods:
            history_path = method_history_path(
                args.output_root,
                model,
                method,
            )
            save_average_csv(history_path)
