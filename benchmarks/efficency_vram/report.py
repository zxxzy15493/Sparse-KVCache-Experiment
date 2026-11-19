"""固定字段的记录、统计、预测文本及简单运行日志。"""

import csv
import fcntl
import json
import statistics
from collections import defaultdict
from pathlib import Path

from benchmarks.common import model_alias

def round_floats(value):
    """测量指标保留两位；阈值、稀疏率参数保留原精度。"""
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, dict):
        return {
            key: item if key in {"threshold", "sparsity"} else round_floats(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [round_floats(item) for item in value]
    return value


def format_number(value) -> str:
    """格式化控制台数值，统一保留两位。"""
    return "—" if value is None else f"{value:.2f}"


def format_parameter(value):
    """参数保留原精度，避免 0.825 与 0.82 的记录和日志混淆。"""
    return f"{value:.2f}" if value == round(value, 2) else str(value)


def csv_row(row: dict) -> dict:
    """格式化 CSV 行中的浮点数。"""
    result = {}
    for key, value in row.items():
        if isinstance(value, float):
            if key in {"threshold", "sparsity"}:
                value = format_parameter(value)
            else:
                value = f"{value:.2f}"
        result[key] = value
    return result


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


def parameter_key(budget, threshold=None, sparsity=None):
    if sparsity is not None:
        return f"sparsity-{format_parameter(sparsity)}"
    return f"threshold-{format_parameter(threshold)}" if threshold is not None else budget


def parameter_label(budget, threshold=None, sparsity=None):
    if sparsity is not None:
        return f"s{format_parameter(sparsity)}"
    return f"t{format_parameter(threshold)}" if threshold is not None else f"k{budget}"


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
        "threshold": event.get("threshold"),
        "sparsity": event.get("sparsity"),
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
        "threshold": latest.get("threshold"),
        "sparsity": latest.get("sparsity"),
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


HISTORY_FIELDS = [
    "record_type",
    "attempted",
    "round_index",
    "method",
    "budget",
    "threshold",
    "sparsity",
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
    "threshold",
    "sparsity",
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


def make_prediction_record(
    sample_index: int,
    output: str,
    round_index: int,
    batch_size: int,
    input_length: int,
    output_length: int,
) -> dict:
    return {
        "round_index": round_index,
        "batch_size": batch_size,
        "input_length": input_length,
        "output_length": output_length,
        "sample_index": sample_index,
        "output": output,
    }

def read_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        rows = list(csv.DictReader(handle))
    for row in rows:
        for name in ("budget", "batch_size", "input_length", "output_length", "round_index"):
            row[name] = int(row[name]) if row.get(name) else None
        row["threshold"] = float(row["threshold"]) if row.get("threshold") else None
        row["sparsity"] = float(row["sparsity"]) if row.get("sparsity") else None
        for name in HISTORY_FIELDS:
            if name.endswith(("_ms", "_s", "_mib")):
                row[name] = float(row[name]) if row.get(name) else None
        # 旧记录没有 attempted；未开始的推导记录不能当作实际执行。
        row["attempted"] = (
            row["attempted"].lower() == "true"
            if row.get("attempted") else not row.get("status", "").startswith("skipped_")
            and row.get("phase") != "not_started"
        )
    return rows


def completed(args):
    """只认实际执行过的记录，不从历史恢复任何支配规则。"""
    return {
        (parameter_key(row["budget"], row["threshold"], row["sparsity"]), row["batch_size"], row["input_length"])
        for row in read_rows(method_history_path(args.output_root, args.model, args.method))
        if row["attempted"]
    }


def append_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8-sig", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing_fields = next(csv.reader(handle), None)
        old_rows = []
        missing_fields = [
            name for name in ("threshold", "sparsity")
            if existing_fields and name not in existing_fields
            and any(row.get(name) is not None for row in rows)
        ]
        rewrite_header = bool(missing_fields)
        if rewrite_header:
            old_rows = list(csv.DictReader(handle, fieldnames=existing_fields))
            existing_fields.extend(missing_fields)
            handle.seek(0)
            handle.truncate()
        if existing_fields:
            fields = existing_fields
        handle.seek(0, 2)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not existing_fields or rewrite_header:
            writer.writeheader()
        writer.writerows(old_rows)
        writer.writerows(csv_row(row) for row in rows)
        handle.flush()


def save_result(args, row):
    path = method_history_path(args.output_root, args.model, args.method)
    rows = history_rows_for_event(row)
    for item in rows:
        item["attempted"] = row.get("attempted", True)
    append_csv(path, HISTORY_FIELDS, rows)
    save_average(path)
    parameter = parameter_label(
        args.budget, getattr(args, "threshold", None), getattr(args, "sparsity", None),
    )
    print(
        f"{args.model}/{args.method} {parameter} b{args.batch_size} "
        f"i{row['input_length']}: {row['status']} "
        f"peak_mib={format_number(row.get('peak_allocated_mib'))}",
        flush=True,
    )


def save_average(path):
    target = path.parent / "average.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a+", encoding="utf-8-sig", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        # 同目录可能由两个 GPU 同时写入；锁内读取最新明细再覆盖汇总。
        latest = {}
        measurements = defaultdict(list)
        for row in read_rows(path):
            key = (parameter_key(row["budget"], row["threshold"], row["sparsity"]), row["batch_size"], row["input_length"])
            latest[key] = row
            if row.get("record_type") == "measurement":
                measurements[key].append(row)
        handle.seek(0)
        handle.truncate()
        writer = csv.DictWriter(handle, fieldnames=AVERAGE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_row(average_attempts(measurements[key], row)) for key, row in latest.items())


def append_predictions(path, predictions):
    if not predictions:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        contents = handle.read()
        document = json.loads(contents) if contents else {"schema_version": 1, "predictions": []}
        document["predictions"].extend(predictions)
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(document, ensure_ascii=False, indent=2) + "\n")


def status(args, length, name, attempted=True, reason="", phase="generation"):
    return dict(
        model=args.model, method=args.method, budget=args.budget,
        threshold=getattr(args, "threshold", None),
        sparsity=getattr(args, "sparsity", None),
        batch_size=args.batch_size, input_length=length, output_length=args.output_length,
        status=name, attempted=attempted, phase=phase, error=reason,
    )


def notify(kind, **values):
    print("EFF_EVENT " + json.dumps(dict(kind=kind, **values)), flush=True)


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


def prefill_summary(records: list[dict]) -> dict:
    """把每个 batch 的 prefill 模式与对照结果汇总成单元格层面的字段。"""
    summary = {}
    modes = sorted({str(record.get("prefill_mode")) for record in records
                    if record.get("prefill_mode")})
    if modes:
        summary["prefill_mode"] = ",".join(modes)
    fallbacks = sorted({str(record["serial_prefill_fallback"]) for record in records
                        if record.get("serial_prefill_fallback")})
    if fallbacks:
        summary["serial_prefill_fallback"] = " | ".join(fallbacks)[:400]
    verifies = sorted({str(record["serial_prefill_verify"]) for record in records
                       if record.get("serial_prefill_verify")})
    if verifies:
        summary["serial_prefill_verify"] = ",".join(verifies)
    mismatches = [record["serial_prefill_mismatch_tokens"] for record in records
                  if record.get("serial_prefill_mismatch_tokens")]
    if mismatches:
        summary["serial_prefill_mismatch_tokens"] = int(sum(mismatches))
    return summary


def print_measurement(batch_index, batch_count, record):
    print(
        f"Batch {batch_index + 1}/{batch_count}: "
        f"prefill={record.get('prefill_mode')}, "
        f"ttft_s={format_number(record.get('ttft_s'))}, "
        f"tpot_ms={format_number(record.get('tpot_ms'))}, "
        f"peak_allocated_mib={format_number(record.get('peak_allocated_mib'))}",
        flush=True,
    )
    if record.get("serial_prefill_fallback"):
        print(f"Serial prefill 回退到批量 prefill：{record['serial_prefill_fallback']}",
              flush=True)
    if record.get("serial_prefill_verify"):
        print(f"Serial prefill 对照：{record.get('serial_prefill_verify')} "
              f"mismatch_tokens={record.get('serial_prefill_mismatch_tokens')}",
              flush=True)

