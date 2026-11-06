"""Aggregate measured batches without combining different actual batch sizes."""

from __future__ import annotations

import statistics


def summarize(records: list[dict]) -> dict:
    groups = {}
    for record in records:
        key = (record["backend"], record["batch_size"], record["input_length"], record["fixed_output_length"])
        groups.setdefault(key, []).append(record)
    summary = {"num_batches": len(records), "num_samples": sum(row["batch_size"] for row in records),
               "groups": []}
    for (backend, batch_size, input_length, output_length), rows in sorted(groups.items()):
        group = {"backend": backend, "batch_size": batch_size, "input_length": input_length,
                 "fixed_output_length": output_length, "num_batches": len(rows),
                 "num_samples": len(rows) * batch_size}
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
        if all("generation_latency_s" in row for row in rows):
            elapsed = sum(row["generation_latency_s"] for row in rows)
            group["output_throughput_tokens_per_s"] = len(rows) * batch_size * output_length / elapsed
        if any("memory_by_device" in row for row in rows):
            for field in ("peak_allocated_mib", "peak_reserved_mib", "sum_device_peak_allocated_mib",
                          "sum_device_peak_reserved_mib"):
                group[field] = max(row[field] for row in rows)
            devices = {device for row in rows for device in row["memory_by_device"]}
            group["memory_by_device"] = {}
            for device in sorted(devices):
                snapshots = [row["memory_by_device"][device] for row in rows if device in row["memory_by_device"]]
                group["memory_by_device"][device] = {
                    field: max(snapshot[field] for snapshot in snapshots) for field in snapshots[0]
                }
        summary["groups"].append(group)
    return summary
