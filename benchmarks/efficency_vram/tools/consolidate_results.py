#!/usr/bin/env python3
"""把已有逐组合 JSON/JSONL 迁移到输出根目录唯一的 pred.json。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_OUTPUT_ROOT = (
    BENCHMARK_DIR
    / "output"
    / "llama31_qwen25_7b1m_gpu1_efficiency_vram_32tok"
)
STATUS_CSV_FILE = DEFAULT_OUTPUT_ROOT / "status.csv"
LEGACY_STATUS_FILE = DEFAULT_OUTPUT_ROOT / "status.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    default_status_file = STATUS_CSV_FILE
    if not STATUS_CSV_FILE.is_file() and LEGACY_STATUS_FILE.is_file():
        default_status_file = LEGACY_STATUS_FILE
    parser.add_argument("--status-file", type=Path, default=default_status_file)
    return parser.parse_args()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []

    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def resolve_source_path(value: str | Path | None, output_root: Path) -> Path | None:
    if not value:
        return None

    source_path = Path(value).resolve()
    try:
        source_path.relative_to(output_root.resolve())
    except ValueError:
        return None
    return source_path


def source_key(value: str | Path | None, output_root: Path) -> str | None:
    path = resolve_source_path(value, output_root)
    return str(path) if path is not None else None


def legacy_result_id(metadata_path: Path, output_root: Path) -> str:
    relative_path = metadata_path.relative_to(output_root).as_posix()
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
    return f"legacy-{digest}"


def migrate_artifacts(output_root: Path) -> tuple[list[dict], dict[str, str]]:
    from benchmarks.efficency_vram.tools.result_store import read_results

    pred_json_path = output_root / "pred.json"
    results_by_id = {
        result["result_id"]: result
        for result in read_results(pred_json_path)
        if result.get("result_id")
    }
    result_id_by_source_path = {}

    for metadata_path in sorted(output_root.rglob("*.metadata.json")):
        metadata = read_json(metadata_path)
        if not isinstance(metadata, dict):
            continue

        arguments = metadata.get("arguments", {})
        source_paths = metadata.get("paths", {})
        result_id = legacy_result_id(metadata_path, output_root)

        metrics_path = resolve_source_path(source_paths.get("metrics"), output_root)
        summary_path = resolve_source_path(source_paths.get("summary"), output_root)
        predictions_path = resolve_source_path(source_paths.get("predictions"), output_root)

        summary = read_json(summary_path) if summary_path is not None else None
        result = {
            "result_id": result_id,
            "model": arguments.get("model"),
            "model_alias": metadata_path.parents[3].name,
            "experiment": arguments.get("experiment"),
            "method": arguments.get("method", metadata_path.parents[1].name),
            "task": arguments.get("task", metadata_path.parent.name),
            "batch_size": arguments.get("batch_size"),
            "input_length": arguments.get("input_length"),
            "output_length": arguments.get("fixed_output_length"),
            "status": "passed" if summary is not None else metadata.get("status", "failed"),
            "measurements": read_jsonl(metrics_path) if metrics_path is not None else [],
            "summary": summary,
            "predictions": read_jsonl(predictions_path) if predictions_path is not None else [],
            "metadata": metadata,
        }
        result["metadata"]["paths"] = {"predictions": str(pred_json_path)}
        results_by_id[result_id] = result

        for source_path in (summary_path, metadata_path):
            if source_path is not None:
                result_id_by_source_path[str(source_path)] = result_id

    return list(results_by_id.values()), result_id_by_source_path


def migrate_status_file(
    status_file: Path,
    output_root: Path,
    pred_json_path: Path,
    results: list[dict],
    result_id_by_source_path: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    results_by_id = {result["result_id"]: result for result in results}
    updated_rows = []

    if status_file.is_file():
        if status_file.suffix == ".csv":
            with status_file.open(encoding="utf-8", newline="") as handle:
                rows = [
                    (event_index, dict(row))
                    for event_index, row in enumerate(csv.DictReader(handle), start=1)
                ]
        else:
            def jsonl_rows():
                with status_file.open(encoding="utf-8") as handle:
                    for event_index, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        try:
                            yield event_index, json.loads(line)
                        except json.JSONDecodeError:
                            continue

            rows = jsonl_rows()

        for event_index, row in rows:
            result_id = row.get("pred_result_id")
            if result_id not in results_by_id:
                result_id = result_id_by_source_path.get(
                    source_key(row.get("summary_path"), output_root)
                )
            if result_id is None:
                result_id = f"legacy-status-{event_index}"
                results_by_id[result_id] = {
                    "result_id": result_id,
                    "model": row.get("model"),
                    "model_alias": row.get("model_alias"),
                    "method": row.get("method"),
                    "batch_size": row.get("batch_size"),
                    "input_length": row.get("input_length"),
                    "output_length": row.get("output_length"),
                    "status": row.get("status"),
                    "started_at": row.get("started_at"),
                    "metrics": row,
                    "measurements": [],
                    "predictions": [],
                }

            row.pop("metadata_path", None)
            row["pred_json_path"] = str(pred_json_path)
            row["pred_result_id"] = result_id
            row["summary_path"] = str(pred_json_path)
            updated_rows.append(row)

    results[:] = list(results_by_id.values())
    return updated_rows, results


def write_status_rows(status_file: Path, rows: list[dict]) -> None:
    from benchmarks.efficency_vram.tools.run_grid import STATUS_FIELDS

    temporary_path = status_file.with_suffix(status_file.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row_to_write = dict(row)
            if isinstance(row_to_write.get("command"), (list, dict)):
                row_to_write["command"] = json.dumps(
                    row_to_write["command"],
                    ensure_ascii=False,
                )
            writer.writerow(row_to_write)
    temporary_path.replace(status_file)


def remove_per_combination_files(output_root: Path) -> int:
    suffixes = (
        ".metadata.json",
        ".summary.json",
        ".metrics.jsonl",
        ".pred.jsonl",
    )
    removed = 0
    for path in output_root.rglob("*"):
        if path.is_file() and path.name.endswith(suffixes):
            path.unlink()
            removed += 1
    return removed


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    status_file = args.status_file.resolve()
    status_csv_path = status_file.parent / "status.csv"
    pred_json_path = output_root / "pred.json"
    output_root.mkdir(parents=True, exist_ok=True)

    results, result_id_by_source_path = migrate_artifacts(output_root)

    from benchmarks.efficency_vram.tools.result_store import read_results, write_results

    status_rows, results = migrate_status_file(
        status_file,
        output_root,
        pred_json_path,
        results,
        result_id_by_source_path,
    )
    write_results(pred_json_path, results)

    consolidated_results = read_results(pred_json_path)
    if len(consolidated_results) != len(results):
        raise RuntimeError("pred.json 写入后记录数校验失败。")

    write_status_rows(status_csv_path, status_rows)
    if status_file != status_csv_path and status_file.is_file():
        status_file.unlink()
    removed_files = remove_per_combination_files(output_root)
    print(
        f"合并 {len(results)} 条结果到 {pred_json_path}；"
        f"删除 {removed_files} 个逐组合 JSON/JSONL 文件。"
    )


if __name__ == "__main__":
    main()
