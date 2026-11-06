"""将一次或多次 benchmark 运行追加到唯一的 pred.json。"""

from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
from typing import Any


EMPTY_DOCUMENT = b'{"schema_version":1,"results":[\n]}\n'
DOCUMENT_SUFFIX = b"\n]}\n"


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def append_result(path: Path, result: dict[str, Any]) -> None:
    """向 pred.json 的 results 数组追加一条记录，不重写已有预测。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)

    with os.fdopen(file_descriptor, "r+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

        handle.seek(0, os.SEEK_END)
        file_size = handle.tell()
        if file_size == 0:
            handle.write(EMPTY_DOCUMENT)
            handle.flush()
            file_size = len(EMPTY_DOCUMENT)

        if file_size < len(DOCUMENT_SUFFIX):
            raise ValueError(f"pred.json 格式不完整：{path}")

        handle.seek(file_size - len(DOCUMENT_SUFFIX))
        suffix = handle.read(len(DOCUMENT_SUFFIX))
        if suffix != DOCUMENT_SUFFIX:
            raise ValueError(f"pred.json 结尾格式异常：{path}")

        handle.seek(file_size - len(DOCUMENT_SUFFIX) - 1)
        previous_byte = handle.read(1)
        separator = b"" if previous_byte == b"[" else bytes((44,))

        result_bytes = json.dumps(
            json_safe(result),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
            allow_nan=False,
        ).encode("utf-8")

        handle.seek(file_size - len(DOCUMENT_SUFFIX))
        handle.truncate()
        handle.write(separator)
        handle.write(b"\n")
        handle.write(result_bytes)
        handle.write(DOCUMENT_SUFFIX)
        handle.flush()

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []

    document = json.loads(path.read_text(encoding="utf-8"))
    return document.get("results", [])


def write_results(path: Path, results: list[dict[str, Any]]) -> None:
    """原子写入迁移后的统一结果文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    serialized_results = [
        json.dumps(
            json_safe(result),
            ensure_ascii=False,
            indent=2,
            default=str,
            allow_nan=False,
        )
        for result in results
    ]
    document = '{"schema_version":1,"results":['
    if serialized_results:
        document += "\n"
        document += ",\n".join(serialized_results)
    document += "\n]}\n"
    temporary_path.write_text(document, encoding="utf-8")
    temporary_path.replace(path)
