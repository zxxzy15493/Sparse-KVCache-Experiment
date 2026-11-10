"""保存简化的预测文本。"""

from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def make_prediction_record(
    sample_index: int,
    output: str,
    round_index: int,
    batch_size: int,
    input_length: int,
    output_length: int,
) -> dict[str, Any]:
    return {
        "round_index": round_index,
        "batch_size": batch_size,
        "input_length": input_length,
        "output_length": output_length,
        "sample_index": sample_index,
        "output": output,
    }


def append_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    """追加解码文本，不写入测量数据或运行元信息。"""
    if not predictions:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)

    with os.fdopen(file_descriptor, "r+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

        handle.seek(0)
        contents = handle.read()
        if contents:
            document = json.loads(contents.decode("utf-8"))
            if not isinstance(document, dict) or not isinstance(document.get("predictions"), list):
                raise ValueError(f"预测文件格式不兼容：{path}")
        else:
            document = {"schema_version": 1, "predictions": []}

        document["predictions"].extend(json_safe(predictions))
        serialized = json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        handle.seek(0)
        handle.truncate()
        handle.write(serialized)
        handle.write(b"\n")
        handle.flush()

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
