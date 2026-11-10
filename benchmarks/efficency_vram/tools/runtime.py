"""运行失败识别和网格剪枝规则。"""

from __future__ import annotations


def log_contains_oom(log_text: str) -> bool:
    """检查日志是否包含常见的设备或分配器 OOM 标记。"""
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
    """从较长的 worker 日志中提取末尾错误上下文。"""
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


def threshold_failures(rows: dict[str, dict], model: str, method: str) -> list[dict]:
    """返回同一模型和方法下可用于剪枝的 OOM/timeout 记录。"""
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
    """查找同时覆盖当前 batch 和输入长度的失败记录。"""
    for failure in failures:
        if batch_size >= failure["batch_size"] and input_length >= failure["input_length"]:
            return failure
    return None
