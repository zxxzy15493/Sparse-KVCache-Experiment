"""子进程隔离、单长度硬超时及仅本次运行有效的超时支配规则。"""

import json
import os
import re
import selectors
import signal
import subprocess
import time
from pathlib import Path

from benchmarks.efficency_vram import report


class Timeouts:
    def __init__(self):
        self.points = []

    def add(self, args, length):
        self.points.append((
            args.model,
            args.method,
            args.budget,
            args.batch_size,
            length,
            getattr(args, "threshold", None),
            getattr(args, "sparsity", None),
        ))

    def source(self, args, length):
        return next((point for point in self.points if
            point[0] == args.model and point[1] == args.method
            and point[5] == getattr(args, "threshold", None)
            and point[6] == getattr(args, "sparsity", None)
            and (args.budget is None or args.budget >= point[2])
            and args.batch_size >= point[3] and length >= point[4]
        ), None)


def stop(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def error_status(error, batch_size):
    text = str(error)
    if any(word in text.lower() for word in ("out of memory", "cuda_error_out_of_memory", "cublas_status_alloc_failed")):
        return "oom"
    if batch_size > 1 and is_unsupported_batch_error(text):
        return "unsupported_batch"
    return "failed"


UNSUPPORTED_BATCH_PATTERNS = (
    r"only supports? batch sizes?\s*(?:greater than\s*)?1\b",
    r"only supports? batch size\s*1\b",
    r"supports? batch size\s*1\s+only\b",
    r"requires? batch size\s*1\b",
    r"(?:do|does) not support (?:batch(?: size)?|bsz)(?:\s*[><=]+\s*\d+)?",
    r"(?:unsupported|not supported) batch(?: size)?(?:\s*[><=]+\s*\d+)?",
    r"cannot handle batch sizes?\s*>\s*1\b",
    r"batching is not implemented",
    r"assert\s+(?:\w+\.)?(?:bsz|batch_size)\s*==\s*1\b",
)


def is_unsupported_batch_error(error_text):
    normalized = error_text.lower().replace("_", " ")
    return any(re.search(pattern, normalized) for pattern in UNSUPPORTED_BATCH_PATTERNS)


def run_worker(args, command, lengths):
    """通过 stdout 开始/完成事件计时，不使用状态文件或历史失败推断。"""
    parameter = report.parameter_label(
        args.budget, getattr(args, "threshold", None), getattr(args, "sparsity", None),
    )
    log_path = (
        args.state_dir / "logs" / args.model.replace("/", "--") / args.method
        / f"{parameter}_b{args.batch_size}_worker.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{args.model}/{args.method} {parameter} b{args.batch_size} {lengths} → {log_path}", flush=True)
    current = lengths[0]
    phase = "loading"
    started = time.monotonic()
    buffer = b""
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while selector.get_map() or process.poll() is None:
                for key, _ in selector.select(timeout=0.1):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    log.write(chunk)
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if line.startswith(b"EFF_EVENT "):
                            event = json.loads(line[len(b"EFF_EVENT "):])
                            if event["kind"] == "start":
                                current = event["length"]
                                phase = "generation"
                                started = time.monotonic()
                            elif event["kind"] == "done":
                                current = None
                                started = time.monotonic()
                if process.poll() is None and time.monotonic() - started >= args.timeout_seconds:
                    stop(process)
                    if current is not None:
                        row = report.status(args, current, "timeout", reason="达到单组合超时上限；详见 worker log。", phase=phase)
                        report.save_result(args, row)
                        return current
                    log.write(b"worker cleanup timed out\n")
                    return None
            code = process.wait()
            if code != 0:
                key = report.parameter_key(
                    args.budget, getattr(args, "threshold", None), getattr(args, "sparsity", None),
                )
                pending = [length for length in lengths if (key, args.batch_size, length) not in report.completed(args)]
                if pending:
                    report.save_result(args, report.status(
                        args, current if current in pending else pending[0], "failed",
                        reason=f"worker exit={code}；详见 {log_path}", phase=phase,
                    ))
        finally:
            selector.close()
            process.stdout.close()
            stop(process)
    return None
