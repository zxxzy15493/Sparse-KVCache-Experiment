#!/usr/bin/env python3
"""Run isolated real-model smoke tests through the unified benchmark entry point."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
ROOT = BENCHMARK_DIR.parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.common import METHODS, MODEL_CHOICES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--model", choices=MODEL_CHOICES, default="qwen3-0.6b")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-length", type=int, default=4096)
    parser.add_argument("--fixed-output-length", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--task", default="niah_single_1")
    parser.add_argument("--timeout", type=float, default=300,
                        help="Per-method timeout in seconds; kills the complete worker process group")
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "benchmarks/ruler/benchmark_root/qwen3-32b/synthetic/4096/data")
    parser.add_argument("--output-root", type=Path,
                        default=BENCHMARK_DIR / "output/qwen3_0.6b_smoke")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.data_dir = args.data_dir.resolve()
    return args


def stop_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def error_from_log(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    useful = [line.strip() for line in lines if line.strip()]
    for line in reversed(useful):
        if "Error:" in line or "Exception:" in line or "AssertionError" in line:
            return line[-1800:]
    return useful[-1][-1800:] if useful else "No child output"


def save_report(root, configuration, rows, finished=False):
    counts = dict(Counter(row["status"] for row in rows))
    report = {"configuration": configuration, "finished": finished, "counts": counts, "results": rows}
    (root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    fields = ["method", "status", "phase", "elapsed_s", "backend", "ttft_ms", "tpot_ms",
              "generation_latency_s", "peak_allocated_mib", "peak_reserved_mib", "error", "log_path"]
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Qwen3 方法冒烟测试", "", f"状态计数：`{counts}`", "",
             f"环境：`{configuration['python']}`；CUDA_VISIBLE_DEVICES=`{configuration['cuda_visible_devices']}`。", "",
             f"模型 `{configuration['model']}`，batch={configuration['batch_size']}，"
             f"input={configuration['input_length']}，output={configuration['fixed_output_length']}，"
             f"budget={configuration['budget']}，warmup={configuration['warmup']}，samples={configuration['num_samples']}。", "",
             "通过表示指定配置下的加载、固定长度生成和统一测量完成；不表示质量或完整性能评估。", "",
             "| 方法 | 状态 | TTFT ms | TPOT ms | allocated peak MiB | 原因 |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for row in rows:
        values = [row.get(key) for key in ("ttft_ms", "tpot_ms", "peak_allocated_mib")]
        numbers = [f"{value:.3f}" if value is not None else "—" for value in values]
        error = (row.get("error") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['method']} | {row['status']} | {' | '.join(numbers)} | {error} |")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    root = args.output_root
    root.mkdir(parents=True)
    import torch
    import transformers

    configuration = {**vars(args), "python": sys.executable,
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "conda_prefix": os.environ.get("CONDA_PREFIX"),
                     "environment": {key: os.environ.get(key) for key in (
                         "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MAX_CPU_IN_USE",
                         "OMP_NUM_THREADS", "MKL_NUM_THREADS", "SUBVEC", "SUBBITS")},
                     "torch_version": torch.__version__, "transformers_version": transformers.__version__}
    rows = []
    save_report(root, configuration, rows)
    for position, method in enumerate(args.methods, 1):
        directory = root / method
        directory.mkdir()
        log = directory / "run.log"
        command = [sys.executable, "-u", str(BENCHMARK_DIR / "pred.py"),
                   "--method", method, "--model", args.model, "--device", "cuda:0",
                   "--batch_size", str(args.batch_size), "--input_length", str(args.input_length),
                   "--fixed_output_length", str(args.fixed_output_length),
                   "--context_length", str(args.context_length), "--budget", str(args.budget),
                   "--num_samples", str(args.num_samples), "--warmup", str(args.warmup),
                   "--task", args.task, "--data-dir", str(args.data_dir), "--measurement", "both",
                   "--save-predictions", "--output-root", str(directory / "results"), "--experiment", "smoke"]
        (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        print(f"[{position}/{len(args.methods)}] START {method}", flush=True)
        started = time.monotonic()
        timed_out = False
        with log.open("w", encoding="utf-8") as handle:
            # PQCache profiles CPU clustering into ./cluster_config.json.
            # Keep such method-owned scratch artifacts with the smoke log.
            process = subprocess.Popen(command, cwd=directory, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=os.environ.copy())
            try:
                returncode = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_process_group(process)
                returncode = process.returncode
            except BaseException:
                stop_process_group(process)
                raise
        row = {"method": method, "status": "timeout" if timed_out else "failed",
               "returncode": returncode, "elapsed_s": round(time.monotonic() - started, 3),
               "command": command, "log_path": str(log)}
        metadata_files = list(directory.glob("results/**/*.metadata.json"))
        metadata = json.loads(metadata_files[0].read_text()) if metadata_files else {}
        row["phase"] = "generation" if metadata.get("backend") else "loading" if metadata else "configuration"
        row["effective_method_config"] = metadata.get("effective_method_config")
        summary_files = list(directory.glob("results/**/*.summary.json"))
        if returncode == 0 and metadata.get("status") == "complete" and summary_files:
            summary = json.loads(summary_files[0].read_text())
            expected_batches = args.num_samples // args.batch_size
            if summary["num_batches"] == expected_batches:
                group = summary["groups"][0]
                row.update(status="passed", phase="complete", backend=group["backend"],
                           num_batches=summary["num_batches"],
                           ttft_ms=group["mean_ttft_s"] * 1000 if group.get("mean_ttft_s") is not None else None,
                           tpot_ms=group.get("mean_tpot_ms"), generation_latency_s=group.get("mean_generation_latency_s"),
                           peak_allocated_mib=group.get("peak_allocated_mib"), peak_reserved_mib=group.get("peak_reserved_mib"),
                           summary_path=str(summary_files[0]))
            else:
                row["error"] = f"Expected {expected_batches} measured batches; got {summary['num_batches']}"
        if row["status"] != "passed" and "error" not in row:
            row["error"] = f"Timeout after {args.timeout}s" if timed_out else metadata.get("error", error_from_log(log))
        if metadata_files and row["status"] != "passed" and metadata.get("status") in {"loading", "running"}:
            # A crashed worker or killed process cannot update its own metadata.
            metadata.update(status="failed", error=row["error"], returncode=returncode,
                            failure_observer="smoke_all_methods",
                            termination_reason="timeout" if timed_out else "child_process_exit")
            metadata_files[0].write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        rows.append(row)
        save_report(root, configuration, rows)
        print(f"[{position}/{len(args.methods)}] {row['status'].upper()} {method}: {row.get('error', '')}", flush=True)
    save_report(root, configuration, rows, finished=True)
    print(f"Report: {root / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
