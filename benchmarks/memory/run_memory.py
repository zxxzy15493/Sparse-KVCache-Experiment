#!/usr/bin/env python3
"""按模型、方法、batch size 和上下文长度测量 CPU 内存。"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from fcntl import LOCK_EX, flock
from itertools import product
from pathlib import Path


SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parents[2]))

from benchmarks.common import METHODS, MODEL_CHOICES, model_alias
from benchmarks.memory.cpu_memory import CPUMemoryMonitor
from benchmarks.memory.reporting import write_aligned_report


SUMMARY_FIELDS = (
    "model", "method", "batch_size", "context_len", "budget", "output_length", "status",
    "model_loading_peak_mib", "inference_peak_mib", "inference_peak_increase_mib",
    "process_peak_mib", "phases_csv",
)
PHASE_FIELDS = (
    "phase", "prefill_mode", "rss_start_mib", "rss_end_mib", "rss_peak_mib", "rss_peak_increase_mib",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", "--model", nargs="+", choices=MODEL_CHOICES, required=True, metavar="MODEL",
                        help="模型别名或 Hugging Face ID。")
    parser.add_argument("--methods", "--method", nargs="+", choices=METHODS, required=True, metavar="METHOD",
                        help=f"可选方法 {'、'.join(METHODS)}")
    parser.add_argument("--batch-sizes", "--batch-size", "--batch_size", nargs="+", type=int, default=[1])
    parser.add_argument("--context-lens", "--context-len", "--context_len", "--input-lengths",
                        nargs="+", type=int, default=[4096])
    parser.add_argument("--data-file", type=Path, required=True, help="RULER JSONL 路径，可用 {model} 占位。")
    parser.add_argument("--output-root", type=Path, default=SCRIPT.parent / "output")
    parser.add_argument("--task", default="niah_single_1")
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measured-rounds", type=int, default=3)
    parser.add_argument("--efficiency-backend", action="store_true", help="使用 Quest / ClusterKV 的效率后端。")
    parser.add_argument("--serial-prefill", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--cell", nargs=4, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(*args.batch_sizes, *args.context_lens, args.budget, args.output_length, args.measured_rounds) < 1:
        parser.error("batch size、上下文长度、budget、输出长度和测量轮数必须为正数。")
    if args.warmup < 0:
        parser.error("预热轮数必须大于等于 0。")
    return args


def make_options(args, model_name, method, batch_size, context_len):
    max_seq_length = context_len
    if method == "pqcache":
        max_seq_length += max(0, args.output_length - 1024)
    return argparse.Namespace(
        model=model_name,
        method=method,
        device="cuda:0",
        batch_size=batch_size,
        input_length=context_len,
        context_length=context_len,
        max_seq_length=max_seq_length,
        fixed_output_length=args.output_length,
        task=args.task,
        budget=args.budget,
        set=[],
        dataset="ruler",
        efficiency_backend=args.efficiency_backend,
        serial_prefill=args.serial_prefill,
        _cleanup=None,
    )


def run_cell(args, model_name, method, batch_size, context_len, phase_path):
    monitor = CPUMemoryMonitor()
    with monitor.measure("runtime_initialization"):
        import random

        import numpy as np
        import torch

        from benchmarks.memory import inference
        from loaders import load_model_and_tokenizer

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    options = make_options(args, model_name, method, batch_size, context_len)
    alias = model_alias(model_name)
    data_file = Path(str(args.data_file).format(model=alias, model_alias=alias)).resolve()
    sample_count = batch_size * args.measured_rounds
    with monitor.measure("data_loading"):
        batches = inference.read_batches(data_file, batch_size, sample_count)

    with monitor.measure("model_loading"):
        model, tokenizer = load_model_and_tokenizer(options)
        input_device, devices = inference.model_devices(model, method)

    prefill_modes = {}
    try:
        for index in range(1, args.warmup + 1):
            with monitor.measure(f"warmup_{index}"):
                inference.predict_batch(options, model, tokenizer, batches[0], input_device, devices)

        for index, batch in enumerate(batches, start=1):
            phase = f"inference_{index}"
            with monitor.measure(phase):
                _, prefill_modes[phase] = inference.predict_batch(
                    options, model, tokenizer, batch, input_device, devices,
                )
    finally:
        del model
        if options._cleanup is not None:
            options._cleanup()

    process_peak = monitor.process_peak_mib()
    phase_path.parent.mkdir(parents=True, exist_ok=True)
    with phase_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, PHASE_FIELDS)
        writer.writeheader()
        for phase, stats in monitor.phases.items():
            writer.writerow({"phase": phase, "prefill_mode": prefill_modes.get(phase, ""),
                             **{key: f"{value:.2f}" for key, value in stats.items()}})
        writer.writerow({"phase": "process", "rss_peak_mib": f"{process_peak:.2f}"})
    write_aligned_report(phase_path)


def phase_file(model, method, batch_size, context_len):
    return Path(model_alias(model)) / method / f"b{batch_size}_c{context_len}" / "phases.csv"


def run_grid(args):
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.csv"
    print(f"CPU 内存汇总表 {summary_path}", flush=True)
    failed = False
    with summary_path.open("a+", encoding="utf-8", newline="") as handle:
        # 同一输出目录的运行共用汇总表和阶段文件，先取得锁再清空汇总。
        flock(handle, LOCK_EX)
        handle.seek(0)
        handle.truncate()
        writer = csv.DictWriter(handle, SUMMARY_FIELDS)
        writer.writeheader()
        handle.flush()
        for model, method, batch_size, context_len in product(
            args.models, args.methods, args.batch_sizes, args.context_lens,
        ):
            label = f"{model_alias(model)} / {method} / batch={batch_size} / context={context_len}"
            print(f"运行 {label}", flush=True)
            result = subprocess.run([
                sys.executable, "-u", str(SCRIPT), *sys.argv[1:], "--output-root", str(output_root),
                "--cell", model, method, str(batch_size), str(context_len),
            ])
            row = dict(model=model, method=method, batch_size=batch_size, context_len=context_len,
                       budget=args.budget, output_length=args.output_length)
            if result.returncode:
                failed = True
                row["status"] = "failed"
                print(f"失败 {label}，退出码 {result.returncode}", flush=True)
            else:
                relative_path = phase_file(model, method, batch_size, context_len)
                with (output_root / relative_path).open(encoding="utf-8", newline="") as phases_handle:
                    phases = {phase["phase"]: phase for phase in csv.DictReader(phases_handle)}
                inference = [phases[f"inference_{index}"] for index in range(1, args.measured_rounds + 1)]
                inference_peak = max(float(phase["rss_peak_mib"]) for phase in inference)
                inference_increase = max(float(phase["rss_peak_increase_mib"]) for phase in inference)
                row.update(status="passed", model_loading_peak_mib=phases["model_loading"]["rss_peak_mib"],
                           inference_peak_mib=f"{inference_peak:.2f}",
                           inference_peak_increase_mib=f"{inference_increase:.2f}",
                           process_peak_mib=phases["process"]["rss_peak_mib"], phases_csv=str(relative_path))
                print(f"推理峰值 {inference_peak:.2f} MiB，最大增量 {inference_increase:.2f} MiB，"
                      f"进程峰值 {row['process_peak_mib']} MiB", flush=True)
            writer.writerow(row)
            handle.flush()
        report_path = write_aligned_report(summary_path)
    print(f"CPU 内存对齐汇总表 {report_path}", flush=True)
    return int(failed)


def main():
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    args = parse_args()
    if args.cell is not None:
        model, method, batch, context = args.cell
        batch_size, context_len = int(batch), int(context)
        run_cell(args, model, method, batch_size, context_len,
                 args.output_root / phase_file(model, method, batch_size, context_len))
        return 0
    return run_grid(args)


if __name__ == "__main__":
    sys.exit(main())
