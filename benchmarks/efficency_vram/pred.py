#!/usr/bin/env python3
"""效率与显存测试：矩阵调度和单组测试共用一个入口。"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import sys
import time
import traceback
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common import EFFICIENCY_METHODS, METHODS, MODEL_CHOICES, method_defaults, model_alias
from benchmarks.efficency_vram import report, runtime
from benchmarks.efficency_vram.report import format_number, summarize, metrics_from_summary


THRESHOLD_METHODS = {"topp", "topp32", "xattention", "flexprefill"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct-1M"])
    parser.add_argument("--methods", nargs="+", default=list(EFFICIENCY_METHODS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--input-lengths", nargs="+", type=int, default=[4096, 8192, 16384, 32768, 49152, 65536, 81920, 98304, 114688, 131072])
    parser.add_argument("--budgets", nargs="+", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.90], help="TopP/TopP32、XAttention、FlexPrefill 的阈值列表。")
    parser.add_argument("--sparsities", nargs="+", type=float, default=[0.50], help="DuoAttention 的 streaming head 比例列表。")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, help="仅指定 log 目录，不产生状态文件。")
    parser.add_argument("--experiment", default="efficiency_vram_matrix")
    parser.add_argument("--task", default="niah_single_1")
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measured-rounds", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--efficiency-backend", action="store_true")
    parser.add_argument("--serial-prefill", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--serial-prefill-verify", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export-csv-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--method", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max-input-length", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--threshold", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--sparsity", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.budgets is not None and args.budget is not None:
        parser.error("--budgets 和 --budget 不能同时使用")
    args.budgets = args.budgets or [args.budget if args.budget is not None else 1024]
    thresholds = [args.threshold] if args.threshold is not None else args.thresholds
    if any(not 0 < value <= 1 for value in thresholds):
        parser.error("threshold 必须位于 (0, 1]")
    sparsities = [args.sparsity] if args.sparsity is not None else args.sparsities
    if any(not 0 <= value <= 1 or value != round(value, 2) for value in sparsities):
        parser.error("sparsity 必须位于 [0, 1]，且最多保留两位小数")
    if any(value < 1 for value in args.budgets + args.batch_sizes + args.input_lengths):
        parser.error("budget、batch 和输入长度必须为正数")
    if args.warmup < 0 or args.measured_rounds < 1 or args.output_length < 2 or args.timeout_seconds <= 0:
        parser.error("warmup≥0，轮数≥1，输出长度≥2，超时>0")
    if not args.worker and (set(args.models) - set(MODEL_CHOICES) or set(args.methods) - set(METHODS)):
        parser.error("未知模型或方法")
    args.output_root = args.output_root.resolve()
    args.state_dir = (args.state_dir or args.output_root).resolve()
    args.data_file = args.data_file.resolve()
    return args


def make_base_args(args: argparse.Namespace) -> argparse.Namespace:
    max_length = getattr(args, "max_input_length", None) or max(args.input_lengths)
    max_seq_length = max_length
    if args.method == "pqcache":
        max_seq_length += max(0, args.output_length - 1024)
    overrides = []
    threshold = getattr(args, "threshold", None)
    if threshold is not None:
        overrides.append(f"fixthreshold={threshold}")
    sparsity = getattr(args, "sparsity", None)
    if sparsity is not None:
        overrides.append(f"sparsity={sparsity}")
    return argparse.Namespace(
        model=args.model,
        method=args.method,
        device="cuda:0",
        batch_size=args.batch_size,
        input_length=max_length,
        context_length=max_length,
        max_seq_length=max_seq_length,
        fixed_output_length=args.output_length,
        task=args.task,
        warmup=args.warmup,
        budget=args.budget if args.budget is not None else method_defaults(args.method)["budget"],
        set=overrides,
        experiment=args.experiment,
        output_root=args.output_root,
        efficiency_backend=args.efficiency_backend,
        serial_prefill=args.serial_prefill,
        serial_prefill_verify=args.serial_prefill_verify,
        threshold=threshold,
        sparsity=sparsity,
        dataset="ruler",
    )


def run_cell(args, options, model, tokenizer, input_device, devices, data_file, input_length):
    """核心测试：固定样本和种子，warmup，测量每一轮，保存生成文本。"""
    import numpy as np
    import torch
    from benchmarks.efficency_vram import generation

    options.input_length = input_length
    options.context_length = input_length
    options.max_seq_length = input_length
    batch_size = args.batch_size
    sample_count = batch_size * args.measured_rounds
    row = report.status(args, input_length, "failed")
    records = []
    predictions = []
    pred_json_path = generation.result_paths(options)["predictions"]
    started = time.monotonic()
    report.notify("start", length=input_length)
    try:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        samples = generation.read_samples(data_file, sample_count)
        batches = list(generation.iter_batches(samples, batch_size))
        for warmup_index in range(options.warmup):
            generation.predict_batch(
                options,
                model,
                tokenizer,
                batches[0],
                input_device,
                devices,
                measured=False,
            )
            print(
                f"Warmup {warmup_index + 1}/{options.warmup} "
                f"b{batch_size} i{input_length}",
                flush=True,
            )

        for batch_index, batch in enumerate(batches):
            tokens, record = generation.predict_batch(
                options,
                model,
                tokenizer,
                batch,
                input_device,
                devices,
                measured=True,
            )
            record["batch_index"] = batch_index
            records.append(record)

            decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            predictions.extend(
                generation.prediction_records(
                    batch,
                    decoded,
                    batch_index + 1,
                    input_length,
                    args.output_length,
                    prefill_mode=record.get("prefill_mode"),
                    budget=args.budget,
                    threshold=getattr(args, "threshold", None),
                    sparsity=getattr(args, "sparsity", None),
                )
            )

            report.print_measurement(batch_index, len(batches), record)

        summary = summarize(records)
        row.update(metrics_from_summary(
            summary,
            batch_size,
            args.measured_rounds,
            args.output_length,
        ))
        row["status"] = "passed"
        row["phase"] = "complete"
    except Exception as error:
        traceback.print_exc()
        row.update(
            status=runtime.error_status(error, batch_size),
            error=str(error).split("\n", 1)[0] or type(error).__name__,
        )
    finally:
        row.update(report.prefill_summary(records))
        row["elapsed_s"] = round(time.monotonic() - started, 2)
        row["round_measurements"] = report.round_floats(report.round_measurement_records(records))
        report.append_predictions(pred_json_path, predictions)
        report.save_result(args, report.round_floats(row))
        gc.collect()
        for device in devices:
            torch.cuda.synchronize(device)
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        report.notify("done", length=input_length)
    return row


def run_test(args):
    """一个进程只加载一次模型，按顺序测试多个输入长度。"""
    import torch
    from loaders import load_model_and_tokenizer
    from benchmarks.efficency_vram import generation

    options = make_base_args(args)
    options._cleanup = None
    try:
        model, tokenizer = load_model_and_tokenizer(options)
        input_device, devices = generation.model_devices(model, args.method)
    except Exception as error:
        traceback.print_exc()
        for length in args.input_lengths:
            report.save_result(args, report.status(args, length, "failed", reason=str(error), phase="loading"))
        return
    try:
        for length in args.input_lengths:
            run_cell(args, options, model, tokenizer, input_device, devices, args.data_file, length)
    finally:
        del model, tokenizer
        if options._cleanup is not None:
            options._cleanup()
        gc.collect()
        torch.cuda.empty_cache()


def worker_command(args, lengths):
    command = [
        sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
        "--model", args.model, "--method", args.method,
        "--batch-size", str(args.batch_size),
        "--input-lengths", *map(str, lengths),
        "--max-input-length", str(max(args.input_lengths)),
        "--data-file", str(args.data_file), "--output-root", str(args.output_root),
        "--experiment", args.experiment, "--task", args.task,
        "--output-length", str(args.output_length), "--warmup", str(args.warmup),
        "--measured-rounds", str(args.measured_rounds),
        "--timeout-seconds", str(args.timeout_seconds),
        "--serial-prefill", args.serial_prefill,
    ]
    if args.budget is not None:
        command.extend(["--budget", str(args.budget)])
    if getattr(args, "threshold", None) is not None:
        command.extend(["--threshold", str(args.threshold)])
    if getattr(args, "sparsity", None) is not None:
        command.extend(["--sparsity", str(args.sparsity)])
    if args.efficiency_backend:
        command.append("--efficiency-backend")
    if args.serial_prefill_verify:
        command.append("--serial-prefill-verify")
    return command


def run_matrix(args):
    """history 判断是否跑过；超时支配集合只在这一次调用中存在。"""
    timeouts = runtime.Timeouts()
    parameter_values = dict.fromkeys(THRESHOLD_METHODS, getattr(args, "thresholds", [0.90]))
    parameter_values["duo-attention"] = getattr(args, "sparsities", [0.50])
    configurations = (
        (model, method, batch_size, value)
        for model, method, batch_size in product(args.models, args.methods, args.batch_sizes)
        for value in parameter_values.get(method, args.budgets)
    )
    for model, method, batch_size, value in configurations:
        group = copy.copy(args)
        group.model, group.method, group.batch_size = model, method, batch_size
        group.threshold = value if method in THRESHOLD_METHODS else None
        group.sparsity = value if method == "duo-attention" else None
        group.budget = None if method in parameter_values else value
        parameter = report.parameter_key(group.budget, group.threshold, group.sparsity)
        group.data_file = Path(str(args.data_file).format(model=model_alias(model), model_alias=model_alias(model)))
        if args.export_csv_only:
            report.save_average(report.method_history_path(args.output_root, model, method))
            continue
        if not group.data_file.is_file():
            raise FileNotFoundError(group.data_file)
        with group.data_file.open() as handle:
            if sum(bool(line.strip()) for line in handle) < batch_size * args.measured_rounds:
                raise ValueError(f"样本不足：{group.data_file}")
        excluded = set()
        while True:
            done = report.completed(group)
            pending = []
            for length in args.input_lengths:
                if (parameter, batch_size, length) in done:
                    continue
                source = timeouts.source(group, length)
                if source is not None:
                    if length not in excluded:
                        source_parameter = report.parameter_label(source[2], source[5], source[6])
                        report.save_result(group, report.status(
                            group, length, "timeout", attempted=False, phase="not_started",
                            reason=f"本次运行被 {source_parameter} b{source[3]} i{source[4]} 的超时支配",
                        ))
                        excluded.add(length)
                else:
                    pending.append(length)
            if not pending:
                break
            timeout = runtime.run_worker(group, worker_command(group, pending), pending)
            if timeout is not None:
                timeouts.add(group, timeout)
            elif not any((parameter, batch_size, length) in report.completed(group) for length in pending):
                raise RuntimeError("worker 未产生结果；详见 log")


def main():
    args = parse_args()
    if args.dry_run:
        values = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
        print(json.dumps(values, ensure_ascii=False, indent=2))
    elif args.worker:
        run_test(args)
    else:
        run_matrix(args)


if __name__ == "__main__":
    main()
