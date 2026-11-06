#!/usr/bin/env python3
"""RULER prediction with optional efficiency and CUDA memory measurements."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import random
import sys
import uuid
from contextlib import ExitStack, nullcontext
from importlib.metadata import version
from itertools import islice
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common import METHODS, MODEL_CHOICES, method_defaults, model_alias, parse_set
from benchmarks.efficency_vram.tools.efficiency import EfficiencyMeter
from benchmarks.efficency_vram.tools.memory import MemoryMonitor
from benchmarks.efficency_vram.tools.result_store import append_result
from benchmarks.efficency_vram.tools.reporting import summarize
from loaders import load_model_and_tokenizer


SINK_KEYS = {"streaming": "start_size", "duo-attention": "sink_size", "clusterkv": "sink",
             "pqcache": "sink_size", "topk": "sink_size", "topk32": "sink_size"}
LOCAL_KEYS = {"streaming": "recent_size", "duo-attention": "recent_size", "h2o": "recent_size",
              "keyformer": "recent_size", "pqcache": "recent_size", "topk": "recent_size", "topk32": "recent_size"}
NATIVE_METHODS = {"retroinfer", "magicpig"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="full_attention")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="llama-3.1-8b")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1)
    parser.add_argument("--input-length", "--input_length", type=int, default=4096,
                        help="Exact input token count after truncating or extending each sample")
    parser.add_argument("--fixed-output-length", "--fixed_output_length", type=int, default=128)
    parser.add_argument("--measurement", choices=("none", "efficiency", "vram", "both"), default="both")
    parser.add_argument("--benchmark", type=str.lower, choices=("ruler",), default="ruler")
    parser.add_argument("--task", default="niah_single_1")
    parser.add_argument("--context-length", "--context_length", type=int)
    parser.add_argument("--data-root", "--data_root", type=Path,
                        default=REPO_ROOT / "benchmarks/ruler/benchmark_root")
    parser.add_argument("--data-dir", "--data_dir", type=Path,
                        help="RULER JSONL file, task directory, or data directory")
    parser.add_argument("--num-samples", "--num_samples", "--limit", type=int, default=8)
    parser.add_argument("--num-batches", "--num_batches", type=int)
    parser.add_argument("--allow-partial-batch", action="store_true")
    parser.add_argument("--short-input", choices=("repeat", "error"), default="repeat",
                        help="How to handle inputs shorter than input_length (default: repeat)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--sink-token", "--sink_token", type=int)
    parser.add_argument("--local-window-token", "--local_window_token", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--experiment", default="default")
    parser.add_argument("--output-root", "--save-dir", "--save_dir", type=Path,
                        default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="兼容旧脚本；统一 pred.json 始终追加新记录。",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show data selection without loading weights")
    args = parser.parse_args(argv)
    args.context_length = args.context_length or args.input_length
    configure_method(args)
    return args


def configure_method(args):
    defaults = {} if args.method == "minference" else method_defaults(args.method)
    overrides = parse_set(args.set, args.method)
    for value, keys in ((args.sink_token, SINK_KEYS), (args.local_window_token, LOCAL_KEYS)):
        if value is not None:
            overrides[keys[args.method]] = value
    if args.method == "streaming" and args.budget is None and "recent_size" in overrides:
        args.budget = overrides.get("start_size", defaults["start_size"]) + overrides["recent_size"]
    if args.budget is None:
        args.budget = defaults.get("budget", 1024)
    if args.method == "retroinfer":
        for key, value in overrides.items():
            setattr(args, key, value)
    args.set = [f"{key}={json.dumps(value)}" for key, value in sorted(overrides.items())]
    args.max_seq_length = max(args.context_length, args.input_length)
    if args.method == "pqcache":
        # The loader adds a 1024-token generation reserve to this hint.
        args.max_seq_length = args.input_length + max(0, args.fixed_output_length - 1024)
    args.dataset = "ruler"


def resolve_data_file(args):
    root = args.data_dir or (args.data_root / model_alias(args.model) / "synthetic" /
                             str(args.context_length) / "data")
    if root.is_file():
        return root.resolve()
    if (root / "validation.jsonl").is_file():
        return (root / "validation.jsonl").resolve()
    return (root / args.task / "validation.jsonl").resolve()


def read_samples(path, limit=None):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in islice(handle, limit)]


def iter_batches(samples, batch_size, allow_partial=False):
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        if len(batch) == batch_size or allow_partial:
            yield batch


def normalize_tokens(tokens, length, short_input="repeat"):
    if len(tokens) == length:
        return tokens, "unchanged"
    if len(tokens) > length:
        first = length // 2
        return tokens[:first] + tokens[-(length - first):], "middle_truncated"
    if short_input == "error":
        raise ValueError(f"RULER input has {len(tokens)} tokens, less than requested {length}")
    missing = length - len(tokens)
    filler = (tokens * math.ceil(missing / len(tokens)))[:missing]
    split = len(tokens) // 2
    return tokens[:split] + filler + tokens[split:], "repeated"


def prepare_batch(samples, tokenizer, input_length, device, short_input="repeat"):
    # RULER inputs already include their prompt template.
    encoded = tokenizer([sample["input"] for sample in samples], add_special_tokens=False,
                        truncation=False, padding=False)["input_ids"]
    normalized = [normalize_tokens(row, input_length, short_input) for row in encoded]
    input_ids = torch.tensor([row for row, _ in normalized], dtype=torch.long, device=device)
    metadata = {"indices": [sample["index"] for sample in samples],
                "original_input_tokens": [len(row) for row in encoded],
                "input_adjustments": [adjustment for _, adjustment in normalized],
                "input_tokens": [input_length] * len(samples)}
    return input_ids, torch.ones_like(input_ids), metadata


def model_devices(model, method):
    if method == "retroinfer":
        devices = {torch.device(layer.device) for layer in model.layers}
        input_device = torch.device(model.layers[0].device)
    elif method == "magicpig":
        input_device = torch.device(model.device)
        devices = {input_device}
    else:
        input_device = model.get_input_embeddings().weight.device
        devices = {parameter.device for parameter in model.parameters()}
    return input_device, tuple(sorted((device for device in devices if device.type == "cuda"), key=str))


def prepare_sample(args, model, input_length):
    if args.method == "clusterkv":
        cluster_args = args._method_args
        cluster_args.nlist = max(args._base_nlist, math.ceil(input_length / 80))
        for module in model.modules():
            if hasattr(module, "flash_forward"):
                args._cluster_attention.apply_cluster_config(module, cluster_args)
                module.token_budget = cluster_args.token_budget
                module.cluster_cache = None
        args._cluster_attention.cluster_reset(model)
    elif args.method == "retroinfer":
        model.kv_cache = None


def finish_sample(args, model):
    if args.method == "clusterkv":
        args._cluster_attention.cluster_reset(model)
    elif args.method == "retroinfer":
        model.kv_cache = None


@torch.inference_mode()
def generate(
    model,
    input_ids,
    attention_mask,
    output_length,
    method="full_attention",
    step_context=nullcontext,
    decode_start_callback=None,
):
    """Fixed-length greedy decoding; step_context optionally observes each forward."""
    if method in NATIVE_METHODS:
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                      max_new_tokens=output_length, do_sample=False, num_beams=1)
        if method == "retroinfer":
            output = model.generate(
                **kwargs,
                fixed_output_length=output_length,
                decode_start_callback=decode_start_callback,
            )
        else:
            previous = model.fixed_output_length
            try:
                model.fixed_output_length = output_length
                output = model.generate(**kwargs)
            finally:
                model.fixed_output_length = previous
        return output[:, input_ids.shape[1]:]

    model_kwargs = dict(attention_mask=attention_mask, use_cache=True,
                        cache_position=torch.arange(input_ids.shape[1], device=input_ids.device))
    parameters = inspect.signature(model.forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in parameters:
            model_kwargs[name] = 1
            break
    sequence, generated = input_ids, []
    for step in range(output_length):
        if step == 1 and decode_start_callback is not None:
            decode_start_callback()
        model_inputs = model.prepare_inputs_for_generation(sequence, **model_kwargs)
        with step_context():
            outputs = model(**model_inputs, return_dict=True)
            next_tokens = outputs.logits[:, -1, :].argmax(dim=-1).to(input_ids.device)
        generated.append(next_tokens)
        if step + 1 < output_length:
            model_kwargs = model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False)
            sequence = torch.cat([sequence, next_tokens[:, None]], dim=1)
        del outputs
    return torch.stack(generated, dim=1)


def predict_batch(args, model, tokenizer, batch, input_device, devices, measured=True):
    input_ids, attention_mask, record = prepare_batch(batch, tokenizer, args.input_length, input_device, args.short_input)
    monitors = []
    if measured and args.measurement in {"vram", "both"}:
        memory_monitor = MemoryMonitor(devices)
        monitors.append(memory_monitor)
    else:
        memory_monitor = None
    timer = None
    if measured and args.measurement in {"efficiency", "both"}:
        timer = EfficiencyMeter(devices, len(batch), args.fixed_output_length)
        monitors.append(timer)
    prepare_sample(args, model, args.input_length)
    try:
        with ExitStack() as stack:
            for monitor in monitors:
                stack.enter_context(monitor)
            tokens = generate(model, input_ids, attention_mask, args.fixed_output_length,
                              args.method, timer.step if timer else nullcontext,
                              memory_monitor.begin_decode_phase if memory_monitor else None)
        for monitor in monitors:
            record.update(monitor.metrics())
        record.update(backend="native_generate" if args.method in NATIVE_METHODS else "forward",
                      batch_size=len(batch), input_length=args.input_length, fixed_output_length=args.fixed_output_length)
        return tokens.cpu(), record
    finally:
        finish_sample(args, model)
        del input_ids, attention_mask
        gc.collect()
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()


def result_paths(args, data_file):
    del data_file
    return {"predictions": args.output_root / "pred.json"}


def run(args):
    data_file = resolve_data_file(args)
    limit = None if args.num_samples == -1 else args.num_samples
    if args.num_batches is not None:
        limit = min(limit, args.num_batches * args.batch_size) if limit is not None else args.num_batches * args.batch_size
    samples = read_samples(data_file, limit)
    batches = list(iter_batches(samples, args.batch_size, args.allow_partial_batch))
    paths = result_paths(args, data_file)
    metadata = {"arguments": {key: value for key, value in vars(args).items() if not key.startswith("_")},
                "data_file": str(data_file), "selected_samples": len(samples),
                "dropped_samples": len(samples) - sum(map(len, batches)), "num_batches": len(batches),
                "paths": paths}
    print(f"RULER: {data_file}\nBatches: {len(batches)}\nPredictions: {paths['predictions']}", flush=True)
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
        return paths
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args._cleanup = None
    args._method_args = argparse.Namespace(**parse_set(args.set, args.method))
    records = []
    predictions = []
    try:
        model, tokenizer = load_model_and_tokenizer(args)
        input_device, devices = model_devices(model, args.method)
        metadata.update(status="running", backend="native_generate" if args.method in NATIVE_METHODS else "forward",
                        model_path=args.model_path,
                        effective_method_config={**vars(args._method_args), "benchmark_budget": args.budget},
                        torch_version=torch.__version__, transformers_version=version("transformers"),
                        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                        devices={str(device): torch.cuda.get_device_name(device) for device in devices})
        for iteration in range(args.warmup):
            predict_batch(args, model, tokenizer, batches[0], input_device, devices, measured=False)
            print(f"Warmup {iteration + 1}/{args.warmup}", flush=True)

        for batch_index, batch in enumerate(batches):
            tokens, record = predict_batch(args, model, tokenizer, batch, input_device, devices)
            record.update(batch_index=batch_index, method=args.method, model=args.model, budget=args.budget,
                          task=args.task, measurement=args.measurement, configured_batch_size=args.batch_size)
            records.append(record)

            if args.save_predictions or args.measurement == "none":
                decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True)
                for sample, ids, prediction in zip(batch, tokens.tolist(), decoded):
                    predictions.append({
                        "index": sample["index"],
                        "pred": prediction,
                        "output_token_ids": ids,
                    })

            print(f"Batch {batch_index + 1}/{len(batches)}: ttft_s={record.get('ttft_s')}, "
                  f"tpot_ms={record.get('tpot_ms')}, peak_allocated_mib={record.get('peak_allocated_mib')}", flush=True)

        metadata["status"] = "complete"
        result = {
            "result_id": uuid.uuid4().hex,
            "model": args.model,
            "model_alias": model_alias(args.model),
            "method": args.method,
            "task": args.task,
            "batch_size": args.batch_size,
            "input_length": args.input_length,
            "output_length": args.fixed_output_length,
            "status": "passed",
            "measurements": records,
            "summary": {
                **summarize(records),
                "method": args.method,
                "model": args.model,
                "task": args.task,
                "measurement": args.measurement,
            },
            "predictions": predictions,
            "metadata": metadata,
        }
        append_result(paths["predictions"], result)
    finally:
        if args._cleanup is not None:
            args._cleanup()

    print(f"结果已追加到：{paths['predictions']}", flush=True)
    return paths


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
