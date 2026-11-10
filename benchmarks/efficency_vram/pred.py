"""长上下文效率与显存 runner 使用的预测函数。"""

from __future__ import annotations

import gc
import inspect
import json
import math
from contextlib import ExitStack, nullcontext
from itertools import islice
from pathlib import Path

import torch

from benchmarks.common import model_alias
from benchmarks.efficency_vram.tools.efficiency import EfficiencyMeter
from benchmarks.efficency_vram.tools.memory import MemoryMonitor
from benchmarks.efficency_vram.tools.result_store import make_prediction_record


NATIVE_METHODS = {"retroinfer", "magicpig"}
# --efficiency-backend 下使用 Quest 分页 KV 和 ClusterKV kernel cache；
# 这两个实现按单请求方式调用。
NATIVE_EFFICIENCY_METHODS = {"quest", "clusterkv"}


def read_samples(path, limit):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in islice(handle, limit)]


def iter_batches(samples, batch_size):
    for start in range(0, len(samples), batch_size):
        yield samples[start:start + batch_size]


def normalize_tokens(tokens, length):
    if len(tokens) == length:
        return tokens, "unchanged"
    if len(tokens) > length:
        first = length // 2
        return tokens[:first] + tokens[-(length - first):], "middle_truncated"
    missing = length - len(tokens)
    filler = (tokens * math.ceil(missing / len(tokens)))[:missing]
    split = len(tokens) // 2
    return tokens[:split] + filler + tokens[split:], "repeated"


def prepare_batch(samples, tokenizer, input_length, device):
    # RULER inputs already include their prompt template.
    encoded = tokenizer([sample["input"] for sample in samples], add_special_tokens=False,
                        truncation=False, padding=False)["input_ids"]
    normalized = [normalize_tokens(row, input_length) for row in encoded]
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
    if args.efficiency_backend and args.method == "quest":
        model.quest_clear()
    elif args.efficiency_backend and args.method == "clusterkv":
        model.clusterkv_clear()
    elif args.method == "clusterkv":
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
    if args.efficiency_backend and args.method == "quest":
        model.quest_clear()
    elif args.efficiency_backend and args.method == "clusterkv":
        model.clusterkv_clear()
    elif args.method == "clusterkv":
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
    efficiency_backend=False,
):
    """定长贪心解码；可通过 step_context 观测每次前向。"""
    if efficiency_backend and method == "quest":
        return generate_quest(model, input_ids, output_length, step_context, decode_start_callback)
    if efficiency_backend and method == "clusterkv":
        return generate_clusterkv(model, input_ids, output_length, step_context, decode_start_callback)
    if method in NATIVE_METHODS:
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                      max_new_tokens=output_length, do_sample=False, num_beams=1)
        if method == "retroinfer":
            output = model.generate(
                **kwargs,
                fixed_output_length=output_length,
                decode_start_callback=decode_start_callback,
                step_context=step_context,
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


@torch.inference_mode()
def generate_quest(model, input_ids, output_length, step_context=nullcontext,
                   decode_start_callback=None):
    """通过 Quest 原生分页 KV 模型进行贪心解码。"""
    model.quest_clear()
    past_key_values = None
    sequence = input_ids
    generated = []
    for step in range(output_length):
        if step == 1 and decode_start_callback is not None:
            decode_start_callback()
        with step_context():
            outputs = model(input_ids=sequence, past_key_values=past_key_values,
                            use_cache=True, logits_to_keep=1)
        past_key_values = outputs.past_key_values
        next_tokens = outputs.logits[:, -1, :].argmax(dim=-1).to(input_ids.device)
        generated.append(next_tokens)
        sequence = next_tokens[:, None]
        del outputs
    model.quest_clear()
    return torch.stack(generated, dim=1)


@torch.inference_mode()
def generate_clusterkv(model, input_ids, output_length, step_context=nullcontext,
                       decode_start_callback=None):
    """通过 ClusterKV 原生 kernel cache 进行贪心解码。"""
    model.clusterkv_clear()
    sequence = input_ids
    generated = []
    for step in range(output_length):
        if step == 1 and decode_start_callback is not None:
            decode_start_callback()
        with step_context():
            outputs = model(input_ids=sequence)
        logits = outputs[0]
        next_tokens = logits[:, -1, :].argmax(dim=-1).to(input_ids.device)
        generated.append(next_tokens)
        sequence = next_tokens[:, None]
        del outputs
    model.clusterkv_clear()
    return torch.stack(generated, dim=1)


def predict_batch(args, model, tokenizer, batch, input_device, devices, measured=True):
    input_ids, attention_mask, record = prepare_batch(
        batch,
        tokenizer,
        args.input_length,
        input_device,
    )
    memory_monitor = MemoryMonitor(devices) if measured else None
    timer = EfficiencyMeter(devices, len(batch), args.fixed_output_length) if measured else None
    monitors = [monitor for monitor in (memory_monitor, timer) if monitor is not None]
    prepare_sample(args, model, args.input_length)
    try:
        with ExitStack() as stack:
            for monitor in monitors:
                stack.enter_context(monitor)
            tokens = generate(model, input_ids, attention_mask, args.fixed_output_length,
                              args.method, timer.step if timer else nullcontext,
                              memory_monitor.begin_decode_phase if memory_monitor else None,
                              efficiency_backend=args.efficiency_backend)
        for monitor in monitors:
            record.update(monitor.metrics())
        native_backend = args.method in NATIVE_METHODS or (
            args.efficiency_backend and args.method in NATIVE_EFFICIENCY_METHODS)
        record.update(backend="native_generate" if native_backend else "forward",
                      batch_size=len(batch), input_length=args.input_length, fixed_output_length=args.fixed_output_length)
        return tokens.cpu(), record
    finally:
        finish_sample(args, model)
        del input_ids, attention_mask
        gc.collect()
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()


def result_paths(args):
    predictions_dir = (
        args.output_root
        / model_alias(args.model)
        / args.experiment
        / args.method
        / args.task
    )
    return {"predictions": predictions_dir / "pred.json"}


def prediction_records(batch, decoded, round_index, input_length, output_length):
    """生成用于核对预测文本的最小记录。"""
    batch_size = len(batch)
    return [
        make_prediction_record(
            sample_index=sample["index"],
            output=output,
            round_index=round_index,
            batch_size=batch_size,
            input_length=input_length,
            output_length=output_length,
        )
        for sample, output in zip(batch, decoded)
    ]
