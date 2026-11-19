"""CPU 内存基准的 RULER 批处理与定长贪心生成。"""

import gc
import json
import math
from contextlib import nullcontext
from itertools import islice

import torch

from benchmarks.common import PQ_METHODS
from benchmarks.memory.serial_prefill import (
    ADAPTIVE_METHODS,
    SERIAL_CACHE_METHODS,
    SerialPrefillUnsupported,
    generate_serial_prefill,
    logits_keep_kwargs,
    serial_adaptive_compression,
    serial_prefill_supported,
)


NATIVE_GENERATE_METHODS = {"retroinfer", "magicpig"}
NATIVE_KERNEL_METHODS = {"quest", "clusterkv"}


def read_batches(path, batch_size, sample_count):
    with path.open(encoding="utf-8") as handle:
        samples = [json.loads(line) for line in islice(handle, sample_count)]
    if len(samples) < sample_count:
        raise ValueError(f"{path} 只有 {len(samples)} 条样本，需要 {sample_count} 条。")
    return [samples[start:start + batch_size] for start in range(0, sample_count, batch_size)]


def normalize_tokens(tokens, length):
    if len(tokens) == length:
        return tokens
    if len(tokens) > length:
        first = length // 2
        return tokens[:first] + tokens[-(length - first):]
    missing = length - len(tokens)
    filler = (tokens * math.ceil(missing / len(tokens)))[:missing]
    split = len(tokens) // 2
    return tokens[:split] + filler + tokens[split:]


def prepare_batch(batch, tokenizer, input_length, device):
    encoded = tokenizer(
        [sample["input"] for sample in batch], add_special_tokens=False,
        truncation=False, padding=False,
    )["input_ids"]
    input_ids = torch.tensor([normalize_tokens(row, input_length) for row in encoded],
                             dtype=torch.long, device=device)
    return input_ids, torch.ones_like(input_ids)


def model_devices(model, method):
    if method == "retroinfer":
        input_device = torch.device(model.layers[0].device)
        devices = {torch.device(layer.device) for layer in model.layers}
    elif method == "magicpig":
        input_device = torch.device(model.device)
        devices = {input_device}
    else:
        input_device = model.get_input_embeddings().weight.device
        devices = {parameter.device for parameter in model.parameters()}
    return input_device, tuple(sorted((device for device in devices if device.type == "cuda"), key=str))


def select_prefill_mode(args, batch_size):
    method = args.method
    if args.serial_prefill == "off" or batch_size < 2:
        return "batched"
    if method in NATIVE_GENERATE_METHODS or method in PQ_METHODS:
        return "batched"
    if args.efficiency_backend and method in NATIVE_KERNEL_METHODS:
        return "batched"
    if method in ADAPTIVE_METHODS:
        return "batched" if "glm" in args.model.lower() else "serial_kv_compression"
    if args.serial_prefill == "on" or method in SERIAL_CACHE_METHODS:
        return "serial"
    return "batched"


def clear_sample(args, model):
    if args.efficiency_backend and args.method == "quest":
        model.quest_clear()
    elif args.efficiency_backend and args.method == "clusterkv":
        model.clusterkv_clear()
    elif args.method == "clusterkv":
        args._cluster_attention.cluster_reset(model)
    elif args.method == "retroinfer":
        model.kv_cache = None


def prepare_sample(args, model):
    if args.method == "clusterkv" and not args.efficiency_backend:
        cluster_args = args._method_args
        cluster_args.nlist = max(args._base_nlist, math.ceil(args.input_length / 80))
        for module in model.modules():
            if hasattr(module, "flash_forward"):
                args._cluster_attention.apply_cluster_config(module, cluster_args)
                module.token_budget = cluster_args.token_budget
                module.cluster_cache = None
    clear_sample(args, model)


def generate_native(model, method, input_ids, attention_mask, output_length):
    kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                  max_new_tokens=output_length, do_sample=False, num_beams=1)
    if method == "retroinfer":
        output = model.generate(**kwargs, fixed_output_length=output_length)
    else:
        previous = model.fixed_output_length
        try:
            model.fixed_output_length = output_length
            output = model.generate(**kwargs)
        finally:
            model.fixed_output_length = previous
    return output[:, input_ids.shape[1]:]


def generate_kernel(model, method, input_ids, output_length):
    sequence = input_ids
    cache = None
    generated = []
    for _ in range(output_length):
        if method == "quest":
            outputs = model(input_ids=sequence, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = outputs.past_key_values
            logits = outputs.logits
        else:
            outputs = model(input_ids=sequence)
            logits = outputs[0]
        next_tokens = logits[:, -1, :].argmax(dim=-1).to(input_ids.device)
        generated.append(next_tokens)
        sequence = next_tokens[:, None]
        del outputs, logits
    return torch.stack(generated, dim=1)


def generate_batch(args, model, input_ids, attention_mask):
    output_length = args.fixed_output_length
    method = args.method
    if method in NATIVE_GENERATE_METHODS:
        return generate_native(model, method, input_ids, attention_mask, output_length)
    if args.efficiency_backend and method in NATIVE_KERNEL_METHODS:
        return generate_kernel(model, method, input_ids, output_length)

    model_kwargs = dict(attention_mask=attention_mask, use_cache=True,
                        cache_position=torch.arange(input_ids.shape[1], device=input_ids.device))
    model_kwargs.update(logits_keep_kwargs(model))
    sequence, generated = input_ids, []
    for step in range(output_length):
        model_inputs = model.prepare_inputs_for_generation(sequence, **model_kwargs)
        outputs = model(**model_inputs, return_dict=True)
        next_tokens = outputs.logits[:, -1, :].argmax(dim=-1).to(input_ids.device)
        generated.append(next_tokens)
        if step + 1 < output_length:
            model_kwargs = model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False)
            sequence = torch.cat([sequence, next_tokens[:, None]], dim=1)
        del outputs
    return torch.stack(generated, dim=1)


@torch.inference_mode()
def predict_batch(args, model, tokenizer, batch, input_device, devices):
    input_ids, attention_mask = prepare_batch(batch, tokenizer, args.input_length, input_device)
    mode = select_prefill_mode(args, len(batch))
    prepare_sample(args, model)
    try:
        if mode == "serial":
            if serial_prefill_supported(model, args.method, input_ids, attention_mask):
                try:
                    tokens = generate_serial_prefill(
                        model, args.method, input_ids, attention_mask, args.fixed_output_length,
                    )
                except SerialPrefillUnsupported:
                    pass
                else:
                    return tokens.cpu(), mode
            # 不同样本的 KV 结构无法合并时，重置本批状态再执行批量 prefill。
            prepare_sample(args, model)
            mode = "batched"

        compression = serial_adaptive_compression(model, args.method) if mode == "serial_kv_compression" else nullcontext()
        with compression:
            tokens = generate_batch(args, model, input_ids, attention_mask)
        return tokens.cpu(), mode
    finally:
        clear_sample(args, model)
        del input_ids, attention_mask
        gc.collect()
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
