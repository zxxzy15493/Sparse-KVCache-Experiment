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

# 串行 prefill：batch 测试仍然按 batch decode，但 prefill 逐条执行（与 RetroInfer 相同），
# 因此 prefill 阶段的显存只与单条序列有关，TTFT 记录整个 batch 的串行 prefill 总时长。
# 这些方法在 prefill 后只把 KV 留在 cache 对象里，逐条 prefill 后合并 cache 即可批量 decode。
SERIAL_PREFILL_CACHE_METHODS = frozenset({
    "full_attention", "snapkv", "streaming", "pyramidkv", "headkv", "adakv",
    "duo-attention", "flexprefill", "minference", "sparq", "xattention",
})
# 这些方法另外在 attention 模块上保存逐条样本的分数，合并 cache 之外还要合并这些张量；
# 默认不开启，需要 --serial-prefill on 显式启用（cakekv 的逐层预算随单条样本分数变化，
# 合并长度通常不一致，只会自动回退到批量 prefill）。
SERIAL_PREFILL_STATE_METHODS = {
    "h2o": (("kv_cache", "hh_score"),),
    "keyformer": (("kv_cache", "key_score"), ("kv_cache_evictor", "key_score")),
    "cakekv": (),
}
# 保持批量 prefill：自带 generate 的原生方法，以及把 KV 副本或分数放在模块/CPU 上的 PQCache 家族。
SERIAL_PREFILL_BATCH_ONLY_METHODS = frozenset(NATIVE_METHODS) | frozenset({
    "pqcache", "topk", "topk32", "topp", "topp32",
})

# 每个 worker 进程内只探测一次：None 表示可以合并，字符串表示回退原因。
_SERIAL_PREFILL_SUPPORT: dict[tuple, str | None] = {}
_SERIAL_PREFILL_VERIFIED: set[tuple] = set()


class SerialPrefillUnsupported(RuntimeError):
    """逐条 prefill 得到的 cache 不能按 batch 维合并。"""


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

def serial_prefill_mode(args, method, batch_size, native_backend=False):
    """决定这一次 batch 的 prefill 走逐条串行还是批量。"""
    policy = getattr(args, "serial_prefill", "auto")
    if policy == "off" or batch_size < 2 or native_backend:
        return "batched"
    if method in SERIAL_PREFILL_BATCH_ONLY_METHODS:
        return "batched"
    if method in SERIAL_PREFILL_CACHE_METHODS or policy == "on":
        return "serial"
    return "batched"


def logits_keep_kwargs(model):
    """只取最后一个位置的 logits，省掉 prefill 位置的输出。"""
    parameters = inspect.signature(model.forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in parameters:
            return {name: 1}
    return {}


def merge_sample_caches(caches):
    """把逐条 prefill 得到的 cache 沿 batch 维合并；结构不支持时返回 None。

    逐层合并并立即释放该层逐条张量，合并期的显存峰值约等于最终 cache 加一层临时副本。
    """
    if not caches:
        return None
    if len(caches) == 1:
        return caches[0]
    first = caches[0]
    layers = getattr(first, "layers", None)
    if layers is not None:
        if not layers or any(len(getattr(cache, "layers", ())) != len(layers) for cache in caches):
            return None
        for index, layer in enumerate(layers):
            keys = [cache.layers[index].keys for cache in caches]
            values = [cache.layers[index].values for cache in caches]
            if not _mergeable(keys, values):
                return None
            merged_keys = _stack_batch(
                keys, lambda position, index=index: setattr(caches[position].layers[index], "keys", None))
            merged_values = _stack_batch(
                values, lambda position, index=index: setattr(caches[position].layers[index], "values", None))
            layer.keys = merged_keys
            layer.values = merged_values
            layer.is_initialized = True
        return first
    key_cache = getattr(first, "key_cache", None)
    value_cache = getattr(first, "value_cache", None)
    if isinstance(key_cache, list) and isinstance(value_cache, list) and key_cache:
        if any(len(cache.key_cache) != len(key_cache) for cache in caches):
            return None
        for index in range(len(key_cache)):
            keys = [cache.key_cache[index] for cache in caches]
            values = [cache.value_cache[index] for cache in caches]
            if not _mergeable(keys, values):
                return None
            merged_keys = _stack_batch(
                keys, lambda position, index=index: caches[position].key_cache.__setitem__(index, None))
            merged_values = _stack_batch(
                values, lambda position, index=index: caches[position].value_cache.__setitem__(index, None))
            key_cache[index] = merged_keys
            value_cache[index] = merged_values
        return first
    return None


def _stack_batch(parts, release=None):
    """沿 batch 维把逐条 [1, ...] cache 张量合并成 [B, ...]，边拷贝边释放逐条张量。"""
    merged = torch.empty((len(parts),) + tuple(parts[0].shape[1:]),
                         dtype=parts[0].dtype, device=parts[0].device)
    for position, part in enumerate(parts):
        merged[position].copy_(part[0])
        parts[position] = None
        if release is not None:
            release(position)
    return merged


def _mergeable(keys, values):
    if any(not torch.is_tensor(key) for key in keys):
        return False
    if any(not torch.is_tensor(value) for value in values):
        return False
    shapes = [tuple(key.shape) for key in keys]
    return len(set(shapes)) == 1 and keys[0].shape[0] == 1


def _read_attribute(module, path):
    owner = module
    for name in path[:-1]:
        owner = getattr(owner, name, None)
        if owner is None:
            return None
    return getattr(owner, path[-1], None)


def _write_attribute(module, path, value):
    owner = module
    for name in path[:-1]:
        owner = getattr(owner, name, None)
        if owner is None:
            return
    setattr(owner, path[-1], value)


class SampleStateTracker:
    """收集逐条 prefill 后写在 attention 模块上的分数，批量 decode 前按 batch 维合并。"""

    def __init__(self, model, attribute_paths):
        self.attribute_paths = tuple(attribute_paths)
        self.modules = dict(model.named_modules())
        self.parts = {}

    def capture(self, index):
        for name, module in self.modules.items():
            for path in self.attribute_paths:
                value = _read_attribute(module, path)
                if torch.is_tensor(value):
                    self.parts.setdefault((name, path), {})[index] = value

    def merge(self, batch_size):
        for (name, path), parts in self.parts.items():
            if len(parts) != batch_size:
                continue
            values = [parts[index] for index in range(batch_size)]
            if any(value.shape[0] != 1 for value in values):
                continue
            if len({value.shape[1:] for value in values}) != 1:
                continue
            _write_attribute(self.modules[name], path, torch.cat(values, dim=0))
            parts.clear()
        self.parts.clear()


def serial_prefill_supported(model, input_ids, attention_mask, mode, method):
    """用两条真实样本探测该方法能否合并逐条 cache；结果按进程缓存，失败返回原因。"""
    key = (method, mode)
    if key in _SERIAL_PREFILL_SUPPORT:
        return _SERIAL_PREFILL_SUPPORT[key]
    cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
    logits_kwargs = logits_keep_kwargs(model)
    caches, reason = [], None
    try:
        with torch.inference_mode():
            for index in range(2):
                outputs = model(
                    input_ids=input_ids[index:index + 1],
                    attention_mask=attention_mask[index:index + 1],
                    use_cache=True,
                    cache_position=cache_position,
                    return_dict=True,
                    **logits_kwargs,
                )
                caches.append(outputs.past_key_values)
                del outputs
        if merge_sample_caches(caches) is None:
            reason = "该方法的 cache 结构不支持按 batch 维合并"
    except Exception as error:  # noqa: BLE001 - 探测失败一律回退批量 prefill
        reason = f"逐条 prefill 探测失败：{type(error).__name__}: {error}"
    finally:
        caches.clear()
        del caches
        gc.collect()
        if input_ids.device.type == "cuda":
            torch.cuda.empty_cache()
    _SERIAL_PREFILL_SUPPORT[key] = reason
    return reason


@torch.inference_mode()
def generate_serial_prefill(
    model,
    input_ids,
    attention_mask,
    output_length,
    method="full_attention",
    step_context=nullcontext,
    decode_start_callback=None,
    state_paths=None,
):
    """逐条 prefill 并合并 cache，再按 batch decode；token 与批量 prefill 一致。"""
    batch_size, prompt_length = input_ids.shape
    device = input_ids.device
    cache_position = torch.arange(prompt_length, device=device)
    logits_kwargs = logits_keep_kwargs(model)
    tracker = SampleStateTracker(model, state_paths) if state_paths else None
    caches, first_tokens = [], []
    with step_context():
        # 整个 batch 的串行 prefill 记为一步，TTFT 是各条 prefill 之和。
        for index in range(batch_size):
            outputs = model(
                input_ids=input_ids[index:index + 1],
                attention_mask=attention_mask[index:index + 1],
                use_cache=True,
                cache_position=cache_position,
                return_dict=True,
                **logits_kwargs,
            )
            caches.append(outputs.past_key_values)
            first_tokens.append(outputs.logits[:, -1, :].argmax(dim=-1))
            if tracker is not None:
                tracker.capture(index)
            del outputs
    cache = merge_sample_caches(caches)
    caches.clear()
    del caches
    if cache is None:
        raise SerialPrefillUnsupported("逐条 prefill 的 cache 无法按 batch 维合并")
    if tracker is not None:
        tracker.merge(batch_size)
    first = torch.cat(first_tokens, dim=0).to(device)
    first_tokens.clear()
    generated = [first]
    if decode_start_callback is not None:
        decode_start_callback()
    sequence = torch.cat([input_ids, first[:, None]], dim=1)
    mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1)
    for step in range(1, output_length):
        cache_position = torch.arange(sequence.shape[1] - 1, sequence.shape[1], device=device)
        model_inputs = model.prepare_inputs_for_generation(
            sequence,
            past_key_values=cache,
            attention_mask=mask,
            cache_position=cache_position,
            use_cache=True,
            **logits_kwargs,
        )
        with step_context():
            outputs = model(**model_inputs, return_dict=True)
            next_tokens = outputs.logits[:, -1, :].argmax(dim=-1).to(device)
        generated.append(next_tokens)
        del outputs
        if step + 1 < output_length:
            sequence = torch.cat([sequence, next_tokens[:, None]], dim=1)
            mask = torch.cat([mask, mask.new_ones((batch_size, 1))], dim=1)
    return torch.stack(generated, dim=1)


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
    model_kwargs.update(logits_keep_kwargs(model))
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


def verify_serial_prefill(args, model, input_ids, attention_mask, tokens):
    """用一次未计量的批量 prefill 解码做对照，核对串行 prefill 的 token 是否一致。"""
    key = (args.method, input_ids.shape[1], args.input_length)
    if key in _SERIAL_PREFILL_VERIFIED:
        return {}
    _SERIAL_PREFILL_VERIFIED.add(key)
    prepare_sample(args, model, args.input_length)
    try:
        with torch.inference_mode():
            reference = generate(model, input_ids, attention_mask, args.fixed_output_length,
                                 args.method).cpu()
    except torch.cuda.OutOfMemoryError as error:
        # 对照需要批量 prefill 的显存，OOM 恰好说明串行 prefill 有用。
        return {"serial_prefill_verify": "reference_oom",
                "serial_prefill_verify_error": str(error).splitlines()[0][:120]}
    except Exception as error:  # noqa: BLE001 - 对照失败不影响已有测量
        return {"serial_prefill_verify": "reference_error",
                "serial_prefill_verify_error": f"{type(error).__name__}: {error}"[:120]}
    finally:
        gc.collect()
        if input_ids.device.type == "cuda":
            torch.cuda.empty_cache()
    mismatch = int((reference != tokens.cpu()).sum())
    return {"serial_prefill_verify": "match" if mismatch == 0 else "mismatch",
            "serial_prefill_mismatch_tokens": mismatch}


def run_generation(args, model, input_ids, attention_mask, devices, measured, mode, state_paths):
    """按 prefill 模式跑一次生成，返回 (tokens, 监控器列表)。"""
    memory_monitor = MemoryMonitor(devices) if measured else None
    timer = EfficiencyMeter(devices, input_ids.shape[0], args.fixed_output_length) if measured else None
    monitors = [monitor for monitor in (memory_monitor, timer) if monitor is not None]
    step_context = timer.step if timer else nullcontext
    decode_start_callback = memory_monitor.begin_decode_phase if memory_monitor else None
    with ExitStack() as stack:
        for monitor in monitors:
            stack.enter_context(monitor)
        if mode == "batched":
            tokens = generate(model, input_ids, attention_mask, args.fixed_output_length,
                              args.method, step_context, decode_start_callback,
                              efficiency_backend=args.efficiency_backend)
        else:
            tokens = generate_serial_prefill(
                model, input_ids, attention_mask, args.fixed_output_length, args.method,
                step_context, decode_start_callback, state_paths=state_paths)
    return tokens, monitors


def predict_batch(args, model, tokenizer, batch, input_device, devices, measured=True):
    input_ids, attention_mask, record = prepare_batch(
        batch,
        tokenizer,
        args.input_length,
        input_device,
    )
    native_backend = args.method in NATIVE_METHODS or (
        args.efficiency_backend and args.method in NATIVE_EFFICIENCY_METHODS)
    mode = serial_prefill_mode(args, args.method, len(batch), native_backend)
    state_paths = SERIAL_PREFILL_STATE_METHODS.get(args.method)
    prepare_sample(args, model, args.input_length)
    try:
        if mode != "batched":
            reason = serial_prefill_supported(model, input_ids, attention_mask, mode, args.method)
            if reason is not None:
                # 探测消耗了方法状态，回退前重置一次再走批量 prefill。
                record["serial_prefill_fallback"] = reason
                mode = "batched"
                finish_sample(args, model)
                prepare_sample(args, model, args.input_length)
        try:
            tokens, monitors = run_generation(args, model, input_ids, attention_mask, devices,
                                              measured, mode, state_paths)
        except SerialPrefillUnsupported as error:
            # 合并失败时重新计时再跑批量 prefill，避免半次串行 prefill 污染 TTFT。
            record["serial_prefill_fallback"] = str(error)
            mode = "batched"
            finish_sample(args, model)
            prepare_sample(args, model, args.input_length)
            tokens, monitors = run_generation(args, model, input_ids, attention_mask, devices,
                                              measured, mode, state_paths)
        if measured and mode != "batched" and getattr(args, "serial_prefill_verify", False):
            record.update(verify_serial_prefill(args, model, input_ids, attention_mask, tokens))
        for monitor in monitors:
            record.update(monitor.metrics())
        record.update(backend="native_generate" if native_backend else "forward",
                      batch_size=len(batch), input_length=args.input_length,
                      fixed_output_length=args.fixed_output_length, prefill_mode=mode)
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


def prediction_records(
    batch,
    decoded,
    round_index,
    input_length,
    output_length,
    prefill_mode=None,
    budget=None,
):
    """生成用于核对预测文本的最小记录。"""
    batch_size = len(batch)
    records = [
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
    if prefill_mode is not None:
        for record in records:
            record["prefill_mode"] = prefill_mode
    if budget is not None:
        for record in records:
            record["budget"] = budget
    return records
