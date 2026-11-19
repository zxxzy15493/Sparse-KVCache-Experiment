"""逐条 prefill、cache 合并和自适应 KV 压缩。"""

import gc
import inspect
from contextlib import contextmanager

import torch


ADAPTIVE_METHODS = {"adakv", "headkv"}
SERIAL_CACHE_METHODS = {
    "full_attention", "snapkv", "streaming", "pyramidkv", "duo-attention",
    "flexprefill", "minference", "sparq", "xattention",
}
SERIAL_STATE_PATHS = {
    "h2o": (("kv_cache", "hh_score"),),
    "keyformer": (("kv_cache", "key_score"), ("kv_cache_evictor", "key_score")),
}

_serial_support = {}


class SerialPrefillUnsupported(RuntimeError):
    """逐条 prefill 得到的 cache 无法按 batch 维合并。"""


def logits_keep_kwargs(model):
    parameters = inspect.signature(model.forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in parameters:
            return {name: 1}
    return {}


def prefill_one(model, input_ids, attention_mask, cache_position, logits_kwargs):
    return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                 cache_position=cache_position, return_dict=True, **logits_kwargs)


def _mergeable(keys, values):
    if any(not torch.is_tensor(tensor) for tensor in keys + values):
        return False
    return keys[0].shape[0] == 1 and len({key.shape for key in keys}) == 1


def _stack_batch(parts, release):
    merged = parts[0].new_empty((len(parts), *parts[0].shape[1:]))
    for index, part in enumerate(parts):
        merged[index].copy_(part[0])
        parts[index] = None
        release(index)
    return merged


def merge_sample_caches(caches):
    first = caches[0]
    layers = getattr(first, "layers", None)
    if layers is not None:
        if not layers or any(len(cache.layers) != len(layers) for cache in caches):
            return None
        for index, layer in enumerate(layers):
            sample_layers = [cache.layers[index] for cache in caches]
            keys = [sample.keys for sample in sample_layers]
            values = [sample.values for sample in sample_layers]
            if not _mergeable(keys, values):
                return None
            layer.keys = _stack_batch(keys, lambda pos: setattr(sample_layers[pos], "keys", None))
            layer.values = _stack_batch(values, lambda pos: setattr(sample_layers[pos], "values", None))
            layer.is_initialized = True
        return first

    keys_by_layer = getattr(first, "key_cache", None)
    values_by_layer = getattr(first, "value_cache", None)
    if not isinstance(keys_by_layer, list) or not isinstance(values_by_layer, list) or not keys_by_layer:
        return None
    if any(len(cache.key_cache) != len(keys_by_layer) for cache in caches):
        return None
    for index in range(len(keys_by_layer)):
        keys = [cache.key_cache[index] for cache in caches]
        values = [cache.value_cache[index] for cache in caches]
        if not _mergeable(keys, values):
            return None
        keys_by_layer[index] = _stack_batch(keys, lambda pos: caches[pos].key_cache.__setitem__(index, None))
        values_by_layer[index] = _stack_batch(values, lambda pos: caches[pos].value_cache.__setitem__(index, None))
    return first


class SampleStateTracker:
    """合并 H2O 和 Keyformer attention 模块上的逐样本分数。"""

    def __init__(self, model, paths):
        self.attributes = [
            (module, owner_name, field, [])
            for owner_name, field in paths
            for module in model.modules()
        ]

    def capture(self):
        for module, owner_name, field, values in self.attributes:
            owner = getattr(module, owner_name, None)
            values.append(getattr(owner, field, None))

    def merge(self):
        for module, owner_name, field, values in self.attributes:
            if all(torch.is_tensor(value) and value.shape[0] == 1 for value in values):
                if len({value.shape[1:] for value in values}) == 1:
                    owner = getattr(module, owner_name)
                    setattr(owner, field, torch.cat(values, dim=0))
        self.attributes.clear()


def serial_prefill_supported(model, method, input_ids, attention_mask):
    if method not in _serial_support:
        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
        logits_kwargs = logits_keep_kwargs(model)
        caches = []
        try:
            for index in range(2):
                outputs = prefill_one(model, input_ids[index:index + 1], attention_mask[index:index + 1],
                                      cache_position, logits_kwargs)
                caches.append(outputs.past_key_values)
                del outputs
            _serial_support[method] = merge_sample_caches(caches) is not None
        except Exception:
            # 部分模型只支持批量 forward，探测失败时保留批量路径。
            _serial_support[method] = False
        finally:
            caches.clear()
            gc.collect()
            if input_ids.device.type == "cuda":
                torch.cuda.empty_cache()
    return _serial_support[method]


def generate_serial_prefill(model, method, input_ids, attention_mask, output_length):
    batch_size, prompt_length = input_ids.shape
    device = input_ids.device
    cache_position = torch.arange(prompt_length, device=device)
    logits_kwargs = logits_keep_kwargs(model)
    tracker = SampleStateTracker(model, SERIAL_STATE_PATHS.get(method, ()))
    caches, first_tokens = [], []
    for index in range(batch_size):
        outputs = prefill_one(model, input_ids[index:index + 1], attention_mask[index:index + 1],
                              cache_position, logits_kwargs)
        caches.append(outputs.past_key_values)
        first_tokens.append(outputs.logits[:, -1, :].argmax(dim=-1))
        tracker.capture()
        del outputs

    cache = merge_sample_caches(caches)
    caches.clear()
    if cache is None:
        raise SerialPrefillUnsupported("逐条 prefill 的 cache 无法按 batch 维合并")
    tracker.merge()
    first = torch.cat(first_tokens).to(device)
    first_tokens.clear()
    generated = [first]
    sequence = torch.cat([input_ids, first[:, None]], dim=1)
    mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1)
    for step in range(1, output_length):
        cache_position = torch.arange(sequence.shape[1] - 1, sequence.shape[1], device=device)
        model_inputs = model.prepare_inputs_for_generation(
            sequence, past_key_values=cache, attention_mask=mask, cache_position=cache_position,
            use_cache=True, **logits_kwargs,
        )
        outputs = model(**model_inputs, return_dict=True)
        next_tokens = outputs.logits[:, -1, :].argmax(dim=-1).to(device)
        generated.append(next_tokens)
        del outputs
        if step + 1 < output_length:
            sequence = torch.cat([sequence, next_tokens[:, None]], dim=1)
            mask = torch.cat([mask, mask.new_ones((batch_size, 1))], dim=1)
    return torch.stack(generated, dim=1)


def _rebuild_decode_metadata(cluster, lengths):
    """按 [sample, head] 顺序重建变长 decode 的索引。"""
    lengths = lengths.to(dtype=torch.int32)
    cluster.head_lens = lengths
    cluster.klen_sum = int(lengths.sum().item())
    cluster.max_seqlen_k = int(lengths.max().item())
    cluster.cu_headlens = torch.cumsum(lengths, dim=0, dtype=torch.int32)
    cluster.cu_klen = torch.cat([lengths.new_zeros(1), cluster.cu_headlens])
    head_count = lengths.numel()
    cluster.layer_qlens = torch.ones_like(lengths)
    cluster.qlen_sum = head_count
    cluster.cu_qlen = torch.arange(head_count + 1, dtype=torch.int32, device=lengths.device)
    cluster.cu_offset = cluster.cu_qlen.clone()
    cluster.cu_head_offset = cluster.cu_qlen[1:].clone()


def _serial_update(cluster, original):
    def update(key_states, query_states, value_states):
        keys, values, lengths = [], [], []
        for index in range(key_states.shape[0]):
            key, value = original(key_states[index:index + 1], query_states[index:index + 1],
                                  value_states[index:index + 1])
            keys.append(key)
            values.append(value)
            lengths.append(cluster.head_lens.clone())
        _rebuild_decode_metadata(cluster, torch.cat(lengths))
        return torch.cat(keys), torch.cat(values)

    return update


@contextmanager
def serial_adaptive_compression(model, method):
    """保留批量 forward，逐样本执行 Adakv 和 HeadKV 的每层 KV 压缩。"""
    originals = []
    missing = object()
    initializer = "init_headkv" if method == "adakv" else "init_reason_snapkv"
    try:
        for attention in model.modules():
            if not hasattr(attention, "q_proj") or not hasattr(attention, "layer_idx"):
                continue
            if not hasattr(attention, "kv_cluster"):
                forward = getattr(attention, "_old_forward", attention.forward)
                function = getattr(forward, "__func__", forward)
                function.__globals__[initializer](attention)
            cluster = attention.kv_cluster
            originals.append((cluster, cluster.__dict__.get("update_kv", missing)))
            cluster.update_kv = _serial_update(cluster, cluster.update_kv)
        yield
    finally:
        for cluster, original in originals:
            if original is missing:
                del cluster.update_kv
            else:
                cluster.update_kv = original
