"""仅供效率测试使用的 Adakv/HeadKV 串行 prefill 适配。"""

from contextlib import contextmanager

import torch


ADAPTIVE_METHODS = frozenset({"adakv", "headkv"})


def rebuild_decode_metadata(cluster, lengths):
    """按 [sample, head] 的长度重建现有 varlen decode 需要的全部元数据。"""
    lengths = lengths.to(dtype=torch.int32)
    cluster.head_lens = lengths
    cluster.klen_sum = int(lengths.sum().item())
    cluster.max_seqlen_k = int(lengths.max().item())
    cluster.cu_headlens = torch.cumsum(lengths, dim=0, dtype=torch.int32)
    cluster.cu_klen = torch.cat([lengths.new_zeros(1), cluster.cu_headlens])
    head_count = lengths.numel()
    cluster.layer_qlens = torch.ones_like(lengths)
    cluster.qlen_sum = head_count
    cluster.cu_qlen = torch.arange(
        head_count + 1, dtype=torch.int32, device=lengths.device
    )
    cluster.cu_offset = cluster.cu_qlen.clone()
    cluster.cu_head_offset = cluster.cu_qlen[1:].clone()


def _serial_update(cluster, original):
    def update(key_states, query_states, value_states):
        keys = []
        values = []
        lengths = []
        for index in range(key_states.shape[0]):
            key, value = original(
                key_states[index:index + 1],
                query_states[index:index + 1],
                value_states[index:index + 1],
            )
            keys.append(key)
            values.append(value)
            lengths.append(cluster.head_lens.clone())
        rebuild_decode_metadata(cluster, torch.cat(lengths))
        return torch.cat(keys), torch.cat(values)

    return update


@contextmanager
def serial_adaptive_compression(model, method):
    """保留 batch 前向的数值路径，仅逐样本执行每层的 KV 压缩。"""
    originals = []
    missing = object()
    try:
        for attention in model.modules():
            if not hasattr(attention, "q_proj") or not hasattr(attention, "layer_idx"):
                continue
            if not hasattr(attention, "kv_cluster"):
                forward = getattr(attention, "_old_forward", attention.forward)
                function = getattr(forward, "__func__", forward)
                initializer = "init_headkv" if method == "adakv" else "init_reason_snapkv"
                function.__globals__[initializer](attention)
            cluster = attention.kv_cluster
            original = cluster.update_kv
            originals.append((cluster, cluster.__dict__.get("update_kv", missing)))
            cluster.update_kv = _serial_update(cluster, original)
        yield
    finally:
        for cluster, original in originals:
            if original is missing:
                del cluster.update_kv
            else:
                cluster.update_kv = original


class AdaptivePrefillAdapter:
    """逐条收集各层 packed KV，按 [sample, head] 合并并恢复 batch decode 元数据。"""

    def __init__(self, model):
        self.attentions = sorted(
            (
                module
                for module in model.modules()
                if hasattr(module, "q_proj") and hasattr(module, "layer_idx")
            ),
            key=lambda module: module.layer_idx,
        )
        self.sample_lengths = []

    def reset_sample(self):
        # 每条请求开始前清理上一请求留下的逻辑长度。
        for attention in self.attentions:
            attention.kv_seq_len = 0

    def capture(self):
        # 下一条 prefill 会覆盖模块元数据，不能只保存最后一条样本的状态。
        self.sample_lengths.append([
            attention.kv_cluster.head_lens.clone()
            for attention in self.attentions
        ])

    def merge(self, caches):
        if not caches or len(caches) != len(self.sample_lengths):
            return None
        layer_count = len(self.attentions)
        if not layer_count:
            return None
        for cache in caches:
            if len(getattr(cache, "key_cache", ())) != layer_count:
                return None
            if len(getattr(cache, "value_cache", ())) != layer_count:
                return None
        seen_tokens = [cache.get_seq_length() for cache in caches]
        if len(set(seen_tokens)) != 1:
            return None

        # 在修改源缓存之前检查所有层，避免失败时留下部分合并的数据。
        for layer_index, attention in enumerate(self.attentions):
            if attention.layer_idx != layer_index:
                return None
            first_key = caches[0].key_cache[layer_index]
            if not torch.is_tensor(first_key) or first_key.ndim != 2:
                return None
            for sample_index, cache in enumerate(caches):
                key = cache.key_cache[layer_index]
                value = cache.value_cache[layer_index]
                lengths = self.sample_lengths[sample_index][layer_index]
                if not torch.is_tensor(key) or not torch.is_tensor(value):
                    return None
                if key.ndim != 2 or value.shape != key.shape:
                    return None
                if key.shape[1] != first_key.shape[1]:
                    return None
                if key.device != first_key.device or key.dtype != first_key.dtype:
                    return None
                if value.device != key.device or value.dtype != key.dtype:
                    return None
                if lengths.ndim != 1 or lengths.numel() != attention.num_heads:
                    return None
                if int(lengths.sum().item()) != key.shape[0]:
                    return None

        merged = caches[0]
        for layer_index, attention in enumerate(self.attentions):
            # 每次只产生一层的临时副本，随后释放各条样本的源张量。
            for name in ("key_cache", "value_cache"):
                parts = [getattr(cache, name)[layer_index] for cache in caches]
                combined = torch.cat(parts, dim=0)
                for cache in caches:
                    getattr(cache, name)[layer_index] = None
                parts.clear()
                getattr(merged, name)[layer_index] = combined

            lengths = torch.cat([
                sample[layer_index] for sample in self.sample_lengths
            ])
            rebuild_decode_metadata(attention.kv_cluster, lengths)
            attention.kv_seq_len = seen_tokens[0]
        self.sample_lengths.clear()
        return merged


def adaptive_prefill_adapter(model, method):
    if method in ADAPTIVE_METHODS:
        return AdaptivePrefillAdapter(model)
    return None
