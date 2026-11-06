import torch
from typing import List
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import repeat_kv


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class NativeKVCache(DynamicCache):
    """Full HF KV with SCBench's temporary request/history lifecycle.

    Native attention calls the ordinary Cache.update API. The runner toggles
    permanent/temporary writes here, without replacing attention.forward.
    """

    def __init__(self, config=None):
        super().__init__()
        self.update_global_past_kv = True
        self.temp_key_cache = {}
        self.temp_value_cache = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.update_global_past_kv:
            if self.temp_key_cache:
                raise RuntimeError("Clear temporary KV before updating permanent history")
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
        if layer_idx in self.temp_key_cache:
            key_states = torch.cat((self.temp_key_cache[layer_idx], key_states), dim=-2)
            value_states = torch.cat((self.temp_value_cache[layer_idx], value_states), dim=-2)
        self.temp_key_cache[layer_idx] = key_states
        self.temp_value_cache[layer_idx] = value_states
        layer = self.layers[layer_idx]
        return (torch.cat((layer.keys, key_states), dim=-2),
                torch.cat((layer.values, value_states), dim=-2))

    def get_seq_length(self, layer_idx=0):
        length = super().get_seq_length(layer_idx)
        temporary = self.temp_key_cache.get(layer_idx)
        return length + (temporary.shape[-2] if temporary is not None else 0)

    def get_mask_sizes(self, cache_position, layer_idx=0):
        return self.get_seq_length(layer_idx) + cache_position.shape[0], 0

    def clear_temp_kv_cache(self):
        self.temp_key_cache.clear()
        self.temp_value_cache.clear()


class StartRecentKVCache:
    def __init__(
        self,
        start_size=16,
        recent_size=1008,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim

    def evict_for_space(self, key_states, value_states, attention_mask=None):
        seq_len = key_states.shape[self.k_seq_dim]
        
        if seq_len > self.cache_size:
            if attention_mask is None:
                k_sink = key_states[:, :, :self.start_size, :]
                v_sink = value_states[:, :, :self.start_size, :]
            else:
                mask = attention_mask[:, -seq_len:].to(torch.bool)
                positions = torch.arange(seq_len, device=key_states.device).expand_as(mask)
                sink_idx = positions.masked_fill(~mask, seq_len - 1).sort(dim=-1).values[:, :self.start_size]
                sink_idx = sink_idx[:, None, :, None].expand(-1, key_states.shape[1], -1, key_states.shape[-1])
                k_sink = torch.gather(key_states, self.k_seq_dim, sink_idx)
                v_sink = torch.gather(value_states, self.v_seq_dim, sink_idx)
            
            k_recent = key_states[:, :,  - self.recent_size:, :]
            v_recent = value_states[:, :,  - self.recent_size:, :]
            
            key_states = torch.cat([k_sink, k_recent], dim=self.k_seq_dim)
            value_states = torch.cat([v_sink, v_recent], dim=self.v_seq_dim)
            
        return key_states, value_states


# SCBench cache lifecycle from MInference-main/minference/modules/kvcompression.py.
# Copyright (c) 2024-2025 Microsoft; MIT License.
class _DenseKVCluster:
    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        return key_states, value_states


class StreamingLLMKVCluster:
    def __init__(
        self,
        window_size=4096 - 128,
        max_capacity_prompt=4096,
        kernel_size=5,
        pooling="avgpool",
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(
        self,
        window_size=4096 - 128,
        max_capacity_prompt=4096,
        kernel_size=5,
        pooling="avgpool",
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        attention_mask,
        num_key_value_groups,
    ):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            indices = torch.tensor(
                range(self.max_capacity_prompt - self.window_size), dtype=torch.int64
            ).to(key_states.device)
            indices = (
                indices.unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(bsz, num_heads, 1, head_dim)
            )

            k_past_compress = key_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            v_past_compress = value_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            k_cur = key_states[:, :, -self.window_size :, :]
            v_cur = value_states[:, :, -self.window_size :, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states


class BaseKVCache(Cache):
    def __init__(self, config=None):
        super().__init__(layers=[])
        # Used in `generate` to keep tally of how many tokens the cache has seen
        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self.kv_clusters = {}
        self.kv_cluster_granularity = "layer"

        self.temp_key_cache = []
        self.temp_value_cache = []

    def get_kv_cluster_class_config(self, layer_idx: int, head_idx: int = None):
        return "dense", _DenseKVCluster, {}

    def get_kv_cluster_class(self, layer_idx: int, head_idx=None):
        cluster_name, cluster_class, cluster_config = self.get_kv_cluster_class_config(
            layer_idx, head_idx
        )
        if cluster_name not in self.kv_clusters:
            self.kv_clusters[cluster_name] = cluster_class(**cluster_config)
        return self.kv_clusters[cluster_name]

    def compresssed_kv(
        self,
        key_states,
        query_states,
        value_states,
        attention_mask,
        num_key_value_groups,
        layer_idx: int,
    ):
        if self.kv_cluster_granularity == "layer":
            kv_cluster = self.get_kv_cluster_class(layer_idx)

            key_compress, value_compress = kv_cluster.update_kv(
                key_states,
                query_states,
                value_states,
                attention_mask,
                num_key_value_groups,
            )
            self.key_cache.append(key_compress)
            self.value_cache.append(value_compress)
        else:
            assert (
                False
            ), f"kv_cluster_granularity {self.kv_cluster_granularity} not supported"

    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs,
    ):
        # if prefill, then compress; if decode, then update
        # [bsz, num_heads, q_len, head_dim]

        update_global_past_kv = cache_kwargs.get("update_global_past_kv", True)
        query_states = cache_kwargs["query_states"]
        attention_mask = cache_kwargs["attention_mask"]
        num_key_value_groups = cache_kwargs["num_key_value_groups"]

        if key_states.size(1) != query_states.size(1):  # GQA
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        q_len = query_states.shape[-2]
        initializing_kv_cluster = False
        if (
            len(self.key_cache) == layer_idx
        ):  # initialize kv_cluster, ie, the first query/context
            initializing_kv_cluster = True
            self.compresssed_kv(
                key_states,
                query_states,
                value_states,
                attention_mask,
                num_key_value_groups,
                layer_idx,
            )
        else:  # the follow up queries/contexts
            if update_global_past_kv:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )
            else:  # add KVs to temp_kv_cache
                if len(self.temp_key_cache) == layer_idx:
                    self.temp_key_cache.append(key_states)
                    self.temp_value_cache.append(value_states)
                else:
                    self.temp_key_cache[layer_idx] = torch.cat(
                        [self.temp_key_cache[layer_idx], key_states], dim=-2
                    )
                    self.temp_value_cache[layer_idx] = torch.cat(
                        [self.temp_value_cache[layer_idx], value_states], dim=-2
                    )

        torch.cuda.empty_cache()
        if not initializing_kv_cluster:  # return the compressed KV cache
            if self.temp_key_cache:  # concat global past_kv and temp_kv_cache
                key_states = torch.cat(
                    [self.key_cache[layer_idx], self.temp_key_cache[layer_idx]], dim=-2
                )
                value_states = torch.cat(
                    [self.value_cache[layer_idx], self.temp_value_cache[layer_idx]],
                    dim=-2,
                )
            else:
                key_states = self.key_cache[layer_idx]
                value_states = self.value_cache[layer_idx]
        key_states = repeat_kv(key_states, query_states.size(1) // key_states.size(1))
        value_states = repeat_kv(
            value_states, query_states.size(1) // value_states.size(1)
        )
        return key_states, value_states

    def get_seq_length(self, layer_idx=0):
        if len(self.key_cache) <= layer_idx:
            return 0
        return self._seen_tokens

    def to_legacy_cache(self):
        legacy_cache = ()
        for layer_idx in range(len(self.key_cache)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values):
        cache = cls()
        for layer_idx in range(len(past_key_values)):
            key_states, value_states = past_key_values[layer_idx]
            cache.update(key_states, value_states, layer_idx)
        return cache

    def clear_temp_kv_cache(self):
        if self.temp_key_cache:
            self._seen_tokens -= self.temp_key_cache[-1].shape[
                -2
            ]  # seq_len of temp_kv_cache
        self.temp_key_cache = []
        self.temp_value_cache = []

    def get_mask_sizes(self, cache_position, layer_idx=0):
        # Current HF separates physical mask length from logical RoPE positions.
        length = self.key_cache[layer_idx].shape[-2] if layer_idx < len(self.key_cache) else 0
        if layer_idx < len(self.temp_key_cache):
            length += self.temp_key_cache[layer_idx].shape[-2]
        return length + cache_position.shape[0], 0



class StreamingLLMKVCache(BaseKVCache):
    """Continuously retain sink/recent KV, with isolated SCBench requests.

    Each forward attends to the previously retained KV plus its current input.
    Permanent history and disposable request branches are trimmed separately;
    clearing a branch restores the shared document and its logical position.
    """

    def __init__(self, config):
        n_local = config.attn_kwargs.get("n_local", 3968)
        n_init = config.attn_kwargs.get("n_init", 128)
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in (n_init, n_local)):
            raise ValueError("n_init and n_local must be positive integers")
        config.attn_kwargs["window_size"] = n_local
        config.attn_kwargs["max_capacity_prompt"] = n_local + n_init
        super().__init__(config)
        self.window_size = n_local
        self.max_capacity_prompt = n_local + n_init
        self.evictor = StartRecentKVCache(start_size=n_init, recent_size=n_local)
        self._temp_seen_tokens = None

    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        query_states = cache_kwargs["query_states"]
        groups = query_states.shape[1] // key_states.shape[1]
        key_states = repeat_kv(key_states, groups)
        value_states = repeat_kv(value_states, groups)
        input_length = query_states.shape[-2]
        permanent = cache_kwargs.get("update_global_past_kv", True)
        if permanent:
            if self.temp_key_cache:
                raise RuntimeError("Clear temporary KV before updating permanent history")
            target_keys, target_values = self.key_cache, self.value_cache
            previous_keys, previous_values = target_keys, target_values
        else:
            if layer_idx >= len(self.key_cache):
                raise RuntimeError("Temporary decoding requires a prefetched document")
            target_keys, target_values = self.temp_key_cache, self.temp_value_cache
            if layer_idx < len(target_keys):
                previous_keys, previous_values = target_keys, target_values
            else:
                previous_keys, previous_values = self.key_cache, self.value_cache

        if layer_idx < len(previous_keys):
            key_states = torch.cat((previous_keys[layer_idx], key_states), dim=-2)
            value_states = torch.cat((previous_values[layer_idx], value_states), dim=-2)
        retained_key, retained_value = self.evictor.evict_for_space(
            key_states, value_states, cache_kwargs.get("attention_mask"),
        )
        if layer_idx == len(target_keys):
            target_keys.append(retained_key)
            target_values.append(retained_value)
        else:
            target_keys[layer_idx] = retained_key
            target_values[layer_idx] = retained_value
        if layer_idx == 0:
            if permanent:
                self._seen_tokens += input_length
            else:
                if self._temp_seen_tokens is None:
                    self._temp_seen_tokens = self._seen_tokens
                self._temp_seen_tokens += input_length
        return key_states, value_states

    def get_seq_length(self, layer_idx=0):
        if layer_idx < len(self.temp_key_cache):
            return self._temp_seen_tokens
        return self._seen_tokens if layer_idx < len(self.key_cache) else 0

    def get_mask_sizes(self, cache_position, layer_idx=0):
        keys = self.temp_key_cache if layer_idx < len(self.temp_key_cache) else self.key_cache
        length = keys[layer_idx].shape[-2] if layer_idx < len(keys) else 0
        return length + cache_position.shape[0], 0

    def clear_temp_kv_cache(self):
        self.temp_key_cache.clear()
        self.temp_value_cache.clear()
        self._temp_seen_tokens = None
