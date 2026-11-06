"""StreamingLLM cache-eviction hook for Llama on Transformers 4.57+."""

import math
import types
from typing import Optional

import numpy as np
import torch
from flash_attn import flash_attn_func
from transformers.cache_utils import Cache
from streaming_llm.kv_cache import BaseKVCache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

__all__ = ["enable_llama_pos_shift_attention"]

step_layer_recalls = {}


def llama_pos_shift_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Current LlamaAttention.forward with StartRecentKVCache eviction."""
    if not hasattr(self, "kv_cache"):
        from streaming_llm.kv_cache import StartRecentKVCache

        self.kv_cache = StartRecentKVCache(
            start_size=getattr(self.config, "start_size", 16),
            recent_size=getattr(self.config, "recent_size", 1008),
            k_seq_dim=2,
            v_seq_dim=2,
        )

    bsz, q_len, _ = hidden_states.shape
    if q_len > 1:
        self.absolute_full_k = None
        step_layer_recalls["_global_step"] = 0

    hidden_shape = (bsz, q_len, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_states_for_attention = key_states
    value_states_for_attention = value_states
    if past_key_values is not None:
        cache_kwargs = {
            "sin": sin, "cos": cos, "cache_position": cache_position,
            "query_states": query_states, "attention_mask": attention_mask,
            "num_key_value_groups": self.num_key_value_groups,
            "update_global_past_kv": getattr(self, "update_global_past_kv", True),
        }

        full_k = None
        if getattr(self, "check_recall", False):
            with torch.no_grad():
                self.absolute_full_k = (
                    torch.cat([self.absolute_full_k, key_states], dim=2)
                    if self.absolute_full_k is not None
                    else key_states.detach().clone()
                )

        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
        key_states_for_attention = key_states
        value_states_for_attention = value_states
        update_global_past_kv = not isinstance(past_key_values, BaseKVCache)
        if update_global_past_kv:
            compressed_key, compressed_value = self.kv_cache.evict_for_space(
                key_states, value_states
            )

        if getattr(self, "check_recall", False) and q_len == 1 and update_global_past_kv:
            full_k = self.absolute_full_k
            full_len, k_size = full_k.shape[2], compressed_key.shape[2]
            if full_len > k_size:
                k_full_expanded = repeat_kv(full_k, self.num_key_value_groups)
                scores = torch.matmul(query_states, k_full_expanded.transpose(-1, -2)) / math.sqrt(self.head_dim)
                _, real_topk_indices = torch.topk(torch.softmax(scores, dim=-1), k=k_size, dim=-1)
                retained = torch.cat([
                    torch.arange(0, self.kv_cache.start_size, device=query_states.device),
                    torch.arange(full_len - self.kv_cache.recent_size, full_len, device=query_states.device),
                ])
                step_layer_recalls[self.layer_idx] = torch.isin(
                    real_topk_indices.reshape(-1, k_size), retained
                ).float().mean().item()
                if self.layer_idx == getattr(self.config, "num_hidden_layers", 32) - 1:
                    step = step_layer_recalls.get("_global_step", 0)
                    recalls = [step_layer_recalls.get(i, 0.0) for i in range(self.config.num_hidden_layers)]
                    with open("streaming_recall_all_steps.csv", "a") as recall_file:
                        recall_file.write(f"{step}," + ",".join(f"{v:.4f}" for v in recalls) + f",{np.mean(recalls):.4f}\n")
                    step_layer_recalls.clear()
                    step_layer_recalls["_global_step"] = step + 1

        # Full KV remains available for the current attention calculation.
        # The compressed cache is used only by the following forward call.
        if update_global_past_kv:
            cache_layer = past_key_values.layers[self.layer_idx]
            cache_layer.keys = compressed_key
            cache_layer.values = compressed_value

    groups = query_states.shape[1] // key_states_for_attention.shape[1]
    key_states = repeat_kv(key_states_for_attention, groups)
    value_states = repeat_kv(value_states_for_attention, groups)
    attn_output = flash_attn_func(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        dropout_p=self.attention_dropout if self.training else 0.0,
        causal=True,
    )
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    return self.o_proj(attn_output), None


def enable_llama_pos_shift_attention(model):
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            module.forward = types.MethodType(llama_pos_shift_attention_forward, module)
