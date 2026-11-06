"""StreamingLLM cache-eviction hook for Qwen2 on Transformers 4.57+."""

import types
from typing import Optional

import torch
from flash_attn import flash_attn_func
from transformers.cache_utils import Cache
from streaming_llm.kv_cache import BaseKVCache
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    apply_rotary_pos_emb,
    repeat_kv,
)

__all__ = ["enable_qwen_pos_shift_attention"]


def qwen_pos_shift_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Current Qwen2Attention.forward with StartRecentKVCache eviction."""
    if not hasattr(self, "kv_cache"):
        from streaming_llm.kv_cache import StartRecentKVCache

        self.kv_cache = StartRecentKVCache(
            start_size=getattr(self.config, "start_size", 16),
            recent_size=getattr(self.config, "recent_size", 1008),
            k_seq_dim=2,
            v_seq_dim=2,
        )

    bsz, q_len, _ = hidden_states.shape
    hidden_shape = (bsz, q_len, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # In Transformers 4.57 RoPE is computed once by Qwen2Model and passed to
    # every attention layer rather than obtained from self.rotary_emb here.
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
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
        key_states_for_attention = key_states
        value_states_for_attention = value_states

        # Preserve the original hook's behavior: attend to all KV in this
        # call, then save only sink + recent states for the next call.
        if not isinstance(past_key_values, BaseKVCache):
            compressed_key, compressed_value = self.kv_cache.evict_for_space(
                key_states, value_states
            )
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


def enable_qwen_pos_shift_attention(model):
    """Install the hook on every current Qwen2 attention module."""
    for module in model.modules():
        if isinstance(module, Qwen2Attention):
            module.forward = types.MethodType(qwen_pos_shift_attention_forward, module)
