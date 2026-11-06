"""Keyformer KV eviction hook for Qwen3 on Transformers 4.57+."""

import types
from typing import Optional

import torch
from flash_attn import flash_attn_func
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
    repeat_kv,
)


def qwen3_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not hasattr(self, "kv_cache"):
        from .kv_cache import KeyformerKVCache

        self.kv_cache = KeyformerKVCache(
            recent_size=getattr(self.config, "recent_size", 32),
            key_size=getattr(self.config, "key_size", 992),
            tau_init=getattr(self.config, "tau_init", 1.0),
            tau_delta=getattr(self.config, "tau_delta", 0.01),
            k_seq_dim=2,
            v_seq_dim=2,
        )

    bsz, q_len, _ = hidden_states.shape
    if q_len > 1:
        self.itr_count = 0
    elif not hasattr(self, "itr_count"):
        self.itr_count = 0

    hidden_shape = (bsz, q_len, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Keyformer selects independently for every query head, hence repeated KV.
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
        compressed_key, compressed_value = self.kv_cache.evict_for_space(
            query_states, key_states, value_states, self.itr_count
        )
        # Keep full KV for this attention computation; compression is for the
        # following forward call, matching the original Keyformer hook.
        cache_layer = past_key_values.layers[self.layer_idx]
        cache_layer.keys = compressed_key
        cache_layer.values = compressed_value

    if q_len == 1:
        self.itr_count += 1

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
    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            module.forward = types.MethodType(qwen3_attention_forward, module)
