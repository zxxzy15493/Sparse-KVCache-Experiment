"""StreamingLLM cache eviction hook for Qwen3 on current Transformers.

Transformers 4.57 uses one ``Qwen3Attention`` implementation and dispatches the
attention backend internally. This hook replaces that implementation with the
FlashAttention path while retaining only sink and recent KV states.
"""

import types
from typing import Optional

import torch
from flash_attn import flash_attn_func
from transformers.cache_utils import Cache
from streaming_llm.kv_cache import BaseKVCache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
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
    """Qwen3Attention.forward with StartRecentKVCache eviction.

    The signature follows Transformers 4.57's Qwen3Attention. RoPE embeddings
    are computed once by ``Qwen3Model`` and supplied through
    ``position_embeddings``; they must not be recomputed by this hook.
    """
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

    # Qwen3 differs from Qwen2.5 here: normalize Q and K after projection.
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Attention for this call must see every key currently available. Cache
    # eviction is applied only after retaining these tensors for the current
    # computation, so the shortened cache affects the *next* forward call.
    key_states_for_attention = key_states
    value_states_for_attention = value_states

    if past_key_values is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
            "query_states": query_states,
            "attention_mask": attention_mask,
            "num_key_value_groups": self.num_key_value_groups,
            "update_global_past_kv": getattr(self, "update_global_past_kv", True),
        }
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
        key_states_for_attention = key_states
        value_states_for_attention = value_states

        if not isinstance(past_key_values, BaseKVCache):
            compressed_key_states, compressed_value_states = self.kv_cache.evict_for_space(
                key_states, value_states
            )

            # SCBench requests keep their KV temporary, outside shared layers.
            cache_layer = past_key_values.layers[self.layer_idx]
            cache_layer.keys = compressed_key_states
            cache_layer.values = compressed_value_states

    groups = query_states.shape[1] // key_states_for_attention.shape[1]
    key_states = repeat_kv(key_states_for_attention, groups)
    value_states = repeat_kv(value_states_for_attention, groups)

    # flash_attn_func takes [batch, sequence, heads, head_dim]. This streaming
    # runner uses batch size one and no padding, so no 4-D padding mask is sent.
    attn_output = flash_attn_func(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        dropout_p=self.attention_dropout if self.training else 0.0,
        causal=True,
    )
    # Qwen3-4B has an attention width (num_attention_heads * head_dim) that is
    # larger than config.hidden_size (32 * 128 = 4096 vs. 2560). Keep that
    # width here; o_proj performs the projection back to hidden_size. This
    # matches the upstream Qwen3Attention implementation.
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    return self.o_proj(attn_output), None


def enable_qwen_pos_shift_attention(model):
    """Install the streaming attention hook on every Qwen3 decoder layer."""
    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            module.forward = types.MethodType(qwen_pos_shift_attention_forward, module)
