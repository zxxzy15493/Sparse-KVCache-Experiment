"""SnapKV attention hook for Qwen3 on Transformers 4.57+."""

from typing import Optional

import torch
from flash_attn import flash_attn_func
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from snapkv.monkeypatch.snapkv_utils import init_snapkv


def qwen3_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Current Qwen3Attention.forward with SnapKV prompt compression."""
    init_snapkv(self)
    bsz, q_len, _ = hidden_states.shape
    if q_len > 1:
        self.kv_seq_len = 0

    hidden_shape = (bsz, q_len, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # SnapKV's cluster scores each query head, so preserve its original
    # repeated-KV representation in the cache.
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        # Cache.get_seq_length() defaults to layer 0. During prefill that layer
        # is already populated when later decoder layers run, so query this
        # attention layer explicitly.
        cached_length = past_key_values.layers[self.layer_idx].get_seq_length()

        if q_len > 1 and cached_length == 0:
            # Prompt phase: select SnapKV tokens for the cache, while current
            # attention still uses the full prompt KV below.
            self.kv_seq_len = q_len
            compressed = self.kv_cluster.update_kv(
                key_states,
                query_states,
                value_states,
                attention_mask,
                self.num_key_value_groups,
            )
            compressed_key, compressed_value = compressed[:2]
            past_key_values.update(
                compressed_key, compressed_value, self.layer_idx, cache_kwargs
            )
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

    attn_output = flash_attn_func(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        dropout_p=self.attention_dropout if self.training else 0.0,
        causal=True,
    )
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    return self.o_proj(attn_output), None
