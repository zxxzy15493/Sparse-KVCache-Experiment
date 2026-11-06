"""Shared SnapKV hook for the unified Llama/Qwen2 attention APIs in 4.57+."""

from typing import Optional

import torch
from flash_attn import flash_attn_func
from transformers.cache_utils import Cache

from snapkv.monkeypatch.snapkv_utils import init_snapkv


def llama_flash_attn2_forward (
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Current LlamaAttention/Qwen2Attention forward with SnapKV prefill."""
    # Both upstream modules expose these helpers in their defining module.
    module = __import__(self.__class__.__module__, fromlist=["apply_rotary_pos_emb", "repeat_kv"])
    apply_rotary_pos_emb, repeat_kv = module.apply_rotary_pos_emb, module.repeat_kv
    init_snapkv(self)

    bsz, q_len, _ = hidden_states.shape
    if q_len > 1:
        self.kv_seq_len = 0
    shape = (bsz, q_len, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        cached_length = past_key_values.layers[self.layer_idx].get_seq_length()
        if q_len > 1 and cached_length == 0:
            self.kv_seq_len = q_len
            compressed_key, compressed_value = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask, self.num_key_value_groups
            )[:2]
            # Current attention below deliberately uses the full prompt KV.
            past_key_values.update(compressed_key, compressed_value, self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

    output = flash_attn_func(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        dropout_p=self.attention_dropout if self.training else 0.0,
        causal=True,
    )
    return self.o_proj(output.reshape(bsz, q_len, -1).contiguous()), None
