import math
from typing import Optional, Tuple
import types

import torch
from torch import nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2SdpaAttention,
    Qwen2FlashAttention2,
    repeat_kv,
    apply_rotary_pos_emb,
)

__all__ = ["enable_qwen_pos_shift_attention"]


def qwen_pos_shift_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[object] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    if not hasattr(self, "kv_cache"):
        from streaming_llm.kv_cache import StartRecentKVCache
        self.kv_cache = StartRecentKVCache(
            start_size=getattr(self.config, "start_size", 16),
            recent_size=getattr(self.config, "recent_size", 1008),
            k_seq_dim=2,
            v_seq_dim=2
        )
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states   = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    cos, sin = self.rotary_emb(value_states, position_ids)

    query_states,key_states= apply_rotary_pos_emb(query_states,key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"cos": cos,"sin": sin}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        key_states_compress, value_states_compress = self.kv_cache.evict_for_space(key_states, value_states)
        past_key_value.key_cache[self.layer_idx] = key_states_compress
        past_key_value.value_cache[self.layer_idx] = value_states_compress

    key_states   = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    dropout_rate = self.attention_dropout if self.training else 0.0

    attn_output = flash_attn_func(
                    query_states, key_states, value_states, 
                    dropout_p=dropout_rate, 
                    causal=True
                )

    attn_output = attn_output.view(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def enable_qwen_pos_shift_attention(model):
    for name, module in model.named_modules():
        if isinstance(module,(Qwen2Attention, Qwen2SdpaAttention, Qwen2FlashAttention2),):
            module.forward = types.MethodType(
                qwen_pos_shift_attention_forward, module
            )

