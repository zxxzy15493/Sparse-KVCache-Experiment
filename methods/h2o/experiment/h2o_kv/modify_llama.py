import types
from typing import Optional

import torch
from flash_attn import flash_attn_func
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv


def llama_attention_forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], attention_mask: Optional[torch.Tensor] = None, past_key_values: Optional[Cache] = None, cache_position: Optional[torch.LongTensor] = None, **kwargs):
    if not hasattr(self, "kv_cache"):
        from .kv_cache import H2OKVCache
        self.kv_cache = H2OKVCache(recent_size=getattr(self.config, "recent_size", 32), h2o_size=getattr(self.config, "h2o_size", 992), k_seq_dim=2, v_seq_dim=2)
    bsz, q_len, _ = hidden_states.shape
    shape = (bsz, q_len, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states, value_states = repeat_kv(key_states, self.num_key_value_groups), repeat_kv(value_states, self.num_key_value_groups)
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})
        compressed_key, compressed_value = self.kv_cache.evict_for_space(query_states, key_states, value_states)
        # Use full KV below; save H2O-selected KV only for the next step.
        layer = past_key_values.layers[self.layer_idx]
        layer.keys, layer.values = compressed_key, compressed_value
    output = flash_attn_func(query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), dropout_p=self.attention_dropout if self.training else 0.0, causal=True)
    return self.o_proj(output.reshape(bsz, q_len, -1).contiguous()), None


def enable_llama_pos_shift_attention(model):
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            module.forward = types.MethodType(llama_attention_forward, module)
