import math
from typing import Optional, Tuple

import pdb
import torch
from torch import nn
import numpy as np
import torch.utils.checkpoint
import types
import torch.nn.functional as F
from transformers.cache_utils import Cache
from flash_attn import flash_attn_func
# from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
    
)

step_layer_recalls = {} 

def llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    if not hasattr(self, "kv_cache"):
        from .kv_cache import KeyformerKVCache
        self.kv_cache = KeyformerKVCache(
            recent_size=getattr(self.config, "recent_size", 32),
            key_size=getattr(self.config, "key_size", 992),
            tau_init=getattr(self.config, "tau_init", 1.0),
            tau_delta=getattr(self.config, "tau_delta", 0.01),
            k_seq_dim=2,
            v_seq_dim=2
        )
    bsz, q_len, _ = hidden_states.size()
    
    if q_len > 1:
        self.itr_count = 0
        
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
     
    cos, sin = self.rotary_emb(value_states, position_ids)
    
    query_states,key_states= apply_rotary_pos_emb(query_states,key_states, cos, sin, position_ids)
    
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}     
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)       
        key_states_compress, value_states_compress = self.kv_cache.evict_for_space(query_states, key_states,value_states,self.itr_count)
        
                
        past_key_value.key_cache[self.layer_idx] = key_states_compress
        past_key_value.value_cache[self.layer_idx] = value_states_compress
    if q_len == 1:
        self.itr_count += 1
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, key_states.shape[-2]):
            raise ValueError(f"Attention mask size mismatch")

    dropout_rate = self.attention_dropout if self.training else 0.0

    attn_output = flash_attn_func(
                    query_states, key_states, value_states, 
                    dropout_p=dropout_rate, 
                    causal=True
                )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)


    return attn_output, None, past_key_value



def enable_llama_pos_shift_attention(model):

    for name, module in model.named_modules():
        if isinstance(module, (LlamaAttention, LlamaSdpaAttention, LlamaFlashAttention2)):
            module.forward = types.MethodType(llama_attention_forward, module)