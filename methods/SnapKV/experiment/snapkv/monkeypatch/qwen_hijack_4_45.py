import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import warnings
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from flash_attn import flash_attn_func
from transformers.utils import logging
from snapkv.monkeypatch.snapkv_utils import init_snapkv

logger = logging.get_logger(__name__)

def qwen2_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    init_snapkv(self)
    self.num_heads = getattr(self, "num_heads", self.config.num_attention_heads)
    self.num_key_value_heads = getattr(self, "num_key_value_heads", getattr(self.config, "num_key_value_heads", self.config.num_attention_heads))
    self.num_key_value_groups = getattr(self, "num_key_value_groups", self.num_heads // self.num_key_value_heads)
    self.hidden_size = getattr(self, "hidden_size", self.config.hidden_size)
    past_key_value = past_key_values
    
    output_attentions = False
    bsz, q_len, _ = hidden_states.size()
    if(q_len>1):
        self.kv_seq_len = 0
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Qwen3 normalizes projected Q/K per head before applying RoPE.
    if hasattr(self, "q_norm"):
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

    kv_seq_len = key_states.shape[-2]
    
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(f"The attention class {self.__class__.__name__} must be initialized with a layer index.")
        
        if hasattr(self, "kv_seq_len") and self.kv_seq_len != 0:
            kv_seq_len += self.kv_seq_len
        else:
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

    if position_embeddings is None:
        position_ids = position_ids if position_ids is not None else cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(value_states, position_ids)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        
    
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress= self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask, self.num_key_value_groups
            )  
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:
   
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    actual_kv_len = key_states.shape[-2] 
    

    if attention_mask is not None and attention_mask.shape[-1] > actual_kv_len:
        attention_mask = attention_mask[:, -actual_kv_len:]


    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    attn_output = flash_attn_func(
                    query_states, key_states, value_states, 
                    dropout_p=dropout_rate, 
                    causal=True
                )
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None

def prepare_inputs_for_generation_qwen2(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):

    if past_key_values is None:
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0

    if past_key_values is not None:
        past_length = self.model.layers[0].self_attn.kv_seq_len
        
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    model_inputs = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }
    return model_inputs
