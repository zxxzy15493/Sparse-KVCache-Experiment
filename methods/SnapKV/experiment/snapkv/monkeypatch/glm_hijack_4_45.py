import math
from typing import Optional, Tuple, List
import types
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from snapkv.monkeypatch.snapkv_utils import init_snapkv

__all__ = ["enable_snapkv_glm_attention"]

step_layer_recalls = {} 

def glm_snapkv_attention_forward(
    self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
):
    init_snapkv(self)
    import sys
    modeling_chatglm = sys.modules[self.__class__.__module__]
    split_tensor_along_last_dim = modeling_chatglm.split_tensor_along_last_dim
    apply_rotary_pos_emb = modeling_chatglm.apply_rotary_pos_emb

    bsz, q_len, _ = hidden_states.size()

        
    mixed_x_layer = self.query_key_value(hidden_states)

    if self.multi_query_attention:
        (query_layer, key_layer, value_layer) = mixed_x_layer.split(
            [
                self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
            ],
            dim=-1,
        )
        query_layer = query_layer.view(
            query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        )
        key_layer = key_layer.view(
            key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
        value_layer = value_layer.view(
            value_layer.size()[:-1]
            + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
    else:
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

    query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

    if rotary_pos_emb is not None:
        query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
        key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

    if self.multi_query_attention:
        key_layer = key_layer.unsqueeze(2)
        key_layer = key_layer.expand(
            -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
        )
        key_layer = key_layer.contiguous().view(
            key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
        )
        value_layer = value_layer.unsqueeze(2)
        value_layer = value_layer.expand(
            -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
        )
        value_layer = value_layer.contiguous().view(
            value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
        )

    if kv_cache is not None:
        cache_k, cache_v = kv_cache
        if isinstance(kv_cache, tuple):
            if len(cache_k.shape) == 5:
                cache_k, cache_v = cache_k[0], cache_v[0]
        key_layer = torch.cat((cache_k, key_layer), dim=2)
        value_layer = torch.cat((cache_v, value_layer), dim=2)


    if kv_cache is None: # Prefill compression
        # Pass num_key_value_groups=1 because we already expanded key/value to match num_heads
        key_states_compress, value_states_compress, select_idx = self.kv_cluster.update_kv(key_layer, query_layer, value_layer, attention_mask, 1)
        self.select_idx = select_idx
    else: # Decoding standard append
        key_states_compress = key_layer
        value_states_compress = value_layer

    if use_cache:
        if kv_cache is None:
            # First handling kv_cache initialization with compressed states
            kv_cache = torch.cat((key_states_compress.unsqueeze(0).unsqueeze(0), value_states_compress.unsqueeze(0).unsqueeze(0)),
                                 dim=1)
        else:
            kv_cache = (key_states_compress, value_states_compress)
    else:
        kv_cache = None

    context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
    output = self.dense(context_layer)

    return output, kv_cache


def enable_snapkv_glm_attention(model):

    for name, module in model.named_modules():
        if module.__class__.__name__ == "SelfAttention":
            module.config = model.config
            module.forward = types.MethodType(glm_snapkv_attention_forward, module)


