import math
from typing import Optional, Tuple, List
import types
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    "enable_glm_pos_shift_attention",
    "disable_glm_pos_shift_attention",
    "validate_glm_patch",
]


def glm_pos_shift_attention_forward(
    self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
):
    import sys
    modeling_chatglm = sys.modules[self.__class__.__module__]
    split_tensor_along_last_dim = modeling_chatglm.split_tensor_along_last_dim
    apply_rotary_pos_emb = modeling_chatglm.apply_rotary_pos_emb

    if not hasattr(self, "kv_cache_evictor"):
        from streaming_llm.kv_cache import StartRecentKVCache
        self.kv_cache_evictor = StartRecentKVCache(
            start_size=getattr(self.config, "start_size", 16),
            recent_size=getattr(self.config, "recent_size", 1008),
            k_seq_dim=2,
            v_seq_dim=2
        )
        
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


    if kv_cache is not None:
        cache_k, cache_v = kv_cache
        if isinstance(kv_cache, tuple):
            if len(cache_k.shape) == 5:
                cache_k, cache_v = cache_k[0], cache_v[0]
        key_layer = torch.cat((cache_k, key_layer), dim=2)
        value_layer = torch.cat((cache_v, value_layer), dim=2)

    key_states_compress, value_states_compress = self.kv_cache_evictor.evict_for_space(key_layer, value_layer)


    if use_cache:
        if kv_cache is None:
            # First handling kv_cache initialization with compressed states
            kv_cache = torch.cat((key_states_compress.unsqueeze(0).unsqueeze(0), value_states_compress.unsqueeze(0).unsqueeze(0)),
                                 dim=1)
        else:
            kv_cache = (key_states_compress, value_states_compress)
    else:
        kv_cache = None

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

    context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
    output = self.dense(context_layer)

    return output, kv_cache


def enable_glm_pos_shift_attention(model):
    

    if not hasattr(model, "transformer") or not hasattr(model.transformer, "encoder"):
        raise AttributeError("Expected model.transformer.encoder.layers in GLM model.")

    layers = model.transformer.encoder.layers
    if len(layers) == 0:
        raise RuntimeError("No encoder layers found in model.transformer.encoder.layers")

    patched = 0
    for i, layer in enumerate(layers, start=1):
        if not hasattr(layer, "self_attention"):
            raise AttributeError(f"Layer {i} has no self_attention")

        module = layer.self_attention
        module.config = model.config
        module.layer_number = i

        if not hasattr(module, "flash_forward"):
            module.flash_forward = module.forward

        module.forward = types.MethodType(glm_pos_shift_attention_forward, module)
        patched += 1

    if patched == 0:
        raise RuntimeError("No GLM SelfAttention modules were patched.")

    return model


def disable_glm_pos_shift_attention(model):
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "encoder"):
        return model

    for layer in model.transformer.encoder.layers:
        if not hasattr(layer, "self_attention"):
            continue

        module = layer.self_attention
        if hasattr(module, "flash_forward"):
            module.forward = module.flash_forward

    return model


def validate_glm_patch(model):
    rows = []
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "encoder"):
        return rows

    for i, layer in enumerate(model.transformer.encoder.layers, start=1):
        if not hasattr(layer, "self_attention"):
            continue

        module = layer.self_attention
        forward_obj = module.forward
        forward_name = getattr(getattr(forward_obj, "__func__", None), "__name__", type(forward_obj).__name__)

        rows.append(
            {
                "layer": i,
                "patched": hasattr(module, "flash_forward"),
                "forward_name": forward_name,
                "layer_number": getattr(module, "layer_number", None),
            }
        )

    return rows
