# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import functools
import inspect
import types
from typing import List, Optional

import torch
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.llama.modeling_llama import (
    ACT2FN,
)

from minference.modules.minference_forward import minference_prefill_forward

@torch.jit.script
def apply_rotary_pos_emb_glm_legacy(
    x: torch.Tensor, rope_cache: torch.Tensor
) -> torch.Tensor:
    # x: [b, np, sq, hn]
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:, :sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


def glm_forward(
    self,
    hidden_states,
    attention_mask,
    rotary_pos_emb,
    kv_cache=None,
    use_cache=True,
    best_pattern=None,
):
    mixed_x_layer = self.query_key_value(hidden_states)
    q_len = mixed_x_layer.size(1)
    bsz = mixed_x_layer.size(0)
    if self.multi_query_attention:
        (query_layer, key_layer, value_layer) = mixed_x_layer.split(
            [
                self.num_attention_heads_per_partition
                * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition
                * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition
                * self.hidden_size_per_attention_head,
            ],
            dim=-1,
        )
        query_layer = query_layer.view(
            query_layer.size()[:-1]
            + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
        )
        key_layer = key_layer.view(
            key_layer.size()[:-1]
            + (
                self.num_multi_query_groups_per_partition,
                self.hidden_size_per_attention_head,
            )
        )
        value_layer = value_layer.view(
            value_layer.size()[:-1]
            + (
                self.num_multi_query_groups_per_partition,
                self.hidden_size_per_attention_head,
            )
        )
        num_kv_groups = (
            self.num_attention_heads_per_partition
            // self.num_multi_query_groups_per_partition
        )
    else:
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
        (query_layer, key_layer, value_layer) = torch.split(
            mixed_x_layer,
            [
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ],
            dim=-1,
        )

    # [b, sq, np, hn] -> [b, np, sq, hn]
    query_layer, key_layer, value_layer = [
        k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]
    ]

    # apply relative positional encoding (rotary embedding)
    if rotary_pos_emb is not None:
        query_layer = apply_rotary_pos_emb_glm_legacy(query_layer, rotary_pos_emb)
        key_layer = apply_rotary_pos_emb_glm_legacy(key_layer, rotary_pos_emb)
    
    if kv_cache is not None:
        cache_k, cache_v = kv_cache
        key_layer = torch.cat((cache_k, key_layer), dim=2)
        value_layer = torch.cat((cache_v, value_layer), dim=2)
    if use_cache:
        if kv_cache is None:
            kv_cache = torch.cat((key_layer.unsqueeze(0).unsqueeze(0), value_layer.unsqueeze(0).unsqueeze(0)),
                                    dim=1)
        else:
            kv_cache = (key_layer, value_layer)
    else:
        kv_cache = None

    if q_len != 1:  # prefilling 
        attn_output = minference_prefill_forward(  # [bsz, num_heads, q_len, head_dim]
            query_layer,
            key_layer,
            value_layer,
            self.layer_number-1,
            best_pattern,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
    else:  # decoding
        attn_output = _flash_attention_forward(
            query_layer.transpose(1, 2),
            key_layer.transpose(1, 2),
            value_layer.transpose(1, 2),
            attention_mask,
            q_len,
            sliding_window=getattr(self, "sliding_window", None),
            is_causal=True,
        )

    assert attn_output.size(1) == q_len
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    output = self.dense(attn_output)

    return output, kv_cache


