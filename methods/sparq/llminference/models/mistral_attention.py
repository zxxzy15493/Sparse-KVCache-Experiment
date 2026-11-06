# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#
# Copied from transformers==4.40.1, `transformers.models.mistral.modeling_mistral`
# Modified by Graphcore

"""Light modification of MistralAttention to enable plug-in attention sparsity.

Modifications are marked with "# MODIFIED"
"""

# mypy: ignore-errors
# fmt: off

import math
import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.models.mistral import modeling_mistral
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv


class MistralAttention(modeling_mistral.MistralAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        # Transformers 4.57 keeps some of these values on the config only,
        # while SparQ's plug-in attention needs them as module attributes.
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = config.hidden_size

    # MODIFIED (added)
    def _attn(
        self,
        query_states: Tensor,
        key_states: Tensor,
        value_states: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # MODIFIED (added)
        bsz, _, q_len, _ = query_states.shape
        kv_seq_len = key_states.shape[-2]

        # MODIFIED (copied from forward())

        # repeat k/v heads if n_kv_heads < n_heads
        # NOTE: KV repetition should be inside _attn() for GQA to behave as intended
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # MODIFIED (commented out) *per-head masking is needed for ann/eviction masks*
        if attention_mask is not None:
            # if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            #     raise ValueError(
            #         f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            #     )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        past_key_values = kwargs.pop("past_key_value", past_key_values)
        output_attentions = bool(kwargs.pop("output_attentions", False))

        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            raise ValueError(
                "Mistral SparQ attention requires the position_embeddings supplied by Transformers 4.57"
            )
        cos, sin = position_embeddings
        cos = cos.to(query_states.device)
        sin = sin.to(query_states.device)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            if self.layer_idx is None:
                raise ValueError(
                    "MistralAttention needs layer_idx set when using the KV cache."
                )
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        if attention_mask is not None:
            attention_mask = attention_mask[..., : key_states.shape[-2]].to(query_states.device)

        # MODIFIED (call)
        attn_output, attn_weights = self._attn(query_states, key_states, value_states, attention_mask)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.o_proj.in_features)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights
