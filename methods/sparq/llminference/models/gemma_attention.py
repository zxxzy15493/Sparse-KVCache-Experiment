# Copyright 2024 Google and the HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#
# Copied from transformers==4.40.1, `transformers.models.gemma.modeling_gemma`
# Modified by Graphcore
#
# Original license (transformers):
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Light modification of GemmaAttention to enable plug-in attention sparsity.

Modifications are marked with "# MODIFIED"
"""

# mypy: ignore-errors
# fmt: off

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.models.gemma import modeling_gemma
from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb, repeat_kv


class GemmaAttention(modeling_gemma.GemmaAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
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
        # MODIFIED (copied from forward())

        # NOTE: KV repetition should be inside _attn() for GQA to behave as intended
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
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

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            raise ValueError(
                "Gemma SparQ attention requires the position_embeddings supplied by Transformers 4.57"
            )
        cos, sin = position_embeddings
        cos = cos.to(query_states.device)
        sin = sin.to(query_states.device)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # MODIFIED - moved so that attention_mask entering _attn() has the right shape
        if attention_mask is not None:  # no matter the length, we just slice it
            attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # MODIFIED (call)
        attn_output, attn_weights = self._attn(query_states, key_states, value_states, attention_mask)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights
