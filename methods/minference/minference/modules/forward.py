# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from typing import Optional, Tuple, Union, List
import time

import torch
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from ..modules.minference_forward import minference_prefill_forward
from transformers.utils import logging
logger = logging.get_logger(__name__)

def attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # will become mandatory in v4.46
    past_key_value: Cache = None,
    past_key_values: Optional[Cache] = None,
    best_pattern=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Transformers 4.57 passes the shared cache as `past_key_values` to attention.
    if past_key_values is not None:
        past_key_value = past_key_values
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    # Transformers 4.57 derives these dimensions from projection shapes and
    # no longer exposes the legacy attributes on Qwen2Attention.
    self.num_heads = getattr(self, "num_heads", self.config.num_attention_heads)
    self.num_key_value_heads = getattr(
        self, "num_key_value_heads",
        getattr(self.config, "num_key_value_heads", self.num_heads),
    )
    self.num_key_value_groups = getattr(
        self, "num_key_value_groups", self.num_heads // self.num_key_value_heads
    )
    
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)


    # [bsz, q_len, num_heads, head_dim]
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    if attention_mask is not None and attention_mask.device != query_states.device:
        attention_mask = attention_mask.to(query_states.device)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    if cos.device != query_states.device:
        cos = cos.to(query_states.device)
        sin = sin.to(query_states.device)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    is_prefill = past_key_value is None or past_key_value.get_seq_length(self.layer_idx) == 0
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin,"cos": cos,"cache_position": cache_position}

        (key_states,value_states,) = past_key_value.update(key_states,value_states,self.layer_idx,cache_kwargs,)
    
    dropout_rate = self.attention_dropout if self.training else 0.0

    # Transformers 4.57 no longer forwards `use_cache` into every attention
    # layer.  Record the cache state before update, rather than using that
    # optional argument, so one-token decode remains on FlashAttention.
    if is_prefill:
        attn_output = minference_prefill_forward(  # [bsz, num_heads, q_len, head_dim]
            query_states,
            key_states,
            value_states,
            self.layer_idx,
            best_pattern,
            attention_mask,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
    else:  # decoding
        attn_output = _flash_attention_forward(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            is_causal=self.is_causal,
        )

    assert attn_output.size(1) == q_len
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
        
    # Transformers 4.57 updates the cache in place and expects two outputs.
    return attn_output, attn_weights

def Qwen3_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,
    past_key_value: Cache = None,
    past_key_values: Optional[Cache] = None,
    best_pattern=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """MInference attention forward for Qwen3 attention layers.

    Qwen3 differs from Llama and Qwen2.5 by applying per-head Q/K RMSNorm
    after the projection reshape and before RoPE.  Omitting those normalizers
    changes the model computation even when the tensor shapes still match.
    """
    if past_key_values is not None:
        past_key_value = past_key_values
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()
    self.num_heads = getattr(self, "num_heads", self.config.num_attention_heads)
    self.num_key_value_heads = getattr(
        self, "num_key_value_heads",
        getattr(self.config, "num_key_value_heads", self.num_heads),
    )
    self.num_key_value_groups = getattr(
        self, "num_key_value_groups", self.num_heads // self.num_key_value_heads
    )

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # These are required by Qwen3 and must run before rotary embeddings.
    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)

    if attention_mask is not None and attention_mask.device != query_states.device:
        attention_mask = attention_mask.to(query_states.device)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    if cos.device != query_states.device:
        cos = cos.to(query_states.device)
        sin = sin.to(query_states.device)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    is_prefill = past_key_value is None or past_key_value.get_seq_length(self.layer_idx) == 0
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    dropout_rate = self.attention_dropout if self.training else 0.0
    if is_prefill:
        attn_output = minference_prefill_forward(
            query_states,
            key_states,
            value_states,
            self.layer_idx,
            best_pattern,
            attention_mask,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
    else:
        attn_output = _flash_attention_forward(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            is_causal=self.is_causal,
        )

    assert attn_output.size(1) == q_len
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None

def LlamaDecoderLayer_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        """Decoder-layer replacement shared by Llama/Qwen2/Qwen3.

        Qwen3 uses the same pre-norm residual layout as this implementation;
        its model-specific Q/K normalization is handled in Qwen3_attn_forward.
        """

        # ======================================  Pre Attention FFN ================================================

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # ======================================  Attention  ================================================

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # ====================================== Post Feed Forward ================================================

        # Accelerate moves module inputs to the owning shard, while the saved
        # residual can still belong to the preceding shard.
        if residual.device != hidden_states.device:
            residual = residual.to(hidden_states.device)
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if residual.device != hidden_states.device:
            residual = residual.to(hidden_states.device)
        hidden_states = residual + hidden_states

        # In Transformers 4.57 decoder layers return the hidden-state tensor
        # directly unless attention weights were explicitly requested. Cache
        # objects are updated in place by the attention module.
        if output_attentions:
            return hidden_states, self_attn_weights
        return hidden_states
