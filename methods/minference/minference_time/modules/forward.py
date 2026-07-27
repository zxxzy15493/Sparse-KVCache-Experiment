# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from typing import Optional, Tuple
import time

import torch
from transformers.cache_utils import Cache, PretrainedConfig
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.cache_utils import StaticCache
from transformers.utils import logging
logger = logging.get_logger(__name__)

from ..modules.minference_forward import minference_prefill_forward
from ..TimeManager import time_manager, full_time_manager


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
    best_pattern=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()
    
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)


    # [bsz, q_len, num_heads, head_dim]
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    if cos.device != query_states.device:
        cos = cos.to(query_states.device)
        sin = sin.to(query_states.device)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    time_manager.forward_pre_ffn_end_event[time_manager.decode_step * time_manager.num_layers + self.layer_idx].record()

    time_manager.write_kv_start_event[time_manager.decode_step * time_manager.num_layers + self.layer_idx].record()
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin,"cos": cos,"cache_position": cache_position}

        (key_states,value_states,) = past_key_value.update(key_states,value_states,self.layer_idx,cache_kwargs,)

    time_manager.write_kv_end_event[time_manager.decode_step * time_manager.num_layers + self.layer_idx].record()
    
    dropout_rate = self.attention_dropout if self.training else 0.0

    if not use_cache or q_len == past_key_value.get_seq_length(
        self.layer_idx
    ):
        time_manager.forward_attn_start_event[time_manager.decode_step * time_manager.num_layers + self.layer_idx].record()
        attn_output = minference_prefill_forward(  # [bsz, num_heads, q_len, head_dim]
            query_states,
            key_states,
            value_states,
            self.layer_idx,
            best_pattern
        )
        time_manager.forward_attn_end_event[time_manager.decode_step * time_manager.num_layers + self.layer_idx].record()
        attn_output = attn_output.transpose(1, 2).contiguous()
    else:  # decoding
        time_manager.forward_attn_start_event[time_manager.decode_step * time_manager.num_layers + self.layer_idx].record()
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
        time_manager.forward_attn_end_event[time_manager.decode_step * time_manager.num_layers + self.layer_idx].record()

    time_manager.forward_post_ffn_start_event[time_manager.decode_step * time_manager.num_layers + self.layer_idx].record()
    assert attn_output.size(1) == q_len
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
        
    return attn_output, attn_weights, past_key_value


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

        # ======================================  Pre Attention FFN ================================================

        time_manager.forward_pre_ffn_start_event[time_manager.decode_step * self.num_layers + self.layer_idx].record()

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # time_manager.forward_preAttn_ffn_end_event[time_manager.decode_step * self.num_layers + self.layer_idx].record()

        # ======================================  Attention  ================================================

        # time_manager.forward_attn_start_event[time_manager.decode_step * self.num_layers + self.layer_idx].record()

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
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

        # time_manager.forward_attn_end_event[time_manager.decode_step * self.num_layers + self.layer_idx].record()


        # ====================================== Post Feed Forward ================================================


        # time_manager.forward_postAttn_ffn_start_event[time_manager.decode_step * self.num_layers + self.layer_idx].record()


        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        time_manager.forward_post_ffn_end_event[time_manager.decode_step * self.num_layers + self.layer_idx].record()

        if self.layer_idx == self.num_layers - 1:
            time_manager.decode_step += 1

        return outputs


def full_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    full_time_manager.forward_pre_ffn_end_event[full_time_manager.decode_step * full_time_manager.num_layers + self.layer_idx].record()

    full_time_manager.write_kv_start_event[full_time_manager.decode_step * full_time_manager.num_layers + self.layer_idx].record()

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    full_time_manager.write_kv_end_event[full_time_manager.decode_step * full_time_manager.num_layers + self.layer_idx].record()

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    
    full_time_manager.forward_attn_start_event[full_time_manager.decode_step * full_time_manager.num_layers + self.layer_idx].record()

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=getattr(self, "sliding_window", None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
    )

    full_time_manager.forward_attn_end_event[full_time_manager.decode_step * full_time_manager.num_layers + self.layer_idx].record()

    full_time_manager.forward_post_ffn_start_event[full_time_manager.decode_step * full_time_manager.num_layers + self.layer_idx].record()

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def Full_LlamaDecoderLayer_forward(
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

        # ======================================  Pre Attention FFN ================================================

        full_time_manager.forward_pre_ffn_start_event[full_time_manager.decode_step * self.num_layers + self.layer_idx].record()

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # full_time_manager.forward_preAttn_ffn_end_event[full_time_manager.decode_step * self.num_layers + self.layer_idx].record()

        # ======================================  Attention  ================================================

        # time_manager.forward_attn_start_event[time_manager.decode_step * self.num_layers + self.layer_idx].record()

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
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

        # time_manager.forward_attn_end_event[time_manager.decode_step * self.num_layers + self.layer_idx].record()


        # ====================================== Post Feed Forward ================================================


        # time_manager.forward_postAttn_ffn_start_event[time_manager.decode_step * self.num_layers + self.layer_idx].record()


        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        full_time_manager.forward_post_ffn_end_event[full_time_manager.decode_step * self.num_layers + self.layer_idx].record()

        if self.layer_idx == self.num_layers - 1:
            full_time_manager.decode_step += 1

        return outputs
