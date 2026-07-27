"""
Timed DecoderLayer forward with 6 CUDA Events per layer:
  pre-FFN start → pre-FFN end → Attn start → Attn end → post-FFN start → post-FFN end

Reports FFN = pre-FFN + post-FFN total.
"""

import torch
from transformers.cache_utils import Cache
from typing import Optional, Tuple

from TimeManager import time_manager


def timed_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    layer_idx = self.layer_idx
    num_layers = self.num_layers

    # === Pre-Attention FFN (RMSNorm) ===
    time_manager.record_pre_ffn_start(layer_idx)

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    time_manager.record_pre_ffn_end(layer_idx)

    # === Attention (internal events recorded in attention forward) ===
    time_manager.record_attn_start(layer_idx)

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

    time_manager.record_attn_end(layer_idx)

    # === Post-Attention FFN (residual + layernorm + MLP) ===
    time_manager.record_post_ffn_start(layer_idx)

    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    time_manager.record_post_ffn_end(layer_idx)

    # last layer advances the step counter
    if layer_idx == num_layers - 1:
        time_manager.finalize_decode_step()

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)

    return outputs