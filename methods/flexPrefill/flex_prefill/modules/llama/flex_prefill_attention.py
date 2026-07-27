#
#
#
#
import os
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
import triton
from transformers.cache_utils import Cache, StaticCache
from transformers.models.llama.modeling_llama import (
  apply_rotary_pos_emb,
  logger,
  repeat_kv,
)

from flex_prefill.modules.llama.apply_rope import triton_apply_rotary_pos_emb
from flex_prefill.ops.flex_prefill_attention import flex_prefill_attention


def llama_flex_prefill_attention_forward(
  self,
  hidden_states: torch.Tensor,
  attention_mask: Optional[torch.LongTensor] = None,
  position_ids: Optional[torch.LongTensor] = None,
  past_key_value: Optional[Cache] = None,
  output_attentions: bool = False,
  use_cache: bool = False,
  cache_position: Optional[torch.LongTensor] = None,
  position_embeddings: Optional[
    Tuple[torch.Tensor, torch.Tensor]
  ] = None, # will become mandatory in v4.45
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

  query_states = query_states.view(
    bsz, q_len, self.num_heads, self.head_dim
  ).transpose(1, 2)
  key_states = key_states.view(
    bsz, q_len, self.num_key_value_heads, self.head_dim
  ).transpose(1, 2)
  value_states = value_states.view(
    bsz, q_len, self.num_key_value_heads, self.head_dim
  ).transpose(1, 2)

  if position_embeddings is None:
    logger.warning_once(
      "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
      "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
      "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
      "removed and `position_embeddings` will be mandatory."
    )
    cos, sin = self.rotary_emb(value_states, position_ids)
  else:
    cos, sin = position_embeddings

  if triton.__version__ == "3.0.0":
    query_states, key_states = triton_apply_rotary_pos_emb(
      query_states, key_states, cos, sin
    )
  else:
    query_states, key_states = apply_rotary_pos_emb(
      query_states, key_states, cos, sin
    )

  if past_key_value is not None:
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_value.update(
      key_states, value_states, self.layer_idx, cache_kwargs
    )

  query_states = query_states.transpose(1, 2)
  key_states = key_states.transpose(1, 2)
  value_states = value_states.transpose(1, 2)

  dropout_rate = self.attention_dropout if self.training else 0.0


  input_dtype = query_states.dtype
  if input_dtype == torch.float32:
    if torch.is_autocast_enabled():
      target_dtype = torch.get_autocast_gpu_dtype()
    elif hasattr(self.config, "_pre_quantization_dtype"):
      target_dtype = self.config._pre_quantization_dtype
    else:
      target_dtype = self.q_proj.weight.dtype

    logger.warning_once(
      f"The input hidden states seems to be silently casted in float32, this might be related to"
      f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
      f" {target_dtype}."
    )

    query_states = query_states.to(target_dtype)
    key_states = key_states.to(target_dtype)
    value_states = value_states.to(target_dtype)



  layer_id = int(getattr(self, "layer_idx", -1))
  model_name="Llama"
  attn_output = flex_prefill_attention(
    model_name,
    layer_id,
    query_states,
    key_states,
    value_states,
    gamma=self.config.flex_prefill_gamma,
    tau=self.config.flex_prefill_tau,
    block_size=self.config.block_size,
    task=getattr(self.config, "task", None),
    min_budget=getattr(self.config, "flex_prefill_min_budget", None),
    max_budget=getattr(self.config, "flex_prefill_max_budget", None),
    type=getattr(self.config, "type", None),
    save_path=getattr(self.config, "save_path", None),
    sample_id=getattr(self.config, "sample_id", ""),
  )

  attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
  attn_output = self.o_proj(attn_output)

  if not output_attentions:
    attn_weights = None

  return attn_output, attn_weights, past_key_value
