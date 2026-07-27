#
#
#
#
"""PyTorch Qwen2 model."""
import inspect
import math
import os
import time
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from flash_attn import flash_attn_func
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
  _prepare_4d_causal_attention_mask,
  _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
  BaseModelOutputWithPast,
  CausalLMOutputWithPast,
  SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
  apply_rotary_pos_emb,
  logger,
  repeat_kv,
)
from transformers.utils import (
  add_start_docstrings,
  add_start_docstrings_to_model_forward,
  is_flash_attn_2_available,
  is_flash_attn_greater_or_equal_2_10,
  logging,
  replace_return_docstrings,
)

from flex_prefill.modules.qwen2.rotary_compat import get_qwen2_rope_cos_sin
from flex_prefill.ops.streaming_llm_attention import streaming_llm_attention

if is_flash_attn_2_available():
  from flash_attn import flash_attn_func, flash_attn_varlen_func
  from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input # noqa

  _flash_supports_window_size = "window_size" in list(
    inspect.signature(flash_attn_func).parameters
  )


def qwen2_streaming_llm_attention_forward(
  self,
  hidden_states: torch.Tensor,
  attention_mask: Optional[torch.Tensor] = None,
  position_ids: Optional[torch.LongTensor] = None,
  past_key_value: Optional[Cache] = None,
  output_attentions: bool = False,
  use_cache: bool = False,
  **kwargs,
):
  if "padding_mask" in kwargs:
    warnings.warn(
      "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
    )

    attention_mask = kwargs.pop("padding_mask")
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

  kv_seq_len = key_states.shape[-2]
  if past_key_value is not None:
    if self.layer_idx is None:
      raise ValueError(
        f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
        "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
        "with a layer index."
      )
    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

  rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
  cos, sin = get_qwen2_rope_cos_sin(
    self.rotary_emb, value_states, position_ids, rotary_seq_len
  )

  query_states, key_states = apply_rotary_pos_emb(
    query_states, key_states, cos, sin, position_ids
  )

  use_sliding_windows = (
    _flash_supports_window_size
    and getattr(self.config, "sliding_window", None) is not None
    and kv_seq_len > self.config.sliding_window
    and self.config.use_sliding_window
  )

  if not _flash_supports_window_size:
    logger.warning_once(
      "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
      " make sure to upgrade flash-attn library."
    )

  if past_key_value is not None:
    cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
    if (
      getattr(self.config, "sliding_window", None) is not None
      and kv_seq_len > self.config.sliding_window
      and cache_has_contents
    ):
      slicing_tokens = 1 - self.config.sliding_window

      past_key = past_key_value[self.layer_idx][0]
      past_value = past_key_value[self.layer_idx][1]

      past_key = past_key[:, :, slicing_tokens:, :].contiguous()
      past_value = past_value[:, :, slicing_tokens:, :].contiguous()

      if past_key.shape[-2] != self.config.sliding_window - 1:
        raise ValueError(
          f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
          f" {past_key.shape}"
        )

      if attention_mask is not None:
        attention_mask = attention_mask[:, slicing_tokens:]
        attention_mask = torch.cat(
          [attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1
        )

    cache_kwargs = {"sin": sin, "cos": cos} # Specific to RoPE models
    key_states, value_states = past_key_value.update(
      key_states, value_states, self.layer_idx, cache_kwargs
    )

  dropout_rate = 0.0 if not self.training else self.attention_dropout

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

  query_states = query_states.transpose(1, 2)
  key_states = key_states.transpose(1, 2)
  value_states = value_states.transpose(1, 2)

  if query_states.shape[1] > 1:
    attn_output = streaming_llm_attention(
      query_states,
      key_states,
      value_states,
      global_window=self.config.global_window,
      local_window=self.config.local_window,
    )
  else:
    attn_output = flash_attn_func(query_states, key_states, value_states)

  attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
  attn_output = self.o_proj(attn_output)

  if not output_attentions:
    attn_weights = None

  return attn_output, attn_weights, past_key_value
