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
import triton
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
  repeat_kv,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)
from transformers.utils import (
  add_start_docstrings,
  add_start_docstrings_to_model_forward,
  is_flash_attn_2_available,
  is_flash_attn_greater_or_equal_2_10,
  logging,
  replace_return_docstrings,
)

from flex_prefill.modules.llama.apply_rope import triton_apply_rotary_pos_emb
from flex_prefill.modules.qwen2.rotary_compat import get_qwen2_rope_cos_sin
from flex_prefill.ops.flex_prefill_attention import flex_prefill_attention


def qwen2_flex_prefill_attention_forward(
  self,
  hidden_states: torch.Tensor,
  position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
  attention_mask: Optional[torch.Tensor] = None,
  past_key_values: Optional[Cache] = None,
  cache_position: Optional[torch.LongTensor] = None,
  **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

  past_key_values = kwargs.pop("past_key_value", past_key_values)
  position_ids = kwargs.pop("position_ids", None)

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

  # Qwen3 uses per-head RMSNorm before the model-level RoPE embeddings.
  if hasattr(self, "q_norm"):
    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)

  if past_key_values is not None:
    if self.layer_idx is None:
      raise ValueError(
        "Qwen attention needs layer_idx set when using the KV cache."
      )

  if position_embeddings is None:
    raise ValueError(
      "Qwen FlexPrefill requires position_embeddings from Transformers 4.57"
    )
  cos, sin = position_embeddings

  if cos.device != query_states.device:
    cos = cos.to(query_states.device)
  if sin.device != query_states.device:
    sin = sin.to(query_states.device)

  query_states, key_states = apply_rotary_pos_emb(
    query_states, key_states, cos, sin, position_ids
  )

  # )

  #   )

  if past_key_values is not None:
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_values.update(
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

  layer_id = int(getattr(self, "layer_idx", -1))
  model_name="Qwen"

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
    attention_mask=attention_mask,

  )


  attn_output = attn_output.reshape(bsz, q_len, self.o_proj.in_features).contiguous()
  attn_output = self.o_proj(attn_output)

  return attn_output, None
