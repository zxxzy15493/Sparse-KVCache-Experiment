#
#


import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from transformers.cache_utils import Cache
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

try:
  from flash_attn import flash_attn_func

  FLASH_ATTN_AVAILABLE = True
except ImportError:
  FLASH_ATTN_AVAILABLE = False


class Qwen2Attention(modeling_qwen2.Qwen2Attention):
  # -------------------------
  # -------------------------
  def _attn(
    self,
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    attention_mask: Optional[Tensor],
  ) -> Tuple[Tensor, Tensor]:
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    bsz, _, q_len, _ = query_states.shape
    kv_len = key_states.shape[-2]
    dropout_p = self.attention_dropout if self.training else 0.0

    if FLASH_ATTN_AVAILABLE and query_states.shape[0] == 1:
      q = query_states.transpose(1, 2)
      k = key_states.transpose(1, 2)
      v = value_states.transpose(1, 2)
      attn_output = flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        causal=True,
        return_attn_probs=False,
      ).transpose(1, 2)
      return attn_output, None

    try:
      is_causal = attention_mask is None and q_len > 1
      attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
      )
      return attn_output, None
    except Exception:
      pass

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is None and q_len > 1:
      attention_mask = self._build_causal_mask(
        bsz,
        q_len,
        kv_len,
        device=query_states.device,
        dtype=query_states.dtype,
      )

    if attention_mask is not None:
      attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output, attn_weights

  # -------------------------
  # -------------------------
  @staticmethod
  def _apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, position_ids, unsqueeze_dim: int):
    try:
      return apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=unsqueeze_dim)
    except TypeError:
      try:
        return apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim)
      except TypeError:
        return apply_rotary_pos_emb(q, k, cos, sin, position_ids)

  @staticmethod
  def _compute_rope(rotary_emb: nn.Module, value_states: Tensor, position_ids, kv_seq_len: int):
    try:
      return rotary_emb(value_states, position_ids)
    except TypeError:
      try:
        return rotary_emb(value_states, seq_len=kv_seq_len)
      except TypeError:
        return rotary_emb(value_states, kv_seq_len)

  @staticmethod
  def _build_causal_mask(
    bsz: int,
    q_len: int,
    kv_seq_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
  ) -> Tensor:
    key_pos = torch.arange(kv_seq_len, device=device)[None, :]
    query_pos = torch.arange(q_len, device=device)[:, None] + (kv_seq_len - q_len)
    mask = key_pos > query_pos
    mask = mask[None, None, :, :].expand(bsz, 1, q_len, kv_seq_len)
    min_value = torch.finfo(dtype).min if dtype.is_floating_point else -1e4
    return mask.to(dtype=dtype).masked_fill(mask, min_value)

  def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None, # MODIFIED: keep parity with newer decoder APIs
    **kwargs,
  ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:

    if "padding_mask" in kwargs:
      warnings.warn(
        "Passing `padding_mask` is deprecated; use `attention_mask` instead."
      )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    past_key_value = getattr(self, "past_key_value", past_key_value)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
      if self.layer_idx is None:
        raise ValueError(
          "Qwen2Attention needs layer_idx set when using kv cache (past_key_value is not None)."
        )
      kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = self._compute_rope(self.rotary_emb, value_states, position_ids, kv_seq_len)
    
    query_states, key_states = self._apply_rope(query_states, key_states, cos, sin, position_ids, unsqueeze_dim=1)

    if past_key_value is not None:

      cache_kwargs = {"sin": sin, "cos": cos}
      if cache_position is not None:
        cache_kwargs["cache_position"] = cache_position
      key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if isinstance(attention_mask, torch.Tensor):
      attention_mask = attention_mask[..., : key_states.shape[-2]]

    attn_output, attn_weights = self._attn(query_states, key_states, value_states, attention_mask)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
      raise ValueError(
        f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}"
      )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
      attn_weights = None

    return attn_output, attn_weights, past_key_value
