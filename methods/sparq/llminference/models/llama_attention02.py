#
#


import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from transformers.cache_utils import Cache
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

try:
  from flash_attn import flash_attn_func
  FLASH_ATTN_AVAILABLE = True
except ImportError:
  flash_attn_func = None
  FLASH_ATTN_AVAILABLE = False


class LlamaAttention(modeling_llama.LlamaAttention):
  # -------------------------
  # -------------------------
  def _attn(
    self,
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    attention_mask: Optional[Tensor],
  ) -> Tuple[Tensor, Optional[Tensor]]:
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    bsz, _, q_len, _ = query_states.shape
    kv_len = key_states.shape[-2]
    dropout_p = self.attention_dropout if self.training else 0.0

    if FLASH_ATTN_AVAILABLE and bsz == 1:
      q = query_states.transpose(1, 2) # [b, q, h, d]
      k = key_states.transpose(1, 2)  # [b, kv, h, d]
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

    if attention_mask is None:
      key_pos = torch.arange(kv_len, device=query_states.device)[None, :]
      query_pos = torch.arange(q_len, device=query_states.device)[:, None] + (kv_len - q_len)
      causal = key_pos > query_pos
      min_value = torch.finfo(query_states.dtype).min
      attention_mask = causal[None, None, :, :].to(dtype=query_states.dtype).masked_fill(causal[None, None, :, :], min_value)

    attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output, attn_weights

  # -------------------------
  # -------------------------
  @staticmethod
  def _apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, position_ids, unsqueeze_dim: int = 1):
    try:
      return apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=unsqueeze_dim)
    except TypeError:
      try:
        return apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim)
      except TypeError:
        return apply_rotary_pos_emb(q, k, cos, sin)

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
  def _make_position_ids(
    bsz: int,
    q_len: int,
    kv_seq_len: int,
    *,
    device: torch.device,
    cache_position: Optional[torch.LongTensor],
  ) -> torch.LongTensor:
    if cache_position is not None:
      position_ids = cache_position.unsqueeze(0)
      if position_ids.shape[0] != bsz:
        position_ids = position_ids.expand(bsz, -1)
      return position_ids

    start = max(kv_seq_len - q_len, 0)
    return torch.arange(start, start + q_len, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)

  @staticmethod
  def _expand_2d_mask_to_4d(attention_mask: Tensor, q_len: int, dtype: torch.dtype) -> Tensor:
    if attention_mask.dim() != 2:
      return attention_mask
    min_value = torch.finfo(dtype).min
    expanded = (1.0 - attention_mask[:, None, None, :].to(dtype=dtype)) * min_value
    return expanded.expand(-1, 1, q_len, -1)

  def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # MODIFIED: transformers 4.45+
    **kwargs,
  ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    if "padding_mask" in kwargs:
      warnings.warn(
        "Passing `padding_mask` is deprecated; use `attention_mask` instead.",
        stacklevel=2,
      )

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
      key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
      query_slices = self.q_proj.weight.split(
        (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
      )
      key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
      value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

      query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
      query_states = torch.cat(query_states, dim=-1)

      key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
      key_states = torch.cat(key_states, dim=-1)

      value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
      value_states = torch.cat(value_states, dim=-1)
    else:
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
          "LlamaAttention needs layer_idx set when using kv cache (past_key_value is not None)."
        )
      try:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
      except AttributeError:
        pass

    if position_embeddings is None:
      if position_ids is None:
        position_ids = self._make_position_ids(
          bsz,
          q_len,
          kv_seq_len,
          device=hidden_states.device,
          cache_position=cache_position,
        )
      cos, sin = self._compute_rope(self.rotary_emb, value_states, position_ids, kv_seq_len)
    else:
      cos, sin = position_embeddings

    query_states, key_states = self._apply_rope(query_states, key_states, cos, sin, position_ids, unsqueeze_dim=1)

    if past_key_value is not None:
      cache_kwargs = {"sin": sin, "cos": cos}
      if cache_position is not None:
        cache_kwargs["cache_position"] = cache_position
      key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if isinstance(attention_mask, torch.Tensor):
      attention_mask = self._expand_2d_mask_to_4d(attention_mask, q_len, hidden_states.dtype)
      attention_mask = attention_mask[..., : key_states.shape[-2]]

    attn_output, attn_weights = self._attn(query_states, key_states, value_states, attention_mask)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
      raise ValueError(
        f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, "
        f"but is {attn_output.size()}"
      )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

    if self.config.pretraining_tp > 1:
      attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
      o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
      attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
      attn_output = self.o_proj(attn_output)

    if not output_attentions:
      attn_weights = None

    return attn_output, attn_weights, past_key_value
