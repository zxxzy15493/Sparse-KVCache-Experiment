import math
import sys
from pathlib import Path
from typing import Any

import torch

METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
  sys.path.insert(0, str(METHOD_ROOT))

from breakdown.xattention_breakdown_core import XattentionBreakdownProfiler, write_csv


class XattentionNoRepeatKVBreakdownProfiler(XattentionBreakdownProfiler):
  """Component timer for x-attention prefill without loader-side repeat_kv."""

  @staticmethod
  def _identity_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    return hidden_states

  @staticmethod
  def _dense_decode_with_original_repeat(original_repeat_kv):
    def dense_decode_attention(
      query_states: torch.Tensor,
      key_states: torch.Tensor,
      value_states: torch.Tensor,
      attention_mask,
      num_key_value_groups: int,
    ) -> torch.Tensor:
      if key_states.shape[1] != query_states.shape[1]:
        key_states = original_repeat_kv(key_states, num_key_value_groups)
        value_states = original_repeat_kv(value_states, num_key_value_groups)

      attn_weights = torch.matmul(
        query_states,
        key_states.transpose(2, 3),
      ) / math.sqrt(query_states.shape[-1])
      if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

      attn_weights = torch.softmax(
        attn_weights,
        dim=-1,
        dtype=torch.float32,
      ).to(query_states.dtype)
      return torch.matmul(attn_weights, value_states)

    return dense_decode_attention

  def patch_xattn_ops(self, loader_module: Any) -> None:
    if self._patched:
      return

    import xattn.src.Xattention_no_repeatkv as xattn_mod

    wrapped_prefill = self._wrap_callable(
      xattn_mod.Xattention_prefill,
      "xattention_no_repeatkv_prefill",
      "xattention_prefill",
    )
    self._patch_attr(xattn_mod, "Xattention_prefill", wrapped_prefill)
    if hasattr(loader_module, "Xattention_prefill"):
      self._patch_attr(loader_module, "Xattention_prefill", wrapped_prefill)

    original_repeat_kv = getattr(loader_module, "repeat_kv", None)
    if original_repeat_kv is not None:
      self._patch_attr(loader_module, "repeat_kv", self._identity_repeat_kv)
      if hasattr(loader_module, "dense_decode_attention"):
        self._patch_attr(
          loader_module,
          "dense_decode_attention",
          self._dense_decode_with_original_repeat(original_repeat_kv),
        )

    self._patch_attr(
      xattn_mod,
      "xattn_estimate",
      self._wrap_callable(xattn_mod.xattn_estimate, "xattn_estimate"),
    )
    self._patch_attr(
      xattn_mod,
      "xattn_estimate_no_repeatkv",
      self._wrap_callable(
        xattn_mod.xattn_estimate_no_repeatkv,
        "xattn_estimate_no_repeatkv",
        "retrieve",
      ),
    )
    self._patch_attr(
      xattn_mod,
      "find_blocks_chunked",
      self._wrap_callable(xattn_mod.find_blocks_chunked, "find_blocks_chunked"),
    )
    self._patch_attr(
      xattn_mod,
      "block_sparse_attn_func",
      self._wrap_callable(
        xattn_mod.block_sparse_attn_func,
        "block_sparse_attention",
        "attn",
      ),
    )

    loader_flash_attn = getattr(loader_module, "flash_attn_func", None)
    if loader_flash_attn is not None:
      self._patch_attr(
        loader_module,
        "flash_attn_func",
        self._wrap_callable(
          loader_flash_attn,
          "flash_attn_decode",
          "attn",
        ),
      )

    self._patched = True
