import inspect
from typing import Tuple

import torch


def get_qwen2_rope_cos_sin(rotary_emb, value_states: torch.Tensor, position_ids: torch.LongTensor, rotary_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
  """Call Qwen2 rotary embedding with compatibility across transformers versions."""
  try:
    sig = inspect.signature(rotary_emb.forward)
    params = sig.parameters
  except (TypeError, ValueError):
    params = {}

  if "seq_len" in params:
    return rotary_emb(value_states, seq_len=rotary_seq_len)
  if "position_ids" in params:
    return rotary_emb(value_states, position_ids)

  try:
    return rotary_emb(value_states, seq_len=rotary_seq_len)
  except TypeError as e:
    if "seq_len" not in str(e):
      raise
    return rotary_emb(value_states, position_ids)
