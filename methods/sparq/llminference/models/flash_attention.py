"""Required FlashAttention backend for SparQ; unsupported inputs fail explicitly."""

import torch

try:
  from flash_attn import flash_attn_func
except (ImportError, OSError):
  flash_attn_func = None


def flash_attention(
  query,
  key,
  value,
  *,
  attention_mask=None,
  dropout_p=0.0,
  softmax_scale=None,
  causal=False,
):
  if flash_attn_func is None:
    raise RuntimeError("SparQ requires flash_attn_func; install FlashAttention in the active Python environment.")
  if attention_mask is not None:
    raise NotImplementedError("SparQ FlashAttention does not support attention_mask; masked/padded inputs are unsupported.")
  if not query.is_cuda:
    raise RuntimeError("SparQ FlashAttention requires CUDA tensors.")
  if query.dtype not in (torch.float16, torch.bfloat16):
    raise TypeError("SparQ FlashAttention requires float16 or bfloat16 tensors.")
  return flash_attn_func(
    query,
    key,
    value,
    dropout_p=dropout_p,
    softmax_scale=softmax_scale,
    causal=causal,
    return_attn_probs=False,
  )
