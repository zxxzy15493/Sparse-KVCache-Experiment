"""
Quest attention patch for local GLM-4-9B-Chat-1M.

Design:
- Patch GLM's CoreAttention.forward in-place.
- Keep original SelfAttention logic untouched:
 QKV projection / RoPE / KV cache concat / MQA expansion stay in official code.
- Only replace the final dense attention math during decode (q_len == 1).

Assumptions:
- Uses model from HuggingFace: zai-org/glm-4-9b-chat-1m
- That repo contains modeling_chatglm.py
- Model is loaded with trust_remote_code=True
- This version currently targets eager/CoreAttention backend

Typical usage:
  from quest_glm_attention import enable_quest_glm_attention_eval, validate_glm_patch

  model = AutoModelForCausalLM.from_pretrained(
    "zai-org/glm-4-9b-chat-1m",
    trust_remote_code=True,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto",
  ).eval()

  class Args:
    token_budget = 1024
    chunk_size = 16
    quest_start_layer = 3

  args = Args()
  enable_quest_glm_attention_eval(model, args)
  print(validate_glm_patch(model))
"""

import math
import types

from typing import Optional

import torch
from torch import nn


def _dtype_min(dtype: torch.dtype) -> float:
  return torch.finfo(dtype).min


def _pad_to_chunk_length(seq_length: int, chunk_size: int) -> int:
  return chunk_size - ((seq_length - 1) % chunk_size + 1)



def local_heavy_hitter_mask(
  attn_weights: torch.Tensor,
  token_budget: int,
  chunk_size: int,
) -> torch.Tensor:
  """
  attn_weights: (bs, n_heads, q, k)
  return: bool mask of shape (bs, n_heads, q, k), True means selected
  """
  seq_length = attn_weights.shape[-1]
  padding_length = _pad_to_chunk_length(seq_length, chunk_size)

  if padding_length > 0:
    pad = torch.full(
      (
        attn_weights.shape[0],
        attn_weights.shape[1],
        attn_weights.shape[2],
        padding_length,
      ),
      fill_value=_dtype_min(attn_weights.dtype),
      device=attn_weights.device,
      dtype=attn_weights.dtype,
    )
    attn_weights = torch.cat([attn_weights, pad], dim=-1)

  chunk_attn_weights = attn_weights.reshape(
    attn_weights.shape[0],
    attn_weights.shape[1],
    attn_weights.shape[2],
    attn_weights.shape[3] // chunk_size,
    chunk_size,
  ).amax(dim=-1)

  k_chunks = min(max(3, math.ceil(token_budget / chunk_size)), chunk_attn_weights.size(-1))
  _, topk = chunk_attn_weights.topk(k=k_chunks, dim=-1)

  topk = (
    topk.unsqueeze(-1).repeat(1, 1, 1, 1, chunk_size) * chunk_size
    + torch.arange(chunk_size, device=topk.device)
  )
  topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)

  mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
  mask_bottom.scatter_(-1, topk, True)

  return mask_bottom[..., :seq_length]





def apply_attention_mask(
  scores: torch.Tensor,
  attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
  """
  Support both:
  - bool mask: True means masked-out
  - additive mask: valid positions ~ 0, invalid positions ~ very negative
  """
  if attention_mask is None:
    return scores

  min_val = _dtype_min(scores.dtype)

  if attention_mask.dtype == torch.bool:
    return scores.masked_fill(attention_mask, min_val)

  scores = scores + attention_mask.to(scores.dtype)
  return torch.clamp(scores, min=min_val)



def valid_mask_from_attention_mask(
  attention_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
  """
  Return a bool mask where True means valid / selectable.
  Broadcast shape is preserved.
  """
  if attention_mask is None:
    return None

  if attention_mask.dtype == torch.bool:
    return ~attention_mask

  if torch.is_floating_point(attention_mask):
    threshold = _dtype_min(attention_mask.dtype) / 2
    return attention_mask > threshold

  return attention_mask == 0



def should_use_quest(self, q_len: int) -> bool:
  if q_len != 1:
    return False

  if getattr(self, "layer_id", 1) < getattr(self, "quest_start_layer", 3):
    return False
  return True

# =========================
# =========================


def glm_core_attention_forward(
  self,
  query_layer: torch.Tensor,  # (bs, n_heads, q, head_dim)
  key_layer: torch.Tensor,   # (bs, n_heads, k, head_dim)
  value_layer: torch.Tensor,  # (bs, n_heads, k, head_dim)
  attention_mask: Optional[torch.Tensor],
):
  """
  Patch point for GLM CoreAttention.

  We assume SelfAttention has already done:
  - fused QKV
  - reshape to [bs, heads, seq, head_dim]
  - RoPE
  - KV cache concat
  - MQA/GQA expansion if needed

  So here we only replace the final dense attention computation.
  """
  bsz, n_heads, q_len, head_dim = query_layer.shape
  kv_seq_len = key_layer.shape[-2]

  if not should_use_quest(self, q_len):
    return self.flash_forward(query_layer, key_layer, value_layer, attention_mask)

  attn_scores = torch.matmul(query_layer, key_layer.transpose(2, 3)) / self.norm_factor

  coeff = getattr(self, "coeff", None)
  if coeff is not None:
    attn_scores = attn_scores * coeff

  sign = torch.where(
    query_layer >= 0,
    torch.ones_like(query_layer),
    -torch.ones_like(query_layer),
  )
  positive_query = query_layer * sign
  max_key = key_layer * sign

  seq_length = max_key.shape[-2]
  padding_length = _pad_to_chunk_length(seq_length, self.chunk_size)

  if padding_length > 0:
    pad = torch.full(
      (max_key.shape[0], max_key.shape[1], padding_length, max_key.shape[3]),
      fill_value=_dtype_min(max_key.dtype),
      device=max_key.device,
      dtype=max_key.dtype,
    )
    max_key = torch.cat([max_key, pad], dim=-2)

  chunk_max_key = max_key.reshape(
    max_key.shape[0],
    max_key.shape[1],
    max_key.shape[2] // self.chunk_size,
    self.chunk_size,
    max_key.shape[3],
  ).amax(dim=-2)

  chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, self.chunk_size, 1)
  chunk_max_key = chunk_max_key.reshape(
    chunk_max_key.shape[0],
    chunk_max_key.shape[1],
    -1,
    chunk_max_key.shape[-1],
  )[:, :, :seq_length, :]

  quantized_weight = torch.matmul(
    positive_query.float(),
    chunk_max_key.transpose(2, 3).float(),
  )

  attn_scores = apply_attention_mask(attn_scores, attention_mask)
  quantized_weight = apply_attention_mask(quantized_weight, attention_mask)

  token_budget = int(getattr(self, "token_budget", 0))
  if token_budget <= 0:
    token_budget = kv_seq_len
  token_budget = min(token_budget, kv_seq_len)

  mask_bottom = local_heavy_hitter_mask(
    quantized_weight,
    token_budget,
    self.chunk_size,
  )

  valid_mask = valid_mask_from_attention_mask(attention_mask)
  if valid_mask is not None:
    mask_bottom = mask_bottom & valid_mask

  if kv_seq_len > 0:
    mask_bottom[..., -1] = True

  min_val = _dtype_min(attn_scores.dtype)
  attn_scores = attn_scores.masked_fill(~mask_bottom, min_val)

  attn_probs = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32)
  attn_probs = attn_probs.to(value_layer.dtype)
  attn_probs = self.attention_dropout(attn_probs)

  context_layer = torch.matmul(attn_probs, value_layer) # (bs, n_heads, q, head_dim)

  context_layer = context_layer.transpose(1, 2).contiguous()
  new_context_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
  context_layer = context_layer.reshape(*new_context_shape)

  return context_layer

# =========================
# =========================

#



def enable_quest_glm_attention_eval(model, args, model_dir=None):
  """
  Patch GLM model in-place.
  model_dir kept as parameter placeholder for backward compatibility, not used here.
  """
  if not hasattr(model, "transformer") or not hasattr(model.transformer, "encoder"):
    raise AttributeError("Expected model.transformer.encoder.layers in GLM model.")

  layers = model.transformer.encoder.layers
  if len(layers) == 0:
    raise RuntimeError("No encoder layers found in model.transformer.encoder.layers")


  sample_core = layers[0].self_attention.core_attention
  core_cls = sample_core.__class__

  num_layers = len(layers)
  patched = 0

  for i, layer in enumerate(layers, start=1):
    if not hasattr(layer, "self_attention"):
      raise AttributeError(f"Layer {i} has no self_attention")

    if not hasattr(layer.self_attention, "core_attention"):
      raise AttributeError(f"Layer {i} has no self_attention.core_attention")

    core = layer.self_attention.core_attention


    if not isinstance(core, core_cls):
      raise TypeError(
        f"Layer {i} core_attention is {type(core)}, expected {core_cls}"
      )

    if not hasattr(core, "flash_forward"):
      core.flash_forward = core.forward


    core.forward = types.MethodType(glm_core_attention_forward, core)
    core.layer_id = i
    core.num_layers = num_layers
    core.token_budget = int(getattr(args, "token_budget", 0))
    core.chunk_size = int(getattr(args, "chunk_size", 16))
    core.quest_start_layer = int(getattr(args, "quest_start_layer", 3))

    patched += 1

  if patched == 0:
    raise RuntimeError("No GLM CoreAttention modules were patched.")

  return model


def disable_quest_glm_attention_eval(model):
  if not hasattr(model, "transformer") or not hasattr(model.transformer, "encoder"):
    return model

  for layer in model.transformer.encoder.layers:
    if not hasattr(layer, "self_attention"):
      continue
    if not hasattr(layer.self_attention, "core_attention"):
      continue

    core = layer.self_attention.core_attention
    if hasattr(core, "flash_forward"):
      core.forward = core.flash_forward

  return model


def validate_glm_patch(model):
  """
  Lightweight self-check after patching.
  """
  rows = []
  if not hasattr(model, "transformer") or not hasattr(model.transformer, "encoder"):
    return rows

  for i, layer in enumerate(model.transformer.encoder.layers, start=1):
    core = layer.self_attention.core_attention
    forward_obj = core.forward
    forward_name = getattr(getattr(forward_obj, "__func__", None), "__name__", type(forward_obj).__name__)

    rows.append(
      {
        "layer": i,
        "patched": hasattr(core, "flash_forward"),
        "forward_name": forward_name,
        "layer_id": getattr(core, "layer_id", None),
        "quest_start_layer": getattr(core, "quest_start_layer", None),
        "token_budget": getattr(core, "token_budget", None),
        "chunk_size": getattr(core, "chunk_size", None),
      }
    )
  return rows


enable_quest_attention_eval = enable_quest_glm_attention_eval
