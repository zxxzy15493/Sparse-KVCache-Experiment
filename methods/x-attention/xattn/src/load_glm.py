import math
import types
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from xattn.src.Xattention import Xattention_prefill
from threshold.threshold.glm_threshold import max as glm_fuse_90


class FastPrefillConfig(dict):
  def __init__(
    self,
    threshold: float = None,
    print_detail: bool = False,
    stride: int = 16,
    metric: str = "xattn",
    p: float = 0.9,
  ):
    super().__init__()
    self.print_detail = print_detail
    self.metric = metric
    self.stride = stride
    self.p = p

    if threshold is not None:
      self.threshold = float(threshold)
    elif p == 0.9:
      self.threshold = torch.tensor(glm_fuse_90, dtype=torch.float32)
    else:
      raise ValueError("GLM only has a p=0.9 threshold table; pass threshold=... for a scalar threshold.")


def _layer_threshold(
  config: FastPrefillConfig,
  layer_idx: int,
  num_heads: int,
  device: torch.device,
  dtype: torch.dtype,
):
  threshold = config.threshold
  if not isinstance(threshold, torch.Tensor):
    return threshold

  if threshold.ndim == 0:
    return float(threshold.item())
  if layer_idx >= threshold.shape[0]:
    raise ValueError(
      f"Threshold table has {threshold.shape[0]} layers, but layer_idx={layer_idx} was requested."
    )

  values = threshold[layer_idx].detach().flatten().float()
  if values.numel() == 0:
    raise ValueError("Threshold table row is empty; pass a scalar threshold for this model.")
  if values.numel() >= num_heads:
    values = values[:num_heads]
  else:
    values = torch.cat([values, values.mean().expand(num_heads - values.numel())])
  return values.to(device=device, dtype=dtype)


@torch.no_grad()
def glm_core_attention_forward(
  self,
  query_layer: torch.Tensor,
  key_layer: torch.Tensor,
  value_layer: torch.Tensor,
  attention_mask: Optional[torch.Tensor],
):
  if self.fastprefillconfig.print_detail:
    print(f"q length: {query_layer.shape[-2]} k length: {key_layer.shape[-2]}")

  bsz, num_heads, q_len, head_dim = query_layer.shape
  k_len = key_layer.shape[2]

  if q_len == k_len:
    if self.fastprefillconfig.metric != "xattn":
      raise ValueError(f"Unknown metric: {self.fastprefillconfig.metric}")

    threshold = _layer_threshold(
      self.fastprefillconfig,
      self.layer_idx,
      num_heads,
      key_layer.device,
      key_layer.dtype,
    )
    attn_output = Xattention_prefill(
      query_layer,
      key_layer,
      value_layer,
      self.fastprefillconfig.stride,
      model_name="GLM",
      layer_id=self.layer_idx,
      norm=1,
      threshold=threshold,
      use_triton=True,
      keep_sink=True,
      keep_recent=True,
    )
  else:
    attn_weights = torch.matmul(query_layer, key_layer.transpose(2, 3)) / math.sqrt(head_dim)

    if attention_mask is not None:
      attn_weights = attn_weights.masked_fill(attention_mask, float("-inf"))

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_layer.dtype)
    attn_weights = nn.functional.dropout(
      attn_weights,
      p=self.attention_dropout.p,
      training=self.training,
    )
    attn_output = torch.matmul(attn_weights, value_layer)

  expected_shape = (bsz, num_heads, q_len, head_dim)
  if attn_output.size() != expected_shape:
    raise ValueError(
      f"`attn_output` should be of size {expected_shape}, but is {attn_output.size()}"
    )

  attn_output = attn_output.transpose(1, 2).contiguous()
  attn_output = attn_output.view(bsz, q_len, self.hidden_size_per_partition)
  return attn_output


def patch_glm_attention(model, fastprefillconfig: FastPrefillConfig):
  if not hasattr(model, "transformer") or not hasattr(model.transformer, "encoder"):
    raise AttributeError("Expected model.transformer.encoder.layers in GLM model.")

  layers = model.transformer.encoder.layers
  if len(layers) == 0:
    raise RuntimeError("No encoder layers found in model.transformer.encoder.layers.")

  patched = 0
  for i, layer in enumerate(layers):
    if not hasattr(layer, "self_attention") or not hasattr(layer.self_attention, "core_attention"):
      raise AttributeError(f"Layer {i + 1} has no self_attention.core_attention.")

    core = layer.self_attention.core_attention
    core.fastprefillconfig = fastprefillconfig
    core.layer_idx = i
    core.forward = types.MethodType(glm_core_attention_forward, core)
    patched += 1

  if patched == 0:
    raise RuntimeError("No GLM CoreAttention modules were patched.")
  print(f"[info] patched {patched} GLM core_attention modules")


def load_model(
  fastprefillconfig: FastPrefillConfig = FastPrefillConfig(),
  name_or_path: str = "",
):
  model = AutoModelForCausalLM.from_pretrained(
    name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
  )
  model.eval()
  patch_glm_attention(model, fastprefillconfig)

  tokenizer = AutoTokenizer.from_pretrained(
    name_or_path,
    trust_remote_code=True,
    use_fast=False,
  )
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "left"
  return model, tokenizer
