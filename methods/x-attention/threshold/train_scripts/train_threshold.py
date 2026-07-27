import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

XATTN_ROOT = Path(".")
if str(XATTN_ROOT) not in sys.path:
  sys.path.insert(0, str(XATTN_ROOT))

try:
  from transformers import StaticCache
except Exception:
  StaticCache = tuple

try:
  from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
  )
  from transformers.models.llama.modeling_llama import repeat_kv as llama_repeat_kv
except Exception:
  llama_apply_rotary_pos_emb = None
  llama_repeat_kv = None


logger = logging.getLogger(__name__)
xattn_estimate = None
create_causal_mask = None


def ensure_xattn_imports():
  global xattn_estimate, create_causal_mask
  if xattn_estimate is not None and create_causal_mask is not None:
    return
  try:
    from xattn.src.Xattention import xattn_estimate as imported_xattn_estimate
    from xattn.src.utils import create_causal_mask as imported_create_causal_mask
  except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
      "Failed to import xattn. Run this script from methods/x-attention or add "
      "methods/x-attention to PYTHONPATH, and make sure Block-Sparse-Attention is installed."
    ) from exc
  xattn_estimate = imported_xattn_estimate
  create_causal_mask = imported_create_causal_mask


def str2bool(value):
  if isinstance(value, bool):
    return value
  value = str(value).strip().lower()
  if value in {"1", "true", "t", "yes", "y"}:
    return True
  if value in {"0", "false", "f", "no", "n"}:
    return False
  raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {value}")


def resolve_torch_dtype(dtype_name: str):
  dtype_name = dtype_name.lower()
  if dtype_name == "auto":
    return "auto"
  mapping = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
  }
  if dtype_name not in mapping:
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")
  return mapping[dtype_name]


def p_to_suffix(p: float) -> str:
  value = int(round(p * 100))
  if abs(p * 100 - value) > 1e-6:
    raise ValueError(f"Cannot build threshold suffix from p={p}")
  return str(value)


def infer_requested_family(name_or_path: str) -> str:
  lower_name = name_or_path.lower()
  if "glm" in lower_name:
    return "glm"
  if "llama" in lower_name:
    return "llama"
  if "qwen" in lower_name or "deepseek" in lower_name:
    return "qwen2"
  return "auto"


def infer_output_var(model_type: str, name_or_path: str, p: float) -> str:
  suffix = p_to_suffix(p)
  lower_name = name_or_path.lower()
  if model_type == "chatglm" or "glm" in lower_name:
    return "max"
  if "deepseek" in lower_name:
    return "ds_qwen"
  if model_type == "llama" or "llama" in lower_name:
    return f"llama_fuse_{suffix}"
  if model_type == "qwen2" or "qwen" in lower_name:
    return f"qwen_fuse_{suffix}"
  return f"threshold_p{suffix}"


def save_threshold(threshold_tensor: torch.Tensor, output_path: Path, output_var: str, output_format: str):
  output_path.parent.mkdir(parents=True, exist_ok=True)
  values = threshold_tensor.detach().cpu().tolist()

  if output_format == "auto":
    output_format = "py" if output_path.suffix == ".py" else "json"

  if output_format == "py":
    with output_path.open("w", encoding="utf-8") as f:
      f.write(f"{output_var} = {repr(values)}\n")
      if output_var != "max":
        f.write(f"max = {output_var}\n")
    return

  if output_format == "json":
    with output_path.open("w", encoding="utf-8") as f:
      json.dump(values, f, ensure_ascii=False)
    return

  raise ValueError(f"Unsupported output_format={output_format!r}")


def get_num_heads(module) -> int:
  if hasattr(module, "num_heads"):
    return module.num_heads
  return module.config.num_attention_heads


def get_num_kv_heads(module) -> int:
  if hasattr(module, "num_key_value_heads"):
    return module.num_key_value_heads
  return module.config.num_key_value_heads


def get_num_kv_groups(module) -> int:
  if hasattr(module, "num_key_value_groups"):
    return module.num_key_value_groups
  return get_num_heads(module) // get_num_kv_heads(module)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_local(
  q: torch.Tensor,
  k: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  position_ids: Optional[torch.LongTensor] = None,
  unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
  cos = cos.unsqueeze(unsqueeze_dim)
  sin = sin.unsqueeze(unsqueeze_dim)
  q_embed = (q * cos) + (rotate_half(q) * sin)
  k_embed = (k * cos) + (rotate_half(k) * sin)
  return q_embed, k_embed


def repeat_kv_local(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
  batch, num_key_value_heads, slen, head_dim = hidden_states.shape
  if n_rep == 1:
    return hidden_states
  hidden_states = hidden_states[:, :, None, :, :].expand(
    batch,
    num_key_value_heads,
    n_rep,
    slen,
    head_dim,
  )
  return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@torch.jit.script
def apply_rotary_pos_emb_chatglm(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
  b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
  rot_dim = rope_cache.shape[-2] * 2
  x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
  rope_cache = rope_cache[:, :sq]
  xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
  rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
  x_out2 = torch.stack(
    [
      xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
      xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
    ],
    -1,
  )
  x_out2 = x_out2.flatten(3)
  return torch.cat((x_out2, x_pass), dim=-1)


def chunk_prefill_to_attn_sum(
  query_states: torch.Tensor,
  key_states: torch.Tensor,
  block_size: int,
) -> torch.Tensor:
  device = query_states.device
  batch_size, num_kv_head, k_len, head_dim = key_states.shape
  _, num_q_head, q_len, _ = query_states.shape

  q_num_to_pad = ((q_len + block_size - 1) // block_size) * block_size - q_len
  k_num_to_pad = ((k_len + block_size - 1) // block_size) * block_size - k_len
  q_block_num = (q_len + block_size - 1) // block_size
  k_block_num = (k_len + block_size - 1) // block_size

  key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad), value=0).to(device)
  query_states = F.pad(query_states, (0, 0, 0, q_num_to_pad), value=0).to(device)

  if q_block_num != k_block_num:
    raise AssertionError("Threshold profiling only supports prefill.")
  if num_kv_head != num_q_head:
    raise AssertionError(f"Expected repeated K heads to match Q heads, got K={num_kv_head}, Q={num_q_head}.")

  attn_sum_list = []
  scale = 1.0 / math.sqrt(head_dim)

  for i in range(q_block_num):
    query_states_slice = query_states[:, :, i * block_size : (i + 1) * block_size, :].to(device)
    attn_weights_slice = torch.matmul(
      query_states_slice,
      key_states.transpose(2, 3) * scale,
    ).to(device)

    causal_mask = create_causal_mask(batch_size, num_kv_head, block_size, k_block_num, i)
    causal_mask = F.pad(causal_mask[:, :, :, 0:k_len], (0, k_num_to_pad), value=float("-inf")).to(device)

    if i == q_block_num - 1:
      causal_mask = F.pad(
        causal_mask[:, :, 0 : block_size - q_num_to_pad, :],
        (0, 0, 0, q_num_to_pad),
        value=float("-inf"),
      )

    attn_weights_slice = attn_weights_slice + causal_mask
    attn_weights_slice = F.softmax(attn_weights_slice, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights_slice = F.dropout(attn_weights_slice, p=0, training=False)

    if i == q_block_num - 1:
      attn_weights_slice = F.pad(
        attn_weights_slice[:, :, 0 : block_size - q_num_to_pad, :],
        (0, 0, 0, q_num_to_pad),
      )

    attn_sum = (
      attn_weights_slice.view(batch_size, num_kv_head, block_size, -1, block_size)
      .sum(dim=-1)
      .sum(dim=-2)
      .to(device)
    )
    attn_sum_list.append(attn_sum.unsqueeze(dim=-2))
    del attn_weights_slice

  return torch.cat(attn_sum_list, dim=-2)


class ProfileConfig:
  def __init__(self, p: float, stride: int, causal: bool, block_size: int):
    self.p = p
    self.stride = stride
    self.history_threshold = []
    self.causal = causal
    self.block_size = block_size


def module_layer_idx(module) -> int:
  if hasattr(module, "profile_layer_idx"):
    return int(module.profile_layer_idx)
  return int(getattr(module, "layer_idx", 0))


def xattn_prefill_profile(
  self,
  query_states: torch.Tensor,
  key_states: torch.Tensor,
  value_states: torch.Tensor,
  block_size: int,
  stride: int,
  chunk_size: int = 16384,
  causal: bool = True,
):
  attn_sum = chunk_prefill_to_attn_sum(
    query_states=query_states,
    key_states=key_states,
    block_size=block_size,
  )

  xattn_sum, _ = xattn_estimate(
    query_states,
    key_states,
    block_size=block_size,
    stride=stride,
    norm=1,
    threshold=1,
    select_mode="inverse",
    use_triton=True,
    causal=causal,
    chunk_size=chunk_size,
  )
  xattn_sum = xattn_sum[:, :, : attn_sum.shape[-1], : attn_sum.shape[-1]] * stride

  vals, idx = torch.sort(attn_sum, dim=-1, descending=True)
  cumsum = vals.cumsum(-1)
  need_len = (cumsum < self.profile_config.p * cumsum[..., -1:]).sum(-1) + 1
  rank_idx = torch.arange(vals.size(-1), device=vals.device)
  sel_mask_srt = rank_idx.view(1, 1, 1, -1) < need_len.unsqueeze(-1)

  x_sorted = xattn_sum.gather(-1, idx)
  miniblock = x_sorted.masked_fill(~sel_mask_srt, float("inf")).min(-1).values
  mask_ge_min = xattn_sum >= miniblock.unsqueeze(-1)
  threshold_head = (xattn_sum * mask_ge_min).sum(-1).sum(-1) / (
    xattn_sum.sum(-1).sum(-1) + 1e-8
  )
  threshold_head = threshold_head.detach().to("cpu", dtype=torch.float32)

  if module_layer_idx(self) == 0:
    self.profile_config.history_threshold.append(threshold_head)
  else:
    self.profile_config.history_threshold[-1] = torch.cat(
      [self.profile_config.history_threshold[-1], threshold_head],
      dim=0,
    )

  return flash_attn_func(
    q=query_states.transpose(1, 2),
    k=key_states.transpose(1, 2),
    v=value_states.transpose(1, 2),
    causal=causal,
  ).transpose(1, 2).contiguous()


def forward_profile_llama(
  self,
  hidden_states: torch.Tensor,
  attention_mask: Optional[torch.Tensor] = None,
  position_ids: Optional[torch.LongTensor] = None,
  past_key_value: Optional[object] = None,
  output_attentions: bool = False,
  use_cache: bool = False,
  cache_position: Optional[torch.LongTensor] = None,
  position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
  **kwargs,
):
  if llama_repeat_kv is None or llama_apply_rotary_pos_emb is None:
    raise ImportError("Llama attention helpers are unavailable in this transformers installation.")

  bsz, q_len, _ = hidden_states.size()

  query_states = self.q_proj(hidden_states)
  key_states = self.k_proj(hidden_states)
  value_states = self.v_proj(hidden_states)

  query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
  key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
  value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

  if position_embeddings is None:
    cos, sin = self.rotary_emb(value_states, position_ids)
  else:
    cos, sin = position_embeddings

  query_states, key_states = llama_apply_rotary_pos_emb(query_states, key_states, cos, sin)

  if past_key_value is not None:
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

  _, _, k_len, _ = key_states.shape
  _, _, q_len, _ = query_states.shape
  decoding = (q_len != k_len and q_len == 1)

  if not decoding:
    key_states = llama_repeat_kv(key_states, self.num_key_value_groups).to(query_states.device)
    value_states = llama_repeat_kv(value_states, self.num_key_value_groups).to(query_states.device)

  stride = self.profile_config.stride
  causal = self.profile_config.causal
  block_size = self.profile_config.block_size

  if key_states.shape != query_states.shape:
    raise AssertionError(f"Expected K shape to match Q shape, got K={key_states.shape}, Q={query_states.shape}.")

  attn_output = xattn_prefill_profile(
    self,
    query_states=query_states,
    key_states=key_states,
    value_states=value_states,
    block_size=block_size,
    stride=stride,
    causal=causal,
  )

  if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
    raise ValueError(
      f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
      f" {attn_output.size()}"
    )

  attn_output = attn_output.transpose(1, 2).contiguous()
  attn_output = attn_output.reshape(bsz, q_len, -1)
  attn_output = self.o_proj(attn_output)
  del query_states

  return attn_output, None, past_key_value


def forward_profile_qwen2(
  self,
  hidden_states: torch.Tensor,
  attention_mask: Optional[torch.Tensor] = None,
  position_ids: Optional[torch.LongTensor] = None,
  past_key_value: Optional[object] = None,
  output_attentions: bool = False,
  use_cache: bool = False,
  cache_position: Optional[torch.LongTensor] = None,
  position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
  **kwargs,
):
  bsz, q_len, _ = hidden_states.size()
  num_heads = get_num_heads(self)
  num_kv_heads = get_num_kv_heads(self)
  num_kv_groups = get_num_kv_groups(self)

  query_states = self.q_proj(hidden_states)
  key_states = self.k_proj(hidden_states)
  value_states = self.v_proj(hidden_states)

  query_states = query_states.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
  key_states = key_states.view(bsz, q_len, num_kv_heads, self.head_dim).transpose(1, 2)
  value_states = value_states.view(bsz, q_len, num_kv_heads, self.head_dim).transpose(1, 2)

  if position_embeddings is None:
    if position_ids is None:
      raise ValueError("Profiling forward requires `position_ids` or `position_embeddings`.")
    cos, sin = self.rotary_emb(value_states, position_ids)
  else:
    cos, sin = position_embeddings

  query_states, key_states = apply_rotary_pos_emb_local(query_states, key_states, cos, sin)

  if past_key_value is not None:
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    try:
      key_states, value_states = past_key_value.update(
        key_states,
        value_states,
        self.layer_idx,
        cache_kwargs,
      )
    except TypeError:
      key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

  if isinstance(past_key_value, StaticCache) and cache_position is not None:
    key_states = key_states[:, :, : min(cache_position[-1] + 1, key_states.shape[2]), :]
    value_states = value_states[:, :, : min(cache_position[-1] + 1, value_states.shape[2]), :]

  _, _, k_len, _ = key_states.shape
  _, _, q_len, _ = query_states.shape
  decoding = (q_len != k_len and q_len == 1)

  if not decoding:
    key_states = repeat_kv_local(key_states, num_kv_groups).to(query_states.device)
    value_states = repeat_kv_local(value_states, num_kv_groups).to(query_states.device)

  stride = self.profile_config.stride
  causal = self.profile_config.causal
  block_size = self.profile_config.block_size

  if key_states.shape != query_states.shape:
    raise AssertionError(f"Expected K shape to match Q shape, got K={key_states.shape}, Q={query_states.shape}.")

  attn_output = xattn_prefill_profile(
    self,
    query_states=query_states,
    key_states=key_states,
    value_states=value_states,
    block_size=block_size,
    stride=stride,
    causal=causal,
  )

  if attn_output.size() != (bsz, num_heads, q_len, self.head_dim):
    raise ValueError(
      f"`attn_output` should be of size {(bsz, num_heads, q_len, self.head_dim)}, but is {attn_output.size()}"
    )

  attn_output = attn_output.transpose(1, 2).contiguous()
  attn_output = attn_output.reshape(bsz, q_len, -1)
  attn_output = self.o_proj(attn_output)
  del query_states

  return attn_output, None, past_key_value


def forward_profile_chatglm(
  self,
  hidden_states: torch.Tensor,
  attention_mask: Optional[torch.Tensor],
  rotary_pos_emb: Optional[torch.Tensor],
  kv_cache=None,
  use_cache: bool = True,
):
  bsz, q_len, _ = hidden_states.size()
  num_heads = self.num_attention_heads_per_partition
  head_dim = self.hidden_size_per_attention_head

  mixed_x_layer = self.query_key_value(hidden_states)

  if self.multi_query_attention:
    num_kv_heads = self.num_multi_query_groups_per_partition
    query_width = num_heads * head_dim
    kv_width = num_kv_heads * head_dim
    query_layer, key_layer, value_layer = mixed_x_layer.split(
      [query_width, kv_width, kv_width],
      dim=-1,
    )
    query_layer = query_layer.view(bsz, q_len, num_heads, head_dim)
    key_layer = key_layer.view(bsz, q_len, num_kv_heads, head_dim)
    value_layer = value_layer.view(bsz, q_len, num_kv_heads, head_dim)
  else:
    num_kv_heads = num_heads
    mixed_x_layer = mixed_x_layer.view(bsz, q_len, num_heads, 3 * head_dim)
    query_layer, key_layer, value_layer = torch.chunk(mixed_x_layer, 3, dim=-1)

  query_layer = query_layer.transpose(1, 2)
  key_layer = key_layer.transpose(1, 2)
  value_layer = value_layer.transpose(1, 2)

  if rotary_pos_emb is not None:
    query_layer = apply_rotary_pos_emb_chatglm(query_layer, rotary_pos_emb)
    key_layer = apply_rotary_pos_emb_chatglm(key_layer, rotary_pos_emb)

  if kv_cache is not None:
    cache_k, cache_v = kv_cache
    key_layer = torch.cat((cache_k, key_layer), dim=2)
    value_layer = torch.cat((cache_v, value_layer), dim=2)

  if use_cache:
    if kv_cache is None:
      new_kv_cache = torch.cat(
        (key_layer.unsqueeze(0).unsqueeze(0), value_layer.unsqueeze(0).unsqueeze(0)),
        dim=1,
      )
    else:
      new_kv_cache = (key_layer, value_layer)
  else:
    new_kv_cache = None

  _, _, k_len, _ = key_layer.shape
  decoding = (q_len != k_len and q_len == 1)

  if self.multi_query_attention:
    kv_repeat = num_heads // num_kv_heads
    key_layer = repeat_kv_local(key_layer, kv_repeat)
    value_layer = repeat_kv_local(value_layer, kv_repeat)

  stride = self.profile_config.stride
  causal = self.profile_config.causal
  block_size = self.profile_config.block_size

  if not decoding:
    if key_layer.shape != query_layer.shape:
      raise AssertionError(f"Expected K shape to match Q shape, got K={key_layer.shape}, Q={query_layer.shape}.")
    attn_output = xattn_prefill_profile(
      self,
      query_states=query_layer,
      key_states=key_layer,
      value_states=value_layer,
      block_size=block_size,
      stride=stride,
      causal=causal,
    )
  else:
    attn_output = flash_attn_func(
      q=query_layer.transpose(1, 2),
      k=key_layer.transpose(1, 2),
      v=value_layer.transpose(1, 2),
      causal=causal,
    ).transpose(1, 2).contiguous()

  if attn_output.size() != (bsz, num_heads, q_len, head_dim):
    raise ValueError(
      f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is {attn_output.size()}"
    )

  attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
  output = self.dense(attn_output)
  return output, new_kv_cache


def patch_llama_qwen_for_profiling(model, profile_config: ProfileConfig) -> str:
  model_type = getattr(model.config, "model_type", None)
  if model_type not in {"llama", "qwen2"}:
    raise NotImplementedError(f"Expected model_type llama/qwen2, got {model_type!r}.")

  forward_impl = forward_profile_llama if model_type == "llama" else forward_profile_qwen2
  layers = model.model.layers
  for idx, layer in enumerate(layers):
    layer.self_attn.forward = forward_impl.__get__(
      layer.self_attn,
      type(layer.self_attn),
    )
    layer.self_attn.profile_config = profile_config
    if not hasattr(layer.self_attn, "layer_idx"):
      layer.self_attn.layer_idx = idx
    layer.self_attn.profile_layer_idx = idx
  return model_type


def patch_chatglm_for_profiling(model, profile_config: ProfileConfig) -> str:
  model_type = getattr(model.config, "model_type", None)
  if model_type != "chatglm":
    raise NotImplementedError(f"Expected model_type chatglm, got {model_type!r}.")

  layers = model.transformer.encoder.layers
  for idx, layer in enumerate(layers):
    layer.self_attention.forward = forward_profile_chatglm.__get__(
      layer.self_attention,
      type(layer.self_attention),
    )
    layer.self_attention.profile_config = profile_config
    layer.self_attention.profile_layer_idx = idx
    layer.self_attention.layer_idx = idx
  return model_type


def patch_model_for_profiling(model, profile_config: ProfileConfig, requested_family: str) -> str:
  model_type = getattr(model.config, "model_type", None)

  if requested_family == "glm" or model_type == "chatglm":
    return patch_chatglm_for_profiling(model, profile_config)
  if requested_family in {"llama", "qwen2"} or model_type in {"llama", "qwen2"}:
    return patch_llama_qwen_for_profiling(model, profile_config)

  raise NotImplementedError(
    f"Unsupported model_type={model_type!r}. Supported families: llama, qwen2, chatglm."
  )


def compute_final_threshold_from_history(profile_config: ProfileConfig) -> torch.Tensor:
  history_threshold = profile_config.history_threshold
  if len(history_threshold) == 0:
    raise RuntimeError("No profiling history found. Check input texts and forward patching.")
  return torch.cat([threshold.unsqueeze(0) for threshold in history_threshold]).max(0)[0]


def run_profile_on_texts(model, tokenizer, texts):
  for text in tqdm(texts):
    if not isinstance(text, str):
      raise TypeError("Each item in the profiling text file must be a string.")
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
      _ = model(**inputs, use_cache=False)


def build_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--name_or_path", type=str, required=True)
  parser.add_argument("--model_type", type=str, default="auto", choices=["auto", "llama", "qwen2", "glm"])
  parser.add_argument("--p", type=float, required=True)
  parser.add_argument("--text_path", type=str, default="xattn/threshold/profile_threshold/text.json")
  parser.add_argument("--output_path", type=str, default="profiled_threshold.py")
  parser.add_argument("--output_format", type=str, default="auto", choices=["auto", "py", "json"])
  parser.add_argument("--output_var", type=str, default="")
  parser.add_argument("--stride", type=int, default=None)
  parser.add_argument("--block_size", type=int, default=128)
  parser.add_argument("--causal", type=str2bool, default=True)
  parser.add_argument("--device_map", type=str, default="")
  parser.add_argument("--torch_dtype", type=str, default="bfloat16")
  parser.add_argument("--trust_remote_code", type=str2bool, default=True)
  parser.add_argument("--attn_implementation", type=str, default="")
  parser.add_argument("--chunk_texts", type=int, default=0)
  parser.add_argument("--batch_output_dir", type=str, default="")
  parser.add_argument("--empty_cache_per_chunk", type=str2bool, default=True)
  return parser


def main():
  logging.basicConfig(level=logging.WARNING)
  args = build_parser().parse_args()

  if not 0.0 < args.p <= 1.0:
    raise ValueError(f"p must be in (0, 1], got {args.p}")
  ensure_xattn_imports()

  requested_family = args.model_type
  if requested_family == "auto":
    requested_family = infer_requested_family(args.name_or_path)

  if args.stride is None:
    args.stride = 16 if "deepseek" in args.name_or_path.lower() else 8
  if not args.device_map:
    args.device_map = "balanced" if requested_family == "llama" else "auto"

  torch_dtype = resolve_torch_dtype(args.torch_dtype)
  load_kwargs = {
    "device_map": args.device_map,
    "torch_dtype": torch_dtype,
    "trust_remote_code": args.trust_remote_code,
  }
  if not args.attn_implementation and requested_family == "qwen2":
    args.attn_implementation = "flash_attention_2"
  if args.attn_implementation:
    load_kwargs["attn_implementation"] = args.attn_implementation

  model = AutoModelForCausalLM.from_pretrained(args.name_or_path, **load_kwargs)
  model.eval()

  profile_config = ProfileConfig(
    p=args.p,
    stride=args.stride,
    causal=args.causal,
    block_size=args.block_size,
  )
  model_type = patch_model_for_profiling(model, profile_config, requested_family)
  logger.warning("Profiling threshold with model_type=%s, p=%s, stride=%s", model_type, args.p, args.stride)

  tokenizer = AutoTokenizer.from_pretrained(
    args.name_or_path,
    trust_remote_code=args.trust_remote_code,
  )

  text_path = Path(args.text_path)
  with text_path.open("r", encoding="utf-8") as f:
    texts = json.load(f)

  if not isinstance(texts, list) or len(texts) == 0:
    raise ValueError(f"`{text_path}` must be a non-empty JSON list of strings.")

  output_path = Path(args.output_path)
  output_var = args.output_var or infer_output_var(model_type, args.name_or_path, args.p)

  if args.chunk_texts <= 0:
    run_profile_on_texts(model, tokenizer, texts)
    final_threshold = compute_final_threshold_from_history(profile_config).detach().cpu()
    save_threshold(final_threshold, output_path, output_var, args.output_format)
    print(final_threshold.tolist())
    print(f"Saved threshold file to: {output_path}")
    return

  batch_output_dir = Path(args.batch_output_dir) if args.batch_output_dir else output_path.with_suffix("")
  if not args.batch_output_dir:
    batch_output_dir = Path(str(batch_output_dir) + "_parts")
  batch_output_dir.mkdir(parents=True, exist_ok=True)

  merged_threshold = None
  for chunk_idx, start in enumerate(range(0, len(texts), args.chunk_texts)):
    end = min(start + args.chunk_texts, len(texts))
    profile_config.history_threshold = []
    run_profile_on_texts(model, tokenizer, texts[start:end])

    chunk_threshold = compute_final_threshold_from_history(profile_config).detach().cpu()
    chunk_file = batch_output_dir / f"threshold_chunk_{chunk_idx:03d}_{start}_{end}.json"
    save_threshold(chunk_threshold, chunk_file, output_var, "json")

    if merged_threshold is None:
      merged_threshold = chunk_threshold
    else:
      merged_threshold = torch.maximum(merged_threshold, chunk_threshold)

    if torch.cuda.is_available() and args.empty_cache_per_chunk:
      torch.cuda.empty_cache()

    logger.warning("Finished chunk %d: [%d, %d), saved %s", chunk_idx, start, end, chunk_file)

  if merged_threshold is None:
    raise RuntimeError("No chunk threshold was generated.")

  save_threshold(merged_threshold, output_path, output_var, args.output_format)
  print(merged_threshold.tolist())
  print(f"Saved merged threshold file to: {output_path}")
  print(f"Saved chunk threshold files to: {batch_output_dir}")


if __name__ == "__main__":
  main()
