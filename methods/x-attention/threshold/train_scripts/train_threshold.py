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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

XATTN_ROOT = Path(__file__).resolve().parents[2]
if str(XATTN_ROOT) not in sys.path:
  sys.path.insert(0, str(XATTN_ROOT))

try:
  from transformers import StaticCache
except Exception:
  StaticCache = tuple

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


def resolve_max_memory(value: str, reserve_memory_gib: float, device_map: str):
  """Limit weight placement while leaving room for long-context activations."""
  if not math.isfinite(reserve_memory_gib) or reserve_memory_gib < 0:
    raise ValueError("reserve_memory_gib must be finite and non-negative.")
  inferred_map = device_map in {"auto", "balanced", "balanced_low_0", "sequential"}
  if value:
    from accelerate.utils import convert_file_size_to_int

    if not inferred_map:
      raise ValueError("--max_memory requires an inferred device_map such as balanced or auto; "
                       "an explicit device placement cannot enforce weight budgets.")
    memory = json.loads(value)
    if not isinstance(memory, dict) or not memory:
      raise ValueError('--max_memory must be a non-empty JSON object, e.g. {"0":"40GiB","1":"40GiB"}.')
    result = {}
    for device, limit in memory.items():
      if device.isdecimal():
        device = int(device)
      elif device != "cpu":
        raise ValueError(f"Invalid max_memory device {device!r}; use visible GPU indices or cpu.")
      if isinstance(limit, bool) or not isinstance(limit, (int, str)):
        raise ValueError(f"Invalid max_memory limit {limit!r}; use bytes or a size such as 40GiB.")
      limit = convert_file_size_to_int(limit) if isinstance(limit, str) else limit
      if limit <= 0 or device in result:
        raise ValueError("max_memory limits must be positive and device indices must be unique.")
      result[device] = limit
    return result

  if not inferred_map or not torch.cuda.is_available():
    return None
  reserve_bytes = int(reserve_memory_gib * 1024**3)
  result = {}
  for device in range(torch.cuda.device_count()):
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes <= reserve_bytes:
      raise ValueError(
        f"GPU {device} has only {free_bytes / 1024**3:.2f} GiB free, insufficient for "
        f"the {reserve_memory_gib:g} GiB reserve. Select available CUDA_VISIBLE_DEVICES "
        "or adjust --reserve_memory_gib/--max_memory."
      )
    result[device] = free_bytes - reserve_bytes
  return result


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
  if "qwen3" in lower_name:
    return "qwen3"
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
  if model_type == "qwen3":
    return f"qwen3_fuse_{suffix}"
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


def load_threshold_json(input_path: Path) -> torch.Tensor:
  with input_path.open("r", encoding="utf-8") as f:
    values = json.load(f)
  try:
    threshold = torch.tensor(values, dtype=torch.float32)
  except (TypeError, ValueError) as exc:
    raise ValueError(f"Invalid threshold chunk file: {input_path}") from exc
  if threshold.ndim != 2 or not torch.isfinite(threshold).all():
    raise ValueError(
      f"Threshold chunk `{input_path}` must contain a finite 2D numeric array, "
      f"got shape {tuple(threshold.shape)}."
    )
  return threshold


def get_num_heads(module) -> int:
  if hasattr(module, "num_heads"):
    return module.num_heads
  return module.config.num_attention_heads


def get_num_kv_heads(module) -> int:
  if hasattr(module, "num_key_value_heads"):
    return module.num_key_value_heads
  return module.config.num_key_value_heads


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
  causal: bool = True,
) -> torch.Tensor:
  device = query_states.device
  batch_size, num_kv_head, k_len, head_dim = key_states.shape
  _, num_q_head, q_len, _ = query_states.shape

  q_num_to_pad = ((q_len + block_size - 1) // block_size) * block_size - q_len
  k_num_to_pad = ((k_len + block_size - 1) // block_size) * block_size - k_len
  q_block_num = (q_len + block_size - 1) // block_size
  k_block_num = (k_len + block_size - 1) // block_size

  if k_num_to_pad:
    key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad), value=0)
  if q_num_to_pad:
    query_states = F.pad(query_states, (0, 0, 0, q_num_to_pad), value=0)

  if q_len != k_len:
    raise AssertionError("Threshold profiling only supports prefill.")
  if num_kv_head != num_q_head:
    raise AssertionError(f"Expected repeated K heads to match Q heads, got K={num_kv_head}, Q={num_q_head}.")

  attn_sums = query_states.new_empty(batch_size, num_q_head, q_block_num, k_block_num)
  scaled_key = key_states.transpose(2, 3) * (1.0 / math.sqrt(head_dim))

  for i in range(q_block_num):
    query_states_slice = query_states[:, :, i * block_size : (i + 1) * block_size, :]
    attn_weights_slice = torch.matmul(query_states_slice, scaled_key)

    # Broadcast one mask across heads, directly on the owning layer's device.
    if causal:
      causal_mask = create_causal_mask(1, 1, block_size, k_block_num, i, device=device)
    else:
      causal_mask = torch.zeros(1, 1, block_size, k_block_num * block_size, device=device)
    if k_num_to_pad:
      causal_mask[..., k_len:] = float("-inf")

    attn_weights_slice = attn_weights_slice + causal_mask
    attn_weights_slice = F.softmax(attn_weights_slice, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if i == q_block_num - 1 and q_num_to_pad:
      attn_weights_slice[:, :, block_size - q_num_to_pad :, :] = 0

    attn_sum = (
      attn_weights_slice.view(batch_size, num_kv_head, block_size, -1, block_size)
      .sum(dim=-1)
      .sum(dim=-2)
    )
    attn_sums[:, :, i, :] = attn_sum
    del attn_weights_slice

  return attn_sums


class ProfileConfig:
  def __init__(
    self, p: float, stride: int, causal: bool, block_size: int,
    head_chunk_size: int = 4, estimate_chunk_size: int = 16384,
    mlp_chunk_size: int = 4096,
  ):
    if head_chunk_size < 0:
      raise ValueError("head_chunk_size must be non-negative; 0 profiles all heads together.")
    if stride <= 0 or block_size <= 0 or block_size % stride:
      raise ValueError("block_size must be positive and divisible by the positive stride.")
    alignment = math.lcm(block_size, stride * 64)
    if estimate_chunk_size <= 0 or estimate_chunk_size % alignment:
      raise ValueError(f"estimate_chunk_size must be a positive multiple of {alignment}.")
    if mlp_chunk_size < 0:
      raise ValueError("mlp_chunk_size must be non-negative; 0 disables MLP chunking.")
    self.p = p
    self.stride = stride
    self.history_threshold = []
    self.causal = causal
    self.block_size = block_size
    self.head_chunk_size = head_chunk_size
    self.estimate_chunk_size = estimate_chunk_size
    self.mlp_chunk_size = mlp_chunk_size


def module_layer_idx(module) -> int:
  if hasattr(module, "profile_layer_idx"):
    return int(module.profile_layer_idx)
  return int(getattr(module, "layer_idx", 0))


def profile_head_threshold(
  query_states: torch.Tensor,
  key_states: torch.Tensor,
  block_size: int,
  stride: int,
  chunk_size: int,
  causal: bool,
  p: float,
):
  attn_sum = chunk_prefill_to_attn_sum(
    query_states=query_states,
    key_states=key_states,
    block_size=block_size,
    causal=causal,
  )
  if not torch.isfinite(attn_sum).all():
    raise RuntimeError(
      f"Non-finite exact attention block sums on {query_states.device}; "
      "refusing to generate a threshold from invalid statistics."
    )

  xattn_sum, estimate_mask = xattn_estimate(
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
  del estimate_mask
  if not torch.isfinite(xattn_sum).all():
    raise RuntimeError(
      f"Non-finite XAttention estimates on {query_states.device}; "
      "refusing to generate a threshold from invalid statistics."
    )
  xattn_sum = xattn_sum[:, :, : attn_sum.shape[-1], : attn_sum.shape[-1]] * stride

  vals, idx = torch.sort(attn_sum, dim=-1, descending=True)
  cumsum = vals.cumsum(-1)
  need_len = (cumsum < p * cumsum[..., -1:]).sum(-1) + 1
  rank_idx = torch.arange(vals.size(-1), device=vals.device)
  sel_mask_srt = rank_idx.view(1, 1, 1, -1) < need_len.unsqueeze(-1)

  x_sorted = xattn_sum.gather(-1, idx)
  miniblock = x_sorted.masked_fill(~sel_mask_srt, float("inf")).min(-1).values
  mask_ge_min = xattn_sum >= miniblock.unsqueeze(-1)
  threshold_head = (xattn_sum * mask_ge_min).sum(-1).sum(-1) / (
    xattn_sum.sum(-1).sum(-1) + 1e-8
  )
  return threshold_head.detach().to("cpu", dtype=torch.float32)


def xattn_prefill_profile(
  self,
  query_states: torch.Tensor,
  key_states: torch.Tensor,
  value_states: torch.Tensor,
  block_size: int,
  stride: int,
  chunk_size: Optional[int] = None,
  causal: bool = True,
):
  num_heads = query_states.shape[1]
  num_kv_heads = key_states.shape[1]
  if num_kv_heads <= 0 or num_heads % num_kv_heads:
    raise ValueError("The number of query heads must be divisible by the number of KV heads.")
  kv_groups = num_heads // num_kv_heads
  head_chunk_size = self.profile_config.head_chunk_size or num_heads
  if chunk_size is None:
    chunk_size = self.profile_config.estimate_chunk_size
  head_thresholds = []
  for start in range(0, num_heads, head_chunk_size):
    end = min(start + head_chunk_size, num_heads)
    if kv_groups == 1:
      key_chunk = key_states[:, start:end]
    else:
      # Expand only the KV heads required by this chunk, keeping full K/V compact.
      kv_indices = torch.arange(start, end, device=key_states.device) // kv_groups
      key_chunk = key_states.index_select(1, kv_indices)
    head_thresholds.append(profile_head_threshold(
      query_states[:, start:end], key_chunk, block_size, stride,
      chunk_size, causal, self.profile_config.p,
    ))
    del key_chunk
  threshold_head = torch.cat(head_thresholds, dim=1)

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
  # Llama and Qwen2 share projection/RoPE/GQA semantics in Transformers 4.57.
  past_key_values = kwargs.pop("past_key_values", past_key_value)
  return forward_profile_qwen2(
    self,
    hidden_states,
    position_embeddings=position_embeddings,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    cache_position=cache_position,
    position_ids=position_ids,
    **kwargs,
  )


def forward_profile_qwen2(
  self,
  hidden_states: torch.Tensor,
  position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
  attention_mask: Optional[torch.Tensor] = None,
  past_key_values: Optional[object] = None,
  cache_position: Optional[torch.LongTensor] = None,
  **kwargs,
):
  """Profile Qwen2/Qwen3 attention under the Transformers 4.57 API.

  Qwen3 shares Qwen2's projection/GQA layout, but normalizes Q and K per head
  before applying RoPE. Keeping that operation here is essential: thresholds
  measured from unnormalized Q/K do not describe the attention used by Qwen3.
  """
  past_key_values = kwargs.pop("past_key_value", past_key_values)
  position_ids = kwargs.pop("position_ids", None)

  bsz, q_len, _ = hidden_states.size()
  num_heads = get_num_heads(self)
  num_kv_heads = get_num_kv_heads(self)

  query_states = self.q_proj(hidden_states)
  key_states = self.k_proj(hidden_states)
  value_states = self.v_proj(hidden_states)

  query_states = query_states.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
  key_states = key_states.view(bsz, q_len, num_kv_heads, self.head_dim).transpose(1, 2)
  value_states = value_states.view(bsz, q_len, num_kv_heads, self.head_dim).transpose(1, 2)

  if hasattr(self, "q_norm"):
    query_states = self.q_norm(query_states.transpose(1, 2)).transpose(1, 2)
    key_states = self.k_norm(key_states.transpose(1, 2)).transpose(1, 2)

  if position_embeddings is None:
    if position_ids is None:
      raise ValueError("Profiling forward requires `position_ids` or `position_embeddings`.")
    cos, sin = self.rotary_emb(value_states, position_ids)
  else:
    cos, sin = position_embeddings

  if cos.device != query_states.device:
    cos = cos.to(query_states.device)
  if sin.device != query_states.device:
    sin = sin.to(query_states.device)

  query_states, key_states = apply_rotary_pos_emb_local(query_states, key_states, cos, sin)

  if past_key_values is not None:
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    try:
      key_states, value_states = past_key_values.update(
        key_states,
        value_states,
        self.layer_idx,
        cache_kwargs,
      )
    except TypeError:
      key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

  if isinstance(past_key_values, StaticCache) and cache_position is not None:
    key_states = key_states[:, :, : min(cache_position[-1] + 1, key_states.shape[2]), :]
    value_states = value_states[:, :, : min(cache_position[-1] + 1, value_states.shape[2]), :]

  stride = self.profile_config.stride
  causal = self.profile_config.causal
  block_size = self.profile_config.block_size

  if key_states.shape[2] != q_len:
    raise AssertionError("Threshold profiling only supports full-sequence prefill without a KV cache.")

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

  return attn_output, None


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
    rotary_pos_emb = rotary_pos_emb.to(query_layer.device)
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

  stride = self.profile_config.stride
  causal = self.profile_config.causal
  block_size = self.profile_config.block_size

  if not decoding:
    if k_len != q_len:
      raise AssertionError("Threshold profiling only supports full-sequence prefill without a KV cache.")
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


def set_profiling_forward(module, forward_impl):
  bound_forward = forward_impl.__get__(module, type(module))
  if hasattr(module, "_hf_hook") and hasattr(module, "_old_forward"):
    # Accelerate's wrapper must still move inputs and offloaded weights per layer.
    module._old_forward = bound_forward
  else:
    module.forward = bound_forward


def forward_profile_mlp(self, hidden_states):
  """Bound dense MLP activations during inference; tokens are independent."""
  original_forward = self._profile_original_mlp_forward
  chunk_size = self.profile_mlp_chunk_size
  seq_len = hidden_states.shape[-2]
  if chunk_size == 0 or seq_len <= chunk_size:
    return original_forward(hidden_states)

  output = None
  for start in range(0, seq_len, chunk_size):
    end = min(start + chunk_size, seq_len)
    chunk_output = original_forward(hidden_states[..., start:end, :])
    if output is None:
      output = chunk_output.new_empty(*hidden_states.shape[:-1], chunk_output.shape[-1])
    # Preallocate once rather than retain all chunk outputs and concatenate,
    # which would duplicate the full-sequence output at the end.
    output[..., start:end, :].copy_(chunk_output)
    del chunk_output
  return output


def patch_mlp_for_profiling(module, chunk_size):
  if not hasattr(module, "_profile_original_mlp_forward"):
    if hasattr(module, "_hf_hook") and hasattr(module, "_old_forward"):
      module._profile_original_mlp_forward = module._old_forward
    else:
      module._profile_original_mlp_forward = module.forward
  module.profile_mlp_chunk_size = chunk_size
  set_profiling_forward(module, forward_profile_mlp)


def patch_llama_qwen_for_profiling(model, profile_config: ProfileConfig) -> str:
  model_type = getattr(model.config, "model_type", None)
  if model_type not in {"llama", "qwen2", "qwen3"}:
    raise NotImplementedError(f"Expected model_type llama/qwen2/qwen3, got {model_type!r}.")

  forward_impl = forward_profile_llama if model_type == "llama" else forward_profile_qwen2
  layers = model.model.layers
  for idx, layer in enumerate(layers):
    set_profiling_forward(layer.self_attn, forward_impl)
    layer.self_attn.profile_config = profile_config
    if not hasattr(layer.self_attn, "layer_idx"):
      layer.self_attn.layer_idx = idx
    layer.self_attn.profile_layer_idx = idx
    patch_mlp_for_profiling(layer.mlp, profile_config.mlp_chunk_size)
  return model_type


def patch_chatglm_for_profiling(model, profile_config: ProfileConfig) -> str:
  model_type = getattr(model.config, "model_type", None)
  if model_type != "chatglm":
    raise NotImplementedError(f"Expected model_type chatglm, got {model_type!r}.")

  layers = model.transformer.encoder.layers
  for idx, layer in enumerate(layers):
    set_profiling_forward(layer.self_attention, forward_profile_chatglm)
    layer.self_attention.profile_config = profile_config
    layer.self_attention.profile_layer_idx = idx
    layer.self_attention.layer_idx = idx
  return model_type


def patch_model_for_profiling(model, profile_config: ProfileConfig, requested_family: str) -> str:
  model_type = getattr(model.config, "model_type", None)

  if requested_family == "glm" or model_type == "chatglm":
    return patch_chatglm_for_profiling(model, profile_config)
  if requested_family in {"llama", "qwen2", "qwen3"} or model_type in {"llama", "qwen2", "qwen3"}:
    return patch_llama_qwen_for_profiling(model, profile_config)

  raise NotImplementedError(
    f"Unsupported model_type={model_type!r}. Supported families: llama, qwen2, qwen3, chatglm."
  )


def compute_final_threshold_from_history(profile_config: ProfileConfig) -> torch.Tensor:
  history_threshold = profile_config.history_threshold
  if len(history_threshold) == 0:
    raise RuntimeError("No profiling history found. Check input texts and forward patching.")
  return torch.cat([threshold.unsqueeze(0) for threshold in history_threshold]).max(0)[0]


def validate_threshold_shape(threshold: torch.Tensor, model, model_type: str) -> None:
  if model_type != "qwen3":
    return
  expected = (model.config.num_hidden_layers, model.config.num_attention_heads)
  if tuple(threshold.shape) != expected:
    raise RuntimeError(
      f"Qwen3 threshold table must have shape {expected}, got {tuple(threshold.shape)}. "
      "Profile one prompt at a time (batch size 1) and make sure every attention layer was patched."
    )


def get_profile_backbone(model):
  model_type = getattr(model.config, "model_type", None)
  if model_type in {"llama", "qwen2", "qwen3"}:
    return model.model
  if model_type == "chatglm":
    return model.transformer
  raise NotImplementedError(f"Unsupported profiling backbone for model_type={model_type!r}.")


def get_profile_input_device(model):
  embedding = model.get_input_embeddings()
  hook_device = getattr(getattr(embedding, "_hf_hook", None), "execution_device", None)
  if hook_device is not None:
    return torch.device(f"cuda:{hook_device}" if isinstance(hook_device, int) else hook_device)
  if embedding.weight.device.type == "meta":
    raise RuntimeError("Cannot resolve the input embedding execution device for the dispatched model.")
  return embedding.weight.device


def empty_model_cuda_cache(model):
  if not torch.cuda.is_available():
    return
  device_map = getattr(model, "hf_device_map", None)
  devices = device_map.values() if device_map else [get_profile_input_device(model)]
  cuda_devices = set()
  for device in devices:
    if isinstance(device, int):
      cuda_devices.add(device)
    elif str(device).startswith("cuda"):
      index = torch.device(device).index
      cuda_devices.add(torch.cuda.current_device() if index is None else index)
  for device in sorted(cuda_devices):
    with torch.cuda.device(device):
      torch.cuda.empty_cache()


def run_profile_on_texts(model, tokenizer, texts):
  backbone = get_profile_backbone(model)
  input_device = get_profile_input_device(model)
  for text in tqdm(texts):
    if not isinstance(text, str):
      raise TypeError("Each item in the profiling text file must be a string.")
    inputs = tokenizer(text, return_tensors="pt").to(input_device)
    logger.warning("Profiling text with %d tokens", inputs["input_ids"].shape[-1])
    with torch.no_grad():
      # Profiling runs in attention layers; skip the unused, enormous LM logits.
      # Discard outputs immediately so no previous sample survives the next forward.
      backbone(**inputs, use_cache=False)
    del inputs


def build_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--name_or_path", type=str, required=True)
  parser.add_argument(
    "--model_type",
    type=str,
    default="auto",
    choices=["auto", "llama", "qwen2", "qwen3", "glm"],
  )
  parser.add_argument("--p", type=float, required=True)
  parser.add_argument("--text_path", type=str, default="xattn/threshold/profile_threshold/text.json")
  parser.add_argument("--output_path", type=str, default="profiled_threshold.py")
  parser.add_argument("--output_format", type=str, default="auto", choices=["auto", "py", "json"])
  parser.add_argument("--output_var", type=str, default="")
  parser.add_argument("--stride", type=int, default=None)
  parser.add_argument("--block_size", type=int, default=128)
  parser.add_argument("--causal", type=str2bool, default=True)
  parser.add_argument("--device_map", type=str, default="")
  parser.add_argument(
    "--max_memory", type=str, default="",
    help='JSON weight budgets using visible GPU indices, e.g. {"0":"40GiB","1":"40GiB"}.',
  )
  parser.add_argument(
    "--reserve_memory_gib", type=float, default=8.0,
    help="Reserve this much free memory per GPU for activations when inferring a device map. "
         "An explicit --max_memory overrides automatic budgets.",
  )
  parser.add_argument(
    "--head_chunk_size", type=int, default=4,
    help="Number of attention heads profiled together per layer; 0 profiles all heads together.",
  )
  parser.add_argument(
    "--estimate_chunk_size", type=int, default=16384,
    help="Query tokens per XAttention estimate chunk; must align to block_size and stride*64.",
  )
  parser.add_argument(
    "--mlp_chunk_size", type=int, default=4096,
    help="Tokens per dense Llama/Qwen MLP forward to bound activation memory; 0 disables chunking.",
  )
  parser.add_argument("--torch_dtype", type=str, default="bfloat16")
  parser.add_argument("--trust_remote_code", type=str2bool, default=True)
  parser.add_argument("--attn_implementation", type=str, default="")
  parser.add_argument(
    "--rope_scaling_json",
    type=str,
    default="",
    help="Optional JSON object assigned to config.rope_scaling before model construction.",
  )
  parser.add_argument(
    "--max_position_embeddings",
    type=int,
    default=0,
    help="Optional context limit assigned before model construction; 0 preserves the model config.",
  )
  parser.add_argument("--chunk_texts", type=int, default=0)
  parser.add_argument(
    "--max_texts",
    type=int,
    default=0,
    help="Profile only the first N texts; 0 profiles the complete input file.",
  )
  parser.add_argument("--batch_output_dir", type=str, default="")
  parser.add_argument("--empty_cache_per_chunk", type=str2bool, default=True)
  parser.add_argument(
    "--resume_chunks",
    type=str2bool,
    default=False,
    help="Reuse valid chunk JSON files already present in the batch output directory.",
  )
  return parser


def main():
  logging.basicConfig(level=logging.WARNING)
  args = build_parser().parse_args()

  if not 0.0 < args.p <= 1.0:
    raise ValueError(f"p must be in (0, 1], got {args.p}")
  if args.max_texts < 0:
    raise ValueError(f"max_texts must be non-negative, got {args.max_texts}")
  if args.chunk_texts < 0:
    raise ValueError(f"chunk_texts must be non-negative, got {args.chunk_texts}")
  if args.max_position_embeddings < 0:
    raise ValueError(
      f"max_position_embeddings must be non-negative, got {args.max_position_embeddings}"
    )
  requested_family = args.model_type
  if requested_family == "auto":
    requested_family = infer_requested_family(args.name_or_path)

  if args.stride is None:
    args.stride = 16 if requested_family == "qwen3" or "deepseek" in args.name_or_path.lower() else 8
  if not args.device_map:
    args.device_map = "balanced"

  profile_config = ProfileConfig(
    p=args.p, stride=args.stride, causal=args.causal, block_size=args.block_size,
    head_chunk_size=args.head_chunk_size, estimate_chunk_size=args.estimate_chunk_size,
    mlp_chunk_size=args.mlp_chunk_size,
  )
  max_memory = resolve_max_memory(args.max_memory, args.reserve_memory_gib, args.device_map)
  ensure_xattn_imports()

  torch_dtype = resolve_torch_dtype(args.torch_dtype)
  load_kwargs = {
    "device_map": args.device_map,
    "torch_dtype": torch_dtype,
    "trust_remote_code": args.trust_remote_code,
  }
  if max_memory is not None:
    load_kwargs["max_memory"] = max_memory
    logger.warning("Model weight budgets (bytes): %s", max_memory)
  if args.rope_scaling_json or args.max_position_embeddings:
    model_config = AutoConfig.from_pretrained(
      args.name_or_path,
      trust_remote_code=args.trust_remote_code,
    )
    if args.max_position_embeddings:
      model_config.max_position_embeddings = args.max_position_embeddings
    if args.rope_scaling_json:
      rope_scaling = json.loads(args.rope_scaling_json)
      if not isinstance(rope_scaling, dict):
        raise ValueError("--rope_scaling_json must decode to a JSON object.")
      model_config.rope_scaling = rope_scaling
    load_kwargs["config"] = model_config
  if not args.attn_implementation and requested_family in {"qwen2", "qwen3"}:
    args.attn_implementation = "flash_attention_2"
  if args.attn_implementation:
    load_kwargs["attn_implementation"] = args.attn_implementation

  model = AutoModelForCausalLM.from_pretrained(args.name_or_path, **load_kwargs)
  model.eval()
  hf_device_map = getattr(model, "hf_device_map", None)
  if hf_device_map:
    used_devices = sorted({str(device) for device in hf_device_map.values()})
    logger.warning("Loaded model across devices: %s", ", ".join(used_devices))

  model_type = patch_model_for_profiling(model, profile_config, requested_family)
  logger.warning(
    "Profiling threshold: model_type=%s, p=%s, stride=%s, head_chunk_size=%s, estimate_chunk_size=%s, mlp_chunk_size=%s",
    model_type, args.p, args.stride, args.head_chunk_size, args.estimate_chunk_size,
    args.mlp_chunk_size,
  )

  tokenizer = AutoTokenizer.from_pretrained(
    args.name_or_path,
    trust_remote_code=args.trust_remote_code,
  )

  text_path = Path(args.text_path)
  with text_path.open("r", encoding="utf-8") as f:
    texts = json.load(f)

  if not isinstance(texts, list) or len(texts) == 0:
    raise ValueError(f"`{text_path}` must be a non-empty JSON list of strings.")
  if args.max_texts:
    texts = texts[: args.max_texts]

  output_path = Path(args.output_path)
  output_var = args.output_var or infer_output_var(model_type, args.name_or_path, args.p)

  if args.chunk_texts <= 0:
    run_profile_on_texts(model, tokenizer, texts)
    final_threshold = compute_final_threshold_from_history(profile_config).detach().cpu()
    validate_threshold_shape(final_threshold, model, model_type)
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
    chunk_file = batch_output_dir / f"threshold_chunk_{chunk_idx:03d}_{start}_{end}.json"

    if args.resume_chunks and chunk_file.is_file():
      chunk_threshold = load_threshold_json(chunk_file)
      validate_threshold_shape(chunk_threshold, model, model_type)
      if merged_threshold is None:
        merged_threshold = chunk_threshold
      else:
        merged_threshold = torch.maximum(merged_threshold, chunk_threshold)
      logger.warning(
        "Reused chunk %d: [%d, %d), loaded %s",
        chunk_idx,
        start,
        end,
        chunk_file,
      )
      continue

    profile_config.history_threshold = []
    run_profile_on_texts(model, tokenizer, texts[start:end])

    chunk_threshold = compute_final_threshold_from_history(profile_config).detach().cpu()
    validate_threshold_shape(chunk_threshold, model, model_type)
    save_threshold(chunk_threshold, chunk_file, output_var, "json")

    if merged_threshold is None:
      merged_threshold = chunk_threshold
    else:
      merged_threshold = torch.maximum(merged_threshold, chunk_threshold)

    if args.empty_cache_per_chunk:
      empty_model_cuda_cache(model)

    logger.warning("Finished chunk %d: [%d, %d), saved %s", chunk_idx, start, end, chunk_file)

  if merged_threshold is None:
    raise RuntimeError("No chunk threshold was generated.")

  save_threshold(merged_threshold, output_path, output_var, args.output_format)
  print(merged_threshold.tolist())
  print(f"Saved merged threshold file to: {output_path}")
  print(f"Saved chunk threshold files to: {batch_output_dir}")


if __name__ == "__main__":
  main()
