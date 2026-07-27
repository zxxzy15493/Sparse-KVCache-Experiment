from typing import Optional, Tuple
import time
import os
import pickle
import math

import torch
import flashinfer
try:
  from flash_attn import flash_attn_func
except Exception:
  flash_attn_func = None
from transformers import AutoTokenizer, StaticCache
from transformers.cache_utils import Cache
from transformers.utils import logging
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from threshold.threshold.qwen_threshold import qwen_fuse_16, qwen_fuse_8, qwen_fuse_4,qwen_fuse_80,qwen_fuse_85,qwen_fuse_90,qwen_fuse_95,qwen_little_80
try:
  from xattn.src.Xattention import Xattention_prefill
except Exception:
  print("Xattention Import Fail")

from xattn.src.utils import *

logger = logging.get_logger(__name__)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
  """Rotates half the hidden dims of the input (for RoPE)."""
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
  q: torch.Tensor,
  k: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  position_ids: Optional[torch.LongTensor] = None,
  unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Applies Rotary Position Embedding to the query and key tensors.

  Args:
    q (`torch.Tensor`): The query tensor.
    k (`torch.Tensor`): The key tensor.
    cos (`torch.Tensor`): The cosine part of the rotary embedding.
    sin (`torch.Tensor`): The sine part of the rotary embedding.
    position_ids (`torch.Tensor`, *optional*): Deprecated and unused.
    unsqueeze_dim (`int`, *optional*, defaults to 1):
      Dimension along which to unsqueeze cos and sin to broadcast
      to q and k.

  Returns:
    (q_embed, k_embed): RoPE-applied query and key tensors.
  """
  cos = cos.unsqueeze(unsqueeze_dim)
  sin = sin.unsqueeze(unsqueeze_dim)
  q_embed = (q * cos) + (rotate_half(q) * sin)
  k_embed = (k * cos) + (rotate_half(k) * sin)
  return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
  """
  Equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep).

  (batch, num_kv_heads, seqlen, head_dim)
  -> (batch, num_attention_heads, seqlen, head_dim)
  """
  batch, num_key_value_heads, slen, head_dim = hidden_states.shape
  if n_rep == 1:
    return hidden_states
  hidden_states = hidden_states[:, :, None, :, :].expand(
    batch, num_key_value_heads, n_rep, slen, head_dim
  )
  return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def dense_decode_attention(
  query_states: torch.Tensor,
  key_states: torch.Tensor,
  value_states: torch.Tensor,
  attention_mask: Optional[torch.Tensor],
  num_key_value_groups: int,
) -> torch.Tensor:
  if key_states.shape[1] != query_states.shape[1]:
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

  attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(query_states.shape[-1])
  if attention_mask is not None:
    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = attn_weights + causal_mask

  attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
  return torch.matmul(attn_weights, value_states)


def flash_attn_decode_attention(
  query_states: torch.Tensor,
  key_states: torch.Tensor,
  value_states: torch.Tensor,
  attention_mask: Optional[torch.Tensor],
  num_key_value_groups: int,
) -> torch.Tensor:


  if (
    flash_attn_func is None
    or not query_states.is_cuda
    or query_states.dtype not in (torch.float16, torch.bfloat16)
  ):
    return dense_decode_attention(
      query_states,
      key_states,
      value_states,
      attention_mask,
      num_key_value_groups,
    )

  q_flash = query_states.transpose(1, 2).contiguous()
  k_flash = key_states.transpose(1, 2).contiguous()
  v_flash = value_states.transpose(1, 2).contiguous()

  try:
    attn_output = flash_attn_func(
      q_flash,
      k_flash,
      v_flash,
      dropout_p=0.0,
      softmax_scale=query_states.shape[-1] ** -0.5,
      causal=False,
      return_attn_probs=False,
    )
  except RuntimeError:
    return dense_decode_attention(
      query_states,
      key_states,
      value_states,
      attention_mask,
      num_key_value_groups,
    )

  return attn_output.transpose(1, 2).contiguous()


class FastPrefillConfig(dict):
  """
  Configuration class for FastPrefill, which provides flexible settings
  for optimizing prefill computations in transformer models.

  Attributes:
    threshold (float or torch.Tensor, optional): Threshold for selecting
      relevant attention blocks.
    print_detail (bool): Whether to print detailed timing/debug info.
    stride (int): Fused attention block size (e.g., 16, 8, or 4).
    metric (str): Type of prefill mechanism used
      ('xattn').
  """

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
      self.threshold = torch.ones((28, 28), device="cuda") * threshold
    elif p == 0.9:
      self.threshold = torch.tensor(qwen_fuse_90)
    elif p == 0.85:
      self.threshold = torch.tensor(qwen_fuse_85)
    elif p == 0.8:
      self.threshold = torch.tensor(qwen_fuse_80)
    elif p == 0.95:
      self.threshold = torch.tensor(qwen_fuse_95)
    else:
      raise ValueError(f"Unsupported p: {p}")


def _layer_threshold(config: FastPrefillConfig, layer_idx: int, num_heads: int, device: torch.device):
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
    raise ValueError("Threshold table row is empty; pass a scalar --threshold for this model.")
  if values.numel() >= num_heads:
    values = values[:num_heads]
  else:
    values = torch.cat([values, values.mean().expand(num_heads - values.numel())])
  return values.to(device)


def forward_eval(
  self,
  hidden_states: torch.Tensor,
  attention_mask: Optional[torch.Tensor] = None,
  position_ids: Optional[torch.LongTensor] = None,
  past_key_value: Optional[Cache] = None,
  output_attentions: bool = False,
  use_cache: bool = False,
  cache_position: Optional[torch.LongTensor] = None,
  position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
  **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
  """
  Forward pass of the Qwen2 attention layer with optimized prefill mechanisms.

  Integrates:
  - Fused/approximate attention for prefill (xattn)
  - Efficient KV caching
  - Rotary embeddings (RoPE)
  - Dense attention fallback for single-token decoding

  Returns:
    (attn_output, None, past_key_value)
  """
  if self.fastprefillconfig.print_detail:
    start_time = time.time()

  bsz, q_len, _ = hidden_states.size()


  query_states = self.q_proj(hidden_states)
  key_states = self.k_proj(hidden_states)
  value_states = self.v_proj(hidden_states)

  query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
  key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
  value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

  if self.fastprefillconfig.print_detail:
    torch.cuda.synchronize()
    reshape_time = time.time() - start_time
    print(f"   Reshape took: {reshape_time:.6f} seconds")

  if position_embeddings is None:
    logger.warning_once(
      "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
      "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
      "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
      "removed and `position_embeddings` will be mandatory."
    )
    cos, sin = self.rotary_emb(value_states, position_ids)
  else:
    cos, sin = position_embeddings

  query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


  if self.fastprefillconfig.print_detail:
    start_time = time.time()

  if past_key_value is not None:
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_value.update(
      key_states, value_states, self.layer_idx, cache_kwargs
    )

  if isinstance(past_key_value, StaticCache):
    key_states = key_states[:, :, : min(cache_position[-1] + 1, key_states.shape[2]), :]
    value_states = value_states[:, :, : min(cache_position[-1] + 1, value_states.shape[2]), :]

  _, _, k_len, _ = key_states.shape
  _, _, q_len, _ = query_states.shape
  decoding = (q_len != k_len and q_len == 1)

  if not decoding:
    key_states = repeat_kv(key_states, self.num_key_value_groups).to(query_states.device)
    value_states = repeat_kv(value_states, self.num_key_value_groups).to(query_states.device)

  if self.fastprefillconfig.print_detail:
    torch.cuda.synchronize()
    past_kv_time = time.time() - start_time
    print(f"   Past KV update and repeat took: {past_kv_time:.6f} seconds")


  if self.fastprefillconfig.print_detail:
    start_time = time.time()
    print(f"q length: {q_len} k length: {k_len}")

  stride = self.fastprefillconfig.stride

  if not decoding:

    if self.fastprefillconfig.metric == "xattn":
      threshold = _layer_threshold(
        self.fastprefillconfig,
        self.layer_idx,
        self.num_heads,
        query_states.device,
      )
      modelName="Qwen"
      layer_id = int(getattr(self, "layer_idx", -1))
      attn_output = Xattention_prefill(
        query_states,
        key_states,
        value_states,
        stride,
        model_name=modelName,
        layer_id=layer_id,
        norm=1,
        threshold=threshold,
        use_triton=True,
      )
    else:
      raise ValueError(f"Unknown metric: {self.fastprefillconfig.metric}")
  else:

    if key_states.device != query_states.device:
      key_states = key_states.to(query_states.device)
    if value_states.device != query_states.device:
      value_states = value_states.to(query_states.device)

    attn_output = flash_attn_decode_attention(
      query_states,
      key_states,
      value_states,
      attention_mask,
      self.num_key_value_groups,
    )

  if self.fastprefillconfig.print_detail:
    torch.cuda.synchronize()
    attn_time = time.time() - start_time
    print(f"   Attention computation took: {attn_time:.6f} seconds")


  if self.fastprefillconfig.print_detail:
    start_time = time.time()

  if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
    raise ValueError(
      f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
      f" {attn_output.size()}"
    )

  attn_output = attn_output.transpose(1, 2).contiguous()
  attn_output = attn_output.reshape(bsz, q_len, -1)
  attn_output = self.o_proj(attn_output)
  del query_states

  if self.fastprefillconfig.print_detail:
    torch.cuda.synchronize()
    post_attn_time = time.time() - start_time
    print(f"   Post-attention processing took: {post_attn_time:.6f} seconds")

  return attn_output, None, past_key_value


def forward_to_save(
  self,
  hidden_states: torch.Tensor,
  attention_mask: Optional[torch.Tensor] = None,
  position_ids: Optional[torch.LongTensor] = None,
  past_key_value: Optional[Cache] = None,
  output_attentions: bool = False,
  use_cache: bool = False,
  cache_position: Optional[torch.LongTensor] = None,
  position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
  **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
  """
  """
  if self.fastprefillconfig.print_detail:
    start_time = time.time()

  bsz, q_len, _ = hidden_states.size()

  query_states = self.q_proj(hidden_states)
  key_states = self.k_proj(hidden_states)
  value_states = self.v_proj(hidden_states)

  query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
  key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
  value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

  if self.fastprefillconfig.print_detail:
    torch.cuda.synchronize()
    reshape_time = time.time() - start_time
    print(f"   Reshape took: {reshape_time:.6f} seconds")

  if position_embeddings is None:
    logger.warning_once(
      "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
      "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
      "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
      "removed and `position_embeddings` will be mandatory."
    )
    cos, sin = self.rotary_emb(value_states, position_ids)
  else:
    cos, sin = position_embeddings

  query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


  if self.fastprefillconfig.print_detail:
    start_time = time.time()

  if past_key_value is not None:
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_value.update(
      key_states, value_states, self.layer_idx, cache_kwargs
    )

  if isinstance(past_key_value, StaticCache):
    key_states = key_states[:, :, : min(cache_position[-1] + 1, key_states.shape[2]), :]
    value_states = value_states[:, :, : min(cache_position[-1] + 1, value_states.shape[2]), :]

  _, _, k_len, _ = key_states.shape
  _, _, q_len, _ = query_states.shape
  decoding = (q_len != k_len and q_len == 1)

  if not decoding:
    key_states = repeat_kv(key_states, self.num_key_value_groups).to(query_states.device)
    value_states = repeat_kv(value_states, self.num_key_value_groups).to(query_states.device)

  if self.fastprefillconfig.print_detail:
    torch.cuda.synchronize()
    past_kv_time = time.time() - start_time
    print(f"   Past KV update and repeat took: {past_kv_time:.6f} seconds")


  if self.fastprefillconfig.print_detail:
    start_time = time.time()
    print(f"q length: {q_len} k length: {k_len}")

  stride = self.fastprefillconfig.stride

  if not decoding:
    if self.fastprefillconfig.metric == "xattn":
      threshold = _layer_threshold(
        self.fastprefillconfig,
        self.layer_idx,
        self.num_heads,
        query_states.device,
      )
      modelName="Qwen"
      layer_id = int(getattr(self, "layer_idx", -1))
      attn_output = Xattention_prefill(
        query_states,
        key_states,
        value_states,
        stride,
        model_name=modelName,
        layer_id=layer_id,
        norm=1,
        threshold=threshold,
        use_triton=True,
      )
    else:
      raise ValueError(f"Unknown metric: {self.fastprefillconfig.metric}")
  else:
    if key_states.device != query_states.device:
      key_states = key_states.to(query_states.device)
    if value_states.device != query_states.device:
      value_states = value_states.to(query_states.device)

    attn_output = dense_decode_attention(
      query_states,
      key_states,
      value_states,
      attention_mask,
      self.num_key_value_groups,
    )


  if getattr(self, "layer_idx", None) == getattr(self, "layer_to_save", None):
    os.makedirs("output", exist_ok=True)
    query_path = f"output/query_{self.target_len}.pkl"
    key_path = f"output/key_{self.target_len}.pkl"


    if os.path.exists(query_path) and os.path.getsize(query_path) > 0:
      with open(query_path, "rb") as f:
        loaded_query = pickle.load(f)
      with open(query_path, "wb") as f:
        pickle.dump(torch.cat([loaded_query, query_states], dim=-2), f)
      del loaded_query
    else:
      with open(query_path, "wb") as f:
        pickle.dump(query_states, f)


    if key_states.shape[-2] == self.target_len:
      with open(key_path, "wb") as f:
        pickle.dump(key_states, f)

  if self.fastprefillconfig.print_detail:
    torch.cuda.synchronize()
    attn_time = time.time() - start_time
    print(f"   Attention computation took: {attn_time:.6f} seconds")


  if self.fastprefillconfig.print_detail:
    start_time = time.time()

  if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
    raise ValueError(
      f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
      f" {attn_output.size()}"
    )

  attn_output = attn_output.transpose(1, 2).contiguous()
  attn_output = attn_output.reshape(bsz, q_len, -1)
  attn_output = self.o_proj(attn_output)
  del query_states

  if self.fastprefillconfig.print_detail:
    torch.cuda.synchronize()
    post_attn_time = time.time() - start_time
    print(f"   Post-attention processing took: {post_attn_time:.6f} seconds")

  return attn_output, None, past_key_value


def load_model(
  fastprefillconfig: FastPrefillConfig = FastPrefillConfig(),
  name_or_path: str = "",
):
  """
  Load Qwen2 model with FastPrefill.

  Args:
    fastprefillconfig: FastPrefillConfig instance
    name_or_path: Model name or local path, e.g. "Qwen/Qwen2-7B-Instruct"

  Returns:
    (model, tokenizer)
  """
  model = Qwen2ForCausalLM.from_pretrained(
    name_or_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
  )
  model.eval()

  for layer in model.model.layers:
    layer.self_attn.fastprefillconfig = fastprefillconfig
    layer.self_attn.forward = forward_eval.__get__(layer.self_attn)

  tokenizer = AutoTokenizer.from_pretrained(name_or_path)
  return model, tokenizer


def load_fake_model(
  layer_to_save: int,
  target_len: int,
  name_or_path: str = "",
):
  """
   Load a debug Qwen2 model; saves Q/K of specified layers to output/ during forward.

  Args:
    layer_to_save: Layer index to save (0-based)
    target_len: Save full K when key length reaches this value
    name_or_path: model name or local path

  Returns:
    (model, tokenizer)
  """
  model = Qwen2ForCausalLM.from_pretrained(
    name_or_path,
    device_map="balanced",
    torch_dtype=torch.bfloat16,
  )
  model.eval()

  for layer in model.model.layers:
    layer.self_attn.fastprefillconfig = FastPrefillConfig()
    layer.self_attn.layer_to_save = layer_to_save
    layer.self_attn.target_len = target_len
    layer.self_attn.forward = forward_to_save.__get__(layer.self_attn)

  tokenizer = AutoTokenizer.from_pretrained(name_or_path)
  return model, tokenizer
