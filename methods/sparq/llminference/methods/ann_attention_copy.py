
"""Approximate nearest neighbour methods that approximate `Q @ K.T`."""

from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
from functools import partial
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np

import types


import json
import torch
from torch import Tensor, nn
from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaAttention, GemmaForCausalLM
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import (
  GPTNeoXAttention,
  GPTNeoXForCausalLM,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM

_SPARQ_BREAKDOWN_TIME_MANAGER = None


@contextmanager
def _measure_breakdown_component(component: str):
  """Record a component span without changing the surrounding other span state."""

  timing_manager = _SPARQ_BREAKDOWN_TIME_MANAGER
  if timing_manager is None:
    yield
    return

  record = timing_manager._begin_record(timing_manager.current_phase, component)
  try:
    yield
  finally:
    timing_manager._finish_record(record)

try:
  from transformers.models.llama.modeling_llama import (
    LlamaFlashAttention2,
    LlamaSdpaAttention,
  )
except ImportError:
  LlamaFlashAttention2 = None
  LlamaSdpaAttention = None

from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import (
  MistralAttention,
  MistralForCausalLM,
)
try:
  from transformers.models.mistral.modeling_mistral import (
    MistralFlashAttention2,
    MistralSdpaAttention,
  )
except ImportError:
  MistralFlashAttention2 = None
  MistralSdpaAttention = None

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
  Qwen2Attention, 
  Qwen2ForCausalLM,)
try:
  from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2FlashAttention2,
    Qwen2SdpaAttention,
  )
except ImportError:
  Qwen2FlashAttention2 = None
  Qwen2SdpaAttention = None

from .. import utility
from ..models import gemma_attention, llama_attention, mistral_attention, qwen_attention,llama_attention02
from . import sparse_attention


def _attention_types(*maybe_types: Any) -> Tuple[type, ...]:
  return tuple(t for t in maybe_types if isinstance(t, type))


LLAMA_ATTENTION_TYPES = _attention_types(
  LlamaAttention,
  LlamaSdpaAttention,
  LlamaFlashAttention2,
)
MISTRAL_ATTENTION_TYPES = _attention_types(
  MistralAttention,
  MistralSdpaAttention,
  MistralFlashAttention2,
)
QWEN2_ATTENTION_TYPES = _attention_types(
  Qwen2Attention,
  Qwen2SdpaAttention,
  Qwen2FlashAttention2,
)


GLM_CORE_ATTENTION_CLASS_NAMES = {"CoreAttention", "SdpaAttention", "FlashAttention2"}

def _looks_like_chatglm_model(model: nn.Module) -> bool:
  cls_name = model.__class__.__name__.lower()
  model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
  if "chatglm" in cls_name or model_type in {"chatglm", "glm"}:
    return True
  archs = [str(x).lower() for x in getattr(getattr(model, "config", None), "architectures", [])]
  return any("glm" in x for x in archs) and any(
    name.endswith("core_attention") for name, _ in model.named_modules()
  )


def _glm_build_logmask(
  query: Tensor,
  key: Tensor,
  attention_mask: Optional[Tensor],
) -> Tensor:
  """Convert GLM bool/additive masks into the additive mask expected by AnnAttention."""
  bsz, n_heads, q_len, _ = query.shape
  kv_len = key.shape[-2]
  dtype = query.dtype
  device = query.device

  if attention_mask is None:
    if q_len == kv_len:
      key_pos = torch.arange(kv_len, device=device)[None, :]
      query_pos = torch.arange(q_len, device=device)[:, None] + (kv_len - q_len)
      bool_mask = key_pos > query_pos
      bool_mask = bool_mask[None, None, :, :].expand(bsz, 1, q_len, kv_len)
    else:
      bool_mask = torch.zeros((bsz, 1, q_len, kv_len), device=device, dtype=torch.bool)
    return torch.zeros((bsz, n_heads, q_len, kv_len), device=device, dtype=dtype).masked_fill(
      bool_mask.expand(bsz, n_heads, q_len, kv_len),
      float("-inf"),
    )

  if attention_mask.dtype == torch.bool:
    if attention_mask.dim() == 2:
      attention_mask = attention_mask[:, None, None, :]
    elif attention_mask.dim() == 3:
      attention_mask = attention_mask[:, None, :, :]
    if attention_mask.shape[1] == 1:
      attention_mask = attention_mask.expand(bsz, n_heads, q_len, kv_len)
    else:
      attention_mask = attention_mask.broadcast_to(bsz, n_heads, q_len, kv_len)
    return torch.zeros((bsz, n_heads, q_len, kv_len), device=device, dtype=dtype).masked_fill(
      attention_mask,
      float("-inf"),
    )

  if attention_mask.dim() == 2:
    attention_mask = attention_mask[:, None, None, :]
  elif attention_mask.dim() == 3:
    attention_mask = attention_mask[:, None, :, :]
  if attention_mask.shape[1] == 1:
    attention_mask = attention_mask.expand(bsz, n_heads, q_len, kv_len)
  else:
    attention_mask = attention_mask.broadcast_to(bsz, n_heads, q_len, kv_len)
  return attention_mask.to(dtype=dtype)


def _glm_core_forward_with_ann(
  self: nn.Module,
  query_layer: Tensor,
  key_layer: Tensor,
  value_layer: Tensor,
  attention_mask: Optional[Tensor],
) -> Tensor:
  if query_layer.shape[-2] != 1:
    return self._ann_original_forward(query_layer, key_layer, value_layer, attention_mask)

  batch, n_heads, q_len, head_dim = query_layer.shape
  kv_len = key_layer.shape[-2]

  ann_num_kv_heads = int(getattr(self, "ann_num_kv_heads", key_layer.shape[1]))
  ann_key = key_layer
  ann_value = value_layer
  if key_layer.shape[1] != ann_num_kv_heads and key_layer.shape[1] % ann_num_kv_heads == 0:
    n_heads_per_kv = key_layer.shape[1] // ann_num_kv_heads
    ann_key = key_layer.view(batch, ann_num_kv_heads, n_heads_per_kv, kv_len, head_dim)[:, :, 0]
    ann_value = value_layer.view(batch, ann_num_kv_heads, n_heads_per_kv, kv_len, head_dim)[:, :, 0]

  logmask = _glm_build_logmask(query_layer, key_layer, attention_mask)
  ann_output, _ = self.ann(query_layer, ann_key, ann_value, logmask)

  context_layer = ann_output.transpose(1, 2).contiguous()
  new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
  context_layer = context_layer.reshape(*new_context_layer_shape)
  return context_layer


def _patch_glm_attention(model: nn.Module, settings: "Settings") -> int:
  named_modules = dict(model.named_modules())
  patched = 0

  for name, module in named_modules.items():
    if module.__class__.__name__ not in GLM_CORE_ATTENTION_CLASS_NAMES:
      continue
    if not hasattr(module, "hidden_size_per_partition") or not hasattr(module, "hidden_size_per_attention_head"):
      continue

    parent_name = name.rsplit(".", 1)[0] if "." in name else ""
    parent = named_modules.get(parent_name)

    num_heads = int(
      getattr(parent, "num_attention_heads_per_partition", None)
      or getattr(module, "num_attention_heads_per_partition", None)
      or 0
    )
    if num_heads <= 0:
      continue

    if parent is not None and getattr(parent, "multi_query_attention", False):
      num_kv_heads = int(getattr(parent, "num_multi_query_groups_per_partition", num_heads))
    else:
      num_kv_heads = num_heads

    if not hasattr(module, "_ann_original_forward"):
      module._ann_original_forward = module.forward

    module.ann = AnnAttention(settings, num_kv_heads, int(module.hidden_size_per_attention_head))
    layer_idx = getattr(parent, "layer_number", getattr(module, "layer_number", 0))
    module.ann.layer_idx = int(layer_idx)
    module.ann_num_kv_heads = num_kv_heads
    module.forward = types.MethodType(_glm_core_forward_with_ann, module)
    patched += 1

  return patched


def _resolve_num_kv_heads(module: nn.Module) -> int:
  n_kv_heads = getattr(module, "num_key_value_heads", None)
  if n_kv_heads is None:
    n_kv_heads = getattr(module, "num_kv_heads", None)
  if n_kv_heads is None:
    n_kv_heads = getattr(module, "num_heads", None)
  if n_kv_heads is None and hasattr(module, "config"):
    n_kv_heads = getattr(module.config, "num_key_value_heads", None)
  if n_kv_heads is None and hasattr(module, "config"):
    n_kv_heads = getattr(module.config, "num_attention_heads", None)
  if n_kv_heads is None:
    raise AttributeError("Cannot resolve number of KV heads for ANN attention")
  return int(n_kv_heads)



def _resolve_head_dim(module: nn.Module) -> int:
  head_dim = getattr(module, "head_dim", None)
  if head_dim is None:
    head_dim = getattr(module, "head_size", None)
  if head_dim is None and hasattr(module, "config"):
    hs = getattr(module.config, "hidden_size", None)
    nh = getattr(module, "num_heads", None) or getattr(
      module.config, "num_attention_heads", None
    )
    if hs is not None and nh is not None:
      head_dim = int(hs) // int(nh)
  if head_dim is None:
    raise AttributeError("Cannot resolve head dimension for ANN attention")
  return int(head_dim)


def gather(t: Tensor, dim: int, i: Tensor) -> Tensor:
  """A broadcasting version of torch.gather."""
  dim += (dim < 0) * t.ndim
  return t.gather(dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1 :]))


class LowRank(nn.Module):
  """Use a random orthonormal projection to down-project Q & K."""

  @dataclass
  class Settings:
    rank: int
    name: str = "low_rank"

  def __init__(self, settings: Settings, n_kv_heads: int, head_size: int):
    super().__init__()
    self.settings = settings
    self.weight = nn.Parameter(torch.empty(n_kv_heads, 1, head_size, settings.rank))
    for i in range(n_kv_heads): # can't batch this!
      nn.init.orthogonal_(self.weight[i, 0])

  def forward(self, query: Tensor, key: Tensor) -> Tensor:
    """Compute approximate score for each (query, key).

    query -- (batch, n_kv_heads, n_heads_per_kv, query, head_size)

    key -- (batch, n_kv_heads, 1, key, head_size)

    returns -- (batch, n_kv_heads, n_heads_per_kv, query, key)
    """
    query_proj = query.to(self.weight.dtype) @ self.weight
    key_proj = key.to(self.weight.dtype) @ self.weight
    return cast(
      Tensor,
      (query_proj.div(query.shape[-1] ** 0.5) @ key_proj.transpose(-1, -2)),
    )


class SparseQ(nn.Module):
  """Gather the top (absolute) components of Q from Q & K."""

  @dataclass
  class Settings:
    rank: int
    name: str = "sparse_q"

  def __init__(self, settings: Settings):
    super().__init__()
    self.settings = settings

  def forward(self, query: Tensor, key: Tensor) -> Tensor:
    """Compute approximate score for each (query, key).

    query -- (batch, n_kv_heads, n_heads_per_kv, 1, head_size)

    key -- (batch, n_kv_heads, 1, key, head_size)

    returns -- (batch, n_kv_heads, n_heads_per_kv, 1, key)
    """
    assert query.shape[-2] == 1, "no support for multiple queries"
    head_size = query.shape[-1]

    topk = query.abs().sum(dim=2, keepdim=True).topk(dim=-1, k=self.settings.rank)

    query_proj = gather(query, -1, topk.indices)
    key_proj = gather(key, -1, topk.indices)

    scale = (
      query_proj.abs()
      .sum(-1)
      .div_(query.abs().sum(-1))
      .mul_(head_size)
      .pow_(0.5)
      .unsqueeze(-1)
    )
    return query_proj.div(scale) @ key_proj.transpose(-1, -2)


ScoreSettings = Union[LowRank.Settings, SparseQ.Settings]


@dataclass
class Settings:
  k: int
  local_k: int
  reallocate_to_mean_value: bool
  score: ScoreSettings
  recall_save_path:Optional[str] = None
  type:str=""

  def __init__(
    self,
    k: int,
    local_k: int,
    reallocate_to_mean_value: bool,
    score: Union[ScoreSettings, str],
    recall_save_path:Optional[str]=None,
    type:str="",
    **args: Any,
  ):
    if isinstance(score, str):
      ctor: Any = dict(low_rank=LowRank.Settings, sparse_q=SparseQ.Settings)[
        score
      ]
      score_settings: ScoreSettings = ctor(**args)
    else:
      assert (
        not args
      ), "ann_attention.Setting only accepts **args when `score` is a string"
      score_settings = score
    self.k = k
    self.local_k = local_k
    self.reallocate_to_mean_value = reallocate_to_mean_value
    self.score = score_settings
    self.recall_save_path=recall_save_path
    self.type=type
    


class AnnAttention(nn.Module):
  """Generic ANN with local windowing and masking."""

  def __init__(self, settings: Settings, n_kv_heads: int, head_size: int):
    super().__init__()
    self.settings = settings
    self.score: nn.Module
    self._first_decode_seq: Optional[int] = None
    if isinstance(settings.score, LowRank.Settings):
      self.score = LowRank(settings.score, n_kv_heads, head_size)
    elif isinstance(settings.score, SparseQ.Settings):
      self.score = SparseQ(settings.score)
    else:
      raise ValueError(f"Unexpected settings.score = {settings.score}")
    self.debug_indices: Optional[List[Tensor]] = None
    self.current_sample_id: Optional[Any] = None
    self.share_kv_group_indices: bool = True

  def set_sample_id(self, sample_id: Optional[Any]) -> None:
    self.current_sample_id = sample_id
    self._first_decode_seq = None


  def _attention(
    self,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    logmask: Optional[Tensor],
    kv_weight: Tensor,
    mean_value: Tensor,
    return_attn_weights: bool = True,
  ) -> Tuple[Tensor, Optional[Tensor]]:
    """Dense attention, with left-over weight reallocation.

    query -- (batch, n_kv_heads, n_heads_per_kv, n_query, head_size)

    key -- (batch, n_kv_heads, 1, n_kv, head_size)

    value -- (batch, n_kv_heads, 1, n_heads, n_kv, head_size)

    logmask -- (batch, n_kv_heads, n_heads_per_kv, n_query, n_kv)

    kv_weight -- (batch, n_kv_heads, n_heads_per_kv, n_query) | ()
         -- 1.0 for regular attention (no reallocation)

    mean_value -- (batch, n_kv_heads, n_heads_per_kv, n_query, head_size)
    """
    timing_manager = _SPARQ_BREAKDOWN_TIME_MANAGER

    def attn_timing_context():
      return (
        timing_manager.measure("attn")
        if timing_manager is not None
        else nullcontext()
      )

    B, n_kv, nhpkv, n_q, d = query.shape
    _, _, key_heads, n_kv_seq, _ = key.shape
    is_shared = key_heads == 1

    output = None
    weights = None

    flash_dtype_ok = query.dtype in (torch.float16, torch.bfloat16)
    if query.is_cuda and flash_dtype_ok:
      try:
        from flash_attn import flash_attn_func

        q_flash = (
          query.contiguous()
          .reshape(B * n_kv, nhpkv, n_q, d)
          .transpose(1, 2)
          .contiguous()
        )
        if is_shared:
          k_flash = (
            key.squeeze(2)
            .contiguous()
            .reshape(B * n_kv, n_kv_seq, d)
            .unsqueeze(2)
          )
          v_flash = (
            value.squeeze(2)
            .contiguous()
            .reshape(B * n_kv, n_kv_seq, d)
            .unsqueeze(2)
          )
        else:
          k_flash = (
            key.contiguous()
            .reshape(B * n_kv, nhpkv, n_kv_seq, d)
            .transpose(1, 2)
            .contiguous()
          )
          v_flash = (
            value.contiguous()
            .reshape(B * n_kv, nhpkv, n_kv_seq, d)
            .transpose(1, 2)
            .contiguous()
          )
        with attn_timing_context():
          flash_output = flash_attn_func(
            q_flash,
            k_flash,
            v_flash,
            dropout_p=0.0,
            softmax_scale=d ** -0.5,
            causal=False,
            return_attn_probs=False,
          )
        output = (
          flash_output.transpose(1, 2)
          .contiguous()
          .reshape(B, n_kv, nhpkv, n_q, d)
        )
        output = kv_weight[..., None] * output
      except Exception:
        output = None

    if output is None:
      with attn_timing_context():
        scores = query.div(d ** 0.5) @ key.transpose(-1, -2)
        weights = torch.softmax(scores, -1, dtype=torch.float32).to(value.dtype)
        output = weights @ value
      output = kv_weight[..., None] * output
      weights *= kv_weight[..., None]
    elif return_attn_weights:
      scores = query.div(d ** 0.5) @ key.transpose(-1, -2)
      weights = torch.softmax(scores, -1, dtype=torch.float32).to(value.dtype)
      weights *= kv_weight[..., None]







      #   )





    output += (1 - kv_weight[..., None]) * mean_value
    #print("sparq")
    return output, weights


  def caculate_recall(
    self,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    logmask: Tensor,
    sparqIndices: Tensor,
  ) -> Tuple[Tensor, Tensor]:


    scores_full = query.div(query.shape[-1] ** 0.5) @ key.transpose(-1, -2)
    if logmask is not None:
      scores_full = scores_full.add(logmask)

    topk_score = scores_full # (B, n_kv, n_heads_per_kv, 1, seq)



    k_eff = min(self.settings.k + 1, topk_score.shape[-1])


    full_indices = topk_score.topk(k_eff, dim=-1).indices # (B, n_kv, 1, 1, k)
    
    sparq_indices = sparqIndices.expand(
      scores_full.shape[0],
      scores_full.shape[1],
      scores_full.shape[2],
      1,
      sparqIndices.shape[-1],
    )

    K = full_indices.shape[-1]

    real = full_indices.flatten(0, 2)  # (B*n_kv*n_heads_per_kv, 1, k)
    sparq = sparq_indices.flatten(0, 2) # (B*n_kv*n_heads_per_kv, 1, k)


    ground_truth_idx_cnt = full_indices.numel()
    hit_idx_cnt = 0
    recall_per_head = []

    for i in range(sparq.shape[0]):
      d = sparq[i, 0, :] # (k,)
      r = real[i, 0, :]  # (k,)


      assert d.numel() == torch.unique(d).numel()

      hit = (r.unsqueeze(-1) == d.unsqueeze(-2)).any(dim=-1) # (k,)
      hit_count = hit.int().sum()
      hit_idx_cnt += hit_count
      recall_per_head.append(float(hit_count) / float(k_eff))

    result = hit_idx_cnt.float() / float(ground_truth_idx_cnt)

    k100 = min(100, topk_score.shape[-1])
    full_indices_100 = topk_score.topk(k100, dim=-1).indices # (B, n_kv, 1, 1, k100)
    real_100 = full_indices_100.flatten(0, 2) # (B*n_kv*n_heads_per_kv, 1, k100)

    ground_truth_100_cnt = full_indices_100.numel()
    hit_100_cnt = 0
    recall_top100_per_head = []

    for i in range(sparq.shape[0]):
      d = sparq[i, 0, :] # (k,)
      r100 = real_100[i, 0, :] # (k100,)
      hit100 = (r100.unsqueeze(-1) == d.unsqueeze(-2)).any(dim=-1)
      hit100_count = hit100.int().sum()
      hit_100_cnt += hit100_count
      recall_top100_per_head.append(float(hit100_count) / float(k100))

    result_top100 = hit_100_cnt.float() / float(ground_truth_100_cnt)

    import os

    record = {
      "sample_id": self.current_sample_id,
      "layer":self.layer_idx,
      "k":K,
      "recall":float(result.item()),
      "hit_idx_cnt":int(hit_idx_cnt.item()) if torch.is_tensor(hit_idx_cnt) else int(hit_idx_cnt),
      "ground_truth_idx_cnt":int(ground_truth_idx_cnt.item()) if torch.is_tensor(ground_truth_idx_cnt) else int(ground_truth_idx_cnt),
      "recall_per_head": recall_per_head,
      "recall_top100": float(result_top100.item()),
      "recall_top100_per_head": recall_top100_per_head,
    }

    if os.environ.get("SPARQ_DUMP_RECALL_INDICES", "").lower() in {"1", "true", "yes"}:
      record["full_indices"] = full_indices.detach().cpu().tolist()
      record["sparq_indices"] = sparq_indices.detach().cpu().tolist()
      record["full_indices_top100"] = full_indices_100.detach().cpu().tolist()
    
    path = self.settings.recall_save_path

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not path.endswith(".jsonl"):
      path = path + "_recall.jsonl"
    with open(path, "a", encoding="utf-8") as f:
      json.dump(record, f, ensure_ascii=False)
      f.write("\n")
    

  def caculate_topk_rate(
    self,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    logmask: Tensor,
    sparqIndices: Tensor,
  ) -> Tuple[Tensor, Tensor]:


    scores_full = query.div(query.shape[-1] ** 0.5) @ key.transpose(-1, -2)
    if logmask is not None:
      scores_full = scores_full.add(logmask)

    full_weights = torch.softmax(scores_full, dim=-1, dtype=torch.float32)

    sparq_indices = sparqIndices # (B, n_kv, 1, 1, Ks)


    gather_idx = sparq_indices.expand(1, scores_full.shape[1], scores_full.shape[2], 1, sparq_indices.shape[-1])
    selected_mass = gather(full_weights, -1, gather_idx).sum(-1)

    mass_mean = selected_mass.mean().item()
    logit_sum_per_head = selected_mass.squeeze(-1).flatten().tolist()


    Ks = sparq_indices.shape[-1]

    record = {
      "sample_id": self.current_sample_id,
      "layer": self.layer_idx,
      "k": sparq_indices.shape[-1],
      "logit_sum_all": float(mass_mean),
      "logit_sum_per_head": logit_sum_per_head,
    }
    path = self.settings.recall_save_path
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not path.endswith(".jsonl"):
      path = path + "_attn_score.jsonl"
    with open(path, "a", encoding="utf-8") as f:
      json.dump(record, f, ensure_ascii=False)
      f.write("\n")





  def forward(
    self,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    logmask: Optional[Tensor],
    return_attn_weights: bool = True,
  ) -> Tuple[Tensor, Optional[Tensor]]:
    """Preprocess (key, value, mask) for ANN attention.

    query -- (batch, n_heads, 1, head_size)

    key -- (batch, n_kv_heads, seq, head_size)

    value -- (batch, n_kv_heads, seq, head_size)

    logmask -- (batch, n_heads, 1, seq)

    returns -- (output, weights)
          output -- (batch, n_heads, 1, head_size)
          weights -- (batch, n_heads, 1, seq)
    """

    batch, n_kv_heads, seq, head_size = key.shape
    n_heads_per_kv = query.shape[1] // n_kv_heads



    if logmask is None:
      query, key, value = map(
        partial(torch.unflatten, dim=1, sizes=(n_kv_heads, -1)),
        [query, key, value],
      )
    else:
      query, key, value, logmask = map(
        partial(torch.unflatten, dim=1, sizes=(n_kv_heads, -1)),
        [query, key, value, logmask],
      )

    assert query.shape == (batch, n_kv_heads, n_heads_per_kv, 1, head_size)
    assert key.shape == (batch, n_kv_heads, 1, seq, head_size)
    assert value.shape == (batch, n_kv_heads, 1, seq, head_size)
    if logmask is not None:
      assert logmask.shape == (batch, n_kv_heads, n_heads_per_kv, 1, seq)

    timing_manager = _SPARQ_BREAKDOWN_TIME_MANAGER
    timing_context = (
      timing_manager.measure("retrieve")
      if timing_manager is not None
      else nullcontext()
    )
    with timing_context:

      score = self.score(query, key)
      if logmask is not None:
        score = score + logmask
      score = score.float()


      if logmask is None:
        causal_index = torch.arange(seq - 1, -1, -1, device=query.device, dtype=torch.int32).view(1, 1, 1, 1, seq)
      else:
        causal_index = sparse_attention.causal_index(logmask)
      is_local = (0 <= causal_index) & (causal_index < self.settings.local_k + 1)




      topk_score = score.masked_fill(is_local, torch.finfo(score.dtype).max)
      if self.share_kv_group_indices:
        topk_score = topk_score.sum(dim=2, keepdim=True)





      indices = topk_score.topk(   
        min(self.settings.k + 1, score.shape[-1]), -1
      ).indices # shared: (batch, n_kv_heads, 1, 1, k+1); per-head: (batch, n_kv_heads, n_heads_per_kv, 1, k+1)
  


    if self.settings.type=="recall":
      self.caculate_recall(
        query=query,
        key=key,
        value=value,
        logmask=logmask,
        sparqIndices=indices)
      self.caculate_topk_rate(
        query=query,
        key=key,
        value=value,
        logmask=logmask,
        sparqIndices=indices)


    if self.debug_indices is not None:
      self.debug_indices.append(indices)

    with _measure_breakdown_component("attn"):


      if logmask is None:
        mean_value = value.mean(-2, dtype=torch.float32, keepdim=True).to(value.dtype)
      else:
        value_mask = (
          logmask[:, :, :1].squeeze(-2).unsqueeze(-1).exp()
        ) # (batch, n_kv_heads, 1, seq, 1)
        mean_value = (
          (value * value_mask)
          .sum(-2, dtype=torch.float32, keepdim=True)
          .div_(value_mask.sum(-2, dtype=torch.float32, keepdim=True))
          .to(value.dtype)
        ) # (batch, n_kv_heads, 1, 1, 1)
      

      kv_weight = torch.tensor(1.0, device=query.device)
      if self.settings.reallocate_to_mean_value:

        kv_weight = (
          gather(torch.softmax(score, -1), -1, indices)
          .sum(-1)
          .to(value.dtype)
        ) # (batch, n_kv_heads, n_heads_per_kv, 1)


    kv_indices = indices.squeeze(-2).unsqueeze(-1) # shared: (batch, n_kv_heads, 1, k+1, 1); per-head: (batch, n_kv_heads, n_heads_per_kv, k+1, 1)
    #
    selected_logmask = None if logmask is None else gather(logmask, -1, indices)
    key_for_gather = key
    value_for_gather = value
    if not self.share_kv_group_indices:
      key_for_gather = key.expand(batch, n_kv_heads, n_heads_per_kv, seq, head_size)
      value_for_gather = value.expand(
        batch, n_kv_heads, n_heads_per_kv, seq, head_size
      )

    output, weights = self._attention(
      query,
      gather(key_for_gather, -2, kv_indices),
      gather(value_for_gather, -2, kv_indices),
      selected_logmask,
      kv_weight=kv_weight,
      mean_value=mean_value,
      return_attn_weights=return_attn_weights,
    )


    if not return_attn_weights:
      return output.flatten(1, 2), None
    dense_weights = torch.zeros_like(score if logmask is None else logmask)
    weights = weights.to(dense_weights.dtype)
    return output.flatten(1, 2), dense_weights.scatter(
      -1, indices.expand_as(weights), weights
    ).flatten(1, 2)


Model = Union[
  GPTNeoXForCausalLM, LlamaForCausalLM, MistralForCausalLM, GemmaForCausalLM,Qwen2ForCausalLM
]


class GPTNeoXAttentionWithANN(GPTNeoXAttention): # type:ignore[misc]
  def __init__(self, config: GPTNeoXConfig, settings: Settings):
    super().__init__(config)
    self.ann = AnnAttention(settings, self.num_attention_heads, self.head_size)

  def _attn(
    self,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor] = None,
    head_mask: Optional[Tensor] = None,
  ) -> Tuple[Tensor, Tensor]:
    assert attention_mask is not None
    assert head_mask is None

    if query.shape[-2] == 1:
      return self.ann( # type:ignore[no-any-return]
        query,
        key,
        value,
        attention_mask.broadcast_to(key.unsqueeze(-3).shape[:-1]),
      )

    return super()._attn( # type:ignore[no-any-return]
      query, key, value, attention_mask, head_mask
    )


class LlamaAttentionWithANN(llama_attention02.LlamaAttention):
  def __init__(
    self, config: LlamaConfig, layer_idx: Optional[int], settings: Settings
  ):

    super().__init__(config, layer_idx)
    self.settings = settings
    self.ann = AnnAttention(
      settings,
      _resolve_num_kv_heads(self),
      _resolve_head_dim(self),
    )
    self.ann.layer_idx = layer_idx

  def _attn(
    self,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    logmask: Tensor,
  ) -> Tuple[Tensor, Tensor]:
    if query.shape[-2] == 1:

      # )
      if logmask is None:
        logmask = torch.zeros(
          query.shape[0],
          1,
          query.shape[-2],
          key.shape[-2],
          device=query.device,
          dtype=query.dtype,
        )
      return self.ann(
        query,
        key,
        value,
        logmask.broadcast_to(*query.shape[:-1], key.shape[-2]),
        return_attn_weights=False,
      )
    return super()._attn(query, key, value, logmask)


class GemmaAttentionWithANN(gemma_attention.GemmaAttention):
  def __init__(
    self, config: GemmaConfig, layer_idx: Optional[int], settings: Settings
  ):
    super().__init__(config, layer_idx)
    self.settings = settings
    self.ann = AnnAttention(
      settings,
      _resolve_num_kv_heads(self),
      _resolve_head_dim(self),
    )
    self.ann.layer_idx = layer_idx

  def _attn(
    self, query: Tensor, key: Tensor, value: Tensor, logmask: Tensor
  ) -> Tuple[Tensor, Tensor]:
    if query.shape[-2] == 1:
      return self.ann( # type:ignore[no-any-return]
        query,
        key,
        value,
        logmask.broadcast_to(*query.shape[:-1], key.shape[-2]),
        return_attn_weights=False,
      )
    return super()._attn(query, key, value, logmask)


class MistralAttentionWithANN(mistral_attention.MistralAttention):
  def __init__(
    self, config: MistralConfig, layer_idx: Optional[int], settings: Settings
  ):
    super().__init__(config, layer_idx)
    self.settings = settings
    self.ann = AnnAttention(
      settings,
      _resolve_num_kv_heads(self),
      _resolve_head_dim(self),
    )
    self.ann.layer_idx = layer_idx

  def _attn(
    self, query: Tensor, key: Tensor, value: Tensor, logmask: Tensor
  ) -> Tuple[Tensor, Tensor]:
    if query.shape[-2] == 1:
      return self.ann( # type:ignore[no-any-return]
        query,
        key,
        value,
        logmask.broadcast_to(*query.shape[:-1], key.shape[-2]),
      )
    return super()._attn(query, key, value, logmask)

class Qwen2AttentionWithANN(qwen_attention.Qwen2Attention):
  def __init__(
    self, config: Qwen2Config, layer_idx: Optional[int], settings: Settings
  ):
    super().__init__(config, layer_idx)

    self.settings = settings
    self.ann = AnnAttention(
      settings,
      _resolve_num_kv_heads(self),
      _resolve_head_dim(self),
    )
    self.ann.layer_idx = getattr(self, "layer_idx", layer_idx)

    self.ann.share_kv_group_indices = True

  def _attn(
    self,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    logmask: Tensor,
  ) -> Tuple[Tensor, Tensor]:

    if query.shape[-2] == 1:

      # )
      ann_logmask = None if logmask is None else logmask.broadcast_to(*query.shape[:-1], key.shape[-2])
      return self.ann(
        query,
        key,
        value,
        ann_logmask,
        return_attn_weights=False,
      )

    return super()._attn(query, key, value, logmask)

def convert(model: Model, settings: Settings) -> Model:
  """Convert a model to use KV cache compression using ANN."""


  if _looks_like_chatglm_model(model):
    patched = _patch_glm_attention(model, settings)
    if patched > 0:
      return model

  def _copy_attention_weights(src: nn.Module, dst: nn.Module) -> nn.Module:
    dst.load_state_dict(src.state_dict(), strict=False)
    return dst

  def _replace(m: nn.Module) -> Optional[nn.Module]:

    if isinstance(m, QWEN2_ATTENTION_TYPES):
      return Qwen2AttentionWithANN(model.config, getattr(m, "layer_idx", None), settings)
    if isinstance(m, GPTNeoXAttention):
      return GPTNeoXAttentionWithANN(model.config, settings)
    if isinstance(m, LlamaAttention):
      return _copy_attention_weights(m, LlamaAttentionWithANN(model.config, m.layer_idx, settings))
    if isinstance(m, MISTRAL_ATTENTION_TYPES):
      return MistralAttentionWithANN(model.config, getattr(m, "layer_idx", None), settings)
    if isinstance(m, GemmaAttention):
      return GemmaAttentionWithANN(model.config, getattr(m, "layer_idx", None), settings)

  return utility.convert_module(model, _replace)


def set_current_sample_id(model: nn.Module, sample_id: Optional[Any]) -> None:
  for module in model.modules():
    if isinstance(module, AnnAttention):
      module.set_sample_id(sample_id)
