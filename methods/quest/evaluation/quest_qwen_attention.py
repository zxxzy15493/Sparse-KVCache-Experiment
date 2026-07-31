"""
Quest attention patch for Qwen2/Qwen2.5 models (Transformers).

This file mirrors the structure of `quest_attention.py` (LLaMA) but targets the
Qwen2 attention modules in Hugging Face Transformers.

Typical usage (eval / inference):
  from quest_qwen_attention import enable_quest_qwen_attention_eval
  enable_quest_qwen_attention_eval(model, args)

`args` needs:
  - args.token_budget (int): how many KV tokens to keep per head (approx.)
  - args.chunk_size (int): chunk size for local heavy-hitter selection
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn
import types
import threading

from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np

import json
import torch
from torch import Tensor, nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
  Qwen2Attention, 
  Qwen2ForCausalLM,)


try:
  from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2FlashAttention2,
    Qwen2SdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
  )
except Exception as e: # pragma: no cover
  Qwen2Attention = None
  Qwen2FlashAttention2 = None
  Qwen2SdpaAttention = None
  apply_rotary_pos_emb = None
  repeat_kv = None
  _IMPORT_ERROR = e

try:
  from transformers.cache_utils import DynamicCache
except Exception: # pragma: no cover
  DynamicCache = tuple() # fallback for isinstance checks


_file_lock = threading.Lock()

layerID=1


def local_heavy_hitter_mask(attn_weights: torch.Tensor, token_budget: int, chunk_size: int) -> torch.Tensor:


  seq_length = attn_weights.shape[-1]

  padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)

  attn_weights = torch.cat(
    [
      attn_weights,
      torch.ones(
        (
          attn_weights.shape[0],
          attn_weights.shape[1],
          attn_weights.shape[2],
          padding_length,
        ),
        device=attn_weights.device,
        dtype=attn_weights.dtype,
      )
      * torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device, dtype=attn_weights.dtype),
    ],
    dim=-1,
  )


  chunk_attn_weights = attn_weights.reshape(
    attn_weights.shape[0],
    attn_weights.shape[1],
    attn_weights.shape[2],
    attn_weights.shape[3] // chunk_size,
    chunk_size,
  ).amax(dim=-1)


  # )

  _, topk = chunk_attn_weights.topk(
    k=min(max(3, math.ceil(token_budget / chunk_size)), chunk_attn_weights.size(-1)),
    dim=-1,
  )





  topk = (
    topk.unsqueeze(-1).repeat(1, 1, 1, 1, chunk_size) * chunk_size
    + torch.arange(chunk_size, device=topk.device)
  )
  topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)

  


  

  mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
  mask_bottom.scatter_(-1, topk, True)

  mask_bottom = mask_bottom[:, :, :, :seq_length]
  return mask_bottom


def quest_select_token_indices_from_chunks(
  chunk_scores: torch.Tensor,
  token_budget: int,
  chunk_size: int,
  seq_length: int,
):
  """
  chunk_scores: (bsz, num_heads, q_len, num_chunks)
  return:
    selected_token_idx: (bsz, num_heads, q_len, selected_tokens)
    selected_token_valid: bool mask, same shape
  """
  num_chunks = chunk_scores.size(-1)
  num_select_chunks = min(
    max(3, math.ceil(token_budget / chunk_size)),
    num_chunks,
  )

  top_chunks = chunk_scores.topk(
    k=num_select_chunks,
    dim=-1,
  ).indices

  offsets = torch.arange(
    chunk_size,
    device=chunk_scores.device,
    dtype=top_chunks.dtype,
  )

  selected_token_idx = top_chunks.unsqueeze(-1) * chunk_size + offsets
  selected_token_idx = selected_token_idx.reshape(
    *top_chunks.shape[:-1],
    num_select_chunks * chunk_size,
  )

  selected_token_valid = selected_token_idx < seq_length
  selected_token_idx = selected_token_idx.clamp(max=seq_length - 1)

  return selected_token_idx, selected_token_valid


def calculate_recall_from_mask(
  token_pos:int,
  layerid:int,
  chunk_size:int,
  token_budget: int,    #q=1  nheads=num of query head
  scores_full: torch.Tensor,
  quest_mask: torch.Tensor,
  save_path: str = None,
  task_name: str = None,
):
  

  seq = scores_full.shape[-1]
  k_eff = min(token_budget, seq)

  gt_idx = scores_full.topk(k_eff, dim=-1).indices




  hit = quest_mask.gather(-1, gt_idx)     # (bs, n_heads, q, k_eff) bool
  recall = hit.float().mean()
  recall_per_head = hit.float().mean(dim=-1).squeeze(0).tolist()


  k_100 = min(100, seq)
  gt_idx_100 = scores_full.topk(k_100, dim=-1).indices
  hit_100 = quest_mask.gather(-1, gt_idx_100)
  recall_100 = float(hit_100.float().mean())
  recall_100_per_head = hit_100.float().mean(dim=-1).squeeze(0).tolist() # modify

  from pathlib import Path
  import json


  if save_path is not None:
    outdir = Path(save_path)
  else:
    outdir = Path("efficiency/recall-results")
  outdir.mkdir(parents=True, exist_ok=True)


  if task_name is not None:
    outpath = outdir / f"{task_name}_recall.jsonl"
  else:
    suffix = f'chunk_size{chunk_size}-tokenbudget{token_budget}'
    outpath = outdir / f"{suffix}.jsonl"



  record = {
        "token_pos":token_pos,
        "layer": layerid,
        "k":token_budget,
        "recall": float(recall),
        "recall_100": recall_100,
        "recall_per_head": recall_per_head,
        "recall_100_per_head": recall_100_per_head,
      }
  with _file_lock:
    with open(outpath, "a", encoding="utf-8") as f:
      json.dump(record, f, ensure_ascii=False)
      f.write("\n")

  return recall



def full_attention_topk_mask(
  attn_logits: torch.Tensor,
  token_budget: int,
  attention_mask: torch.Tensor | None = None,  # additive mask (bs,1,q,k) or (bs,n_heads,q,k)
  position_ids: torch.Tensor | None = None,
):

  logits = attn_logits






  k_eff = min(int(token_budget), logits.shape[-1])
  topk_idx = logits.topk(k_eff, dim=-1).indices      # (bs, n_heads, q, k_eff)


  topk_mask = torch.zeros_like(logits, dtype=torch.bool) # (bs, n_heads, q, k)
  topk_mask.scatter_(-1, topk_idx, True)
  return topk_mask, topk_idx




def _filter_kwargs_for_fn(fn, kwargs):
  """Filter kwargs so we don't pass unexpected keys to different Transformers versions."""
  try:
    import inspect
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}
  except Exception:
    return kwargs


def _get_position_ids(
  hidden_states: torch.Tensor,
  position_ids: Optional[torch.LongTensor],
  cache_position: Optional[torch.LongTensor],
  past_key_value,
) -> torch.LongTensor:
  """
  Try to construct position_ids if not provided.
  For generation, Transformers usually passes position_ids already; this is a safe fallback.
  """
  if position_ids is not None:
    return position_ids

  bsz, q_len, _ = hidden_states.shape
  device = hidden_states.device

  if cache_position is not None:
    if cache_position.dim() == 1:
      return cache_position.view(1, -1).expand(bsz, -1).to(device)
    return cache_position.to(device)

  past_len = 0
  if past_key_value is not None:
    if hasattr(past_key_value, "get_seq_length"):
      try:
        past_len = int(past_key_value.get_seq_length())
      except Exception:
        past_len = 0
    elif isinstance(past_key_value, (tuple, list)) and len(past_key_value) > 0:
      past_len = int(past_key_value[0].shape[-2])

  return (torch.arange(past_len, past_len + q_len, device=device, dtype=torch.long)
      .view(1, -1).expand(bsz, -1))




def _get_rope_cos_sin(
  self,
  value_states: torch.Tensor,
  position_ids: torch.LongTensor,
  kv_seq_len: int,
  position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
  """Support both older and newer Qwen2 RoPE APIs + optional precomputed position_embeddings."""
  if position_embeddings is not None:
    return position_embeddings # (cos, sin)

  try:
    return self.rotary_emb(value_states, position_ids.to(value_states.device))
  except TypeError:
    pass

  try:
    return self.rotary_emb(value_states, seq_len=kv_seq_len)
  except TypeError:
    pass

  return self.rotary_emb(value_states, kv_seq_len)

def calculate_topkrate(
  token_pos:int,
  layerid:int,
  chunk_size:int,
  token_budget:int,
  scores_full: torch.Tensor,  # (bs, n_heads, q, seq) full-attn logits (already masked)
  quest_mask: torch.Tensor,  # (bs, n_heads, q, seq) bool, True = selected by Quest
  save_path: str = None,
  task_name: str = None,
):
  
  """
  Return the probability mass captured by Quest-selected tokens under the full attention distribution
   mass = sum_{i in selected} softmax(scores_full)[i]

  Optional: also return “average weight per selected token”:
   avg = mass / (#selected tokens)
  """

  

  attn_full = torch.softmax(scores_full, dim=-1) # (bs, n_heads, q, seq)


  mask_f = quest_mask.to(dtype=attn_full.dtype)
  mass = (attn_full * mask_f).sum(dim=-1)

  out = {}



  out["mass_mean"] = mass.mean()

  out["mass"] = mass               # (bs, n_heads, q)

  out["mass_per_head"] = mass.mean(dim=-1)

  out["mass_per_query"] = mass.mean(dim=1)

  from pathlib import Path
  import json


  if save_path is not None:
    outdir = Path(save_path)
  else:
    outdir = Path("efficiency/topkrate-results")
  outdir.mkdir(parents=True, exist_ok=True)


  if task_name is not None:
    outpath = outdir / f"{task_name}_topkrate.jsonl"
  else:
    suffix = f'chunk_size{chunk_size}-tokenbudget{token_budget}'
    outpath = outdir / f"{suffix}.jsonl"



  record = {
        "token_pos":token_pos,
        "layer": layerid,
        "k":token_budget,
        "avg_captured_mass_all": float(out["mass_mean"].item()),
        "captured_mass_per_head": out["mass_per_head"].squeeze(0).tolist(),
      }
  with _file_lock:
    with open(outpath, "a", encoding="utf-8") as f:

      json.dump(record, f, ensure_ascii=False)
      f.write("\n")

  return out



def forward(
  self,
  hidden_states: torch.Tensor,
  attention_mask: Optional[torch.Tensor] = None,
  position_ids: Optional[torch.LongTensor] = None,
  past_key_value=None,
  output_attentions: bool = False,
  use_cache: bool = False,
  cache_position: Optional[torch.LongTensor] = None,
  position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
  **kwargs,
):
  """
  Quest-patched forward for Qwen2 attention modules.

  Behavior:
   - Prefill (q_len > 1) or early layers (layer_id < 2): fall back to original attention forward.
   - Decode (q_len == 1) in later layers: apply Quest's local heavy-hitter selection on attention scores.
  """
  bsz, q_len, _ = hidden_states.size()
  if q_len > 1 or getattr(self, "layer_id", 0) < 2:
    call_kwargs = dict(
      hidden_states=hidden_states,
      attention_mask=attention_mask,
      position_ids=position_ids,
      past_key_value=past_key_value,
      output_attentions=output_attentions,
      use_cache=use_cache,
      cache_position=cache_position,
      position_embeddings=position_embeddings,
      **kwargs,
    )


    call_kwargs = _filter_kwargs_for_fn(self.flash_forward, call_kwargs)
    return self.flash_forward(**call_kwargs)
  if apply_rotary_pos_emb is None or repeat_kv is None:
    raise RuntimeError(
      "transformers.models.qwen2.modeling_qwen2 is not available in this environment. "
      "Please install/upgrade transformers with Qwen2 support."
    )


  query_states = self.q_proj(hidden_states)
  key_states = self.k_proj(hidden_states)
  value_states = self.v_proj(hidden_states)


  query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
  key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
  value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)





  position_ids = _get_position_ids(hidden_states, position_ids, cache_position, past_key_value)


  abs_pos = int(position_ids[0, -1].item())


  first_pos = getattr(self, "_first_decode_abs_pos", None)
  if first_pos is None:
    self._first_decode_abs_pos = abs_pos
    first_pos = abs_pos


  token_no = abs_pos - first_pos + 1



  kv_seq_len = key_states.shape[-2]
  if past_key_value is not None:
    if hasattr(past_key_value, "get_usable_length"):
      kv_seq_len += past_key_value.get_usable_length(kv_seq_len, getattr(self, "layer_idx", None))
    elif isinstance(past_key_value, DynamicCache) or hasattr(past_key_value, "get_seq_length"):
      try:
        past_len = past_key_value.get_seq_length()
      except TypeError:
        try:
          past_len = past_key_value.get_seq_length(getattr(self, "layer_idx", None))
        except Exception:
          past_len = 0
      kv_seq_len += int(past_len)
    elif isinstance(past_key_value, (tuple, list)) and len(past_key_value) > 0:
      kv_seq_len += past_key_value[0].shape[-2]



  cos, sin = _get_rope_cos_sin(self, value_states, position_ids, kv_seq_len, position_embeddings)
  

  query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)



  if past_key_value is not None:

    if hasattr(past_key_value, "update") and not (isinstance(past_key_value, (tuple, list))):
      cache_kwargs = {"sin": sin, "cos": cos}   
      try:
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
      except TypeError:
        try:
          key_states, value_states = past_key_value.update(
            key_states, value_states, layer_idx=self.layer_idx, cache_kwargs=cache_kwargs
          )
        except TypeError:
          key_states, value_states = past_key_value.update(key_states, value_states, layer_idx=self.layer_idx)
      if not use_cache:
        pass
    else:

      key_states = torch.cat([past_key_value[0], key_states], dim=2)
      value_states = torch.cat([past_key_value[1], value_states], dim=2)

  if isinstance(past_key_value, (tuple, list)):
    past_key_value = (key_states, value_states) if use_cache else None




  key_states = repeat_kv(key_states, self.num_key_value_groups)
  value_states = repeat_kv(value_states, self.num_key_value_groups)

  sign = torch.where(
    query_states > 0,
    torch.ones_like(query_states),
    -torch.ones_like(query_states),
  )
  signed_key_states = key_states * sign
  positive_query = query_states * sign

  seq_length = signed_key_states.shape[-2]
  padding_length = (self.chunk_size - seq_length % self.chunk_size) % self.chunk_size

  if padding_length > 0:
    pad_key = torch.full(
      (
        signed_key_states.shape[0],
        signed_key_states.shape[1],
        padding_length,
        signed_key_states.shape[3],
      ),
      torch.finfo(signed_key_states.dtype).min,
      device=signed_key_states.device,
      dtype=signed_key_states.dtype,
    )
    signed_key_states_pad = torch.cat([signed_key_states, pad_key], dim=-2)
  else:
    signed_key_states_pad = signed_key_states

  num_chunks = signed_key_states_pad.shape[-2] // self.chunk_size
  chunk_max_key = signed_key_states_pad.reshape(
    bsz,
    self.num_heads,
    num_chunks,
    self.chunk_size,
    self.head_dim,
  ).amax(dim=-2)

  chunk_scores = torch.matmul(
    positive_query.float(),
    chunk_max_key.transpose(2, 3).float(),
  )

  if attention_mask is not None:
    expanded_mask = attention_mask.expand(
      bsz,
      self.num_heads,
      q_len,
      seq_length,
    )

    if padding_length > 0:
      pad_mask = torch.full(
        (
          bsz,
          self.num_heads,
          q_len,
          padding_length,
        ),
        torch.finfo(expanded_mask.dtype).min,
        device=expanded_mask.device,
        dtype=expanded_mask.dtype,
      )
      expanded_mask_pad = torch.cat([expanded_mask, pad_mask], dim=-1)
    else:
      expanded_mask_pad = expanded_mask

    chunk_mask = expanded_mask_pad.reshape(
      bsz,
      self.num_heads,
      q_len,
      num_chunks,
      self.chunk_size,
    ).amax(dim=-1)
    chunk_scores = chunk_scores + chunk_mask.float()

  token_budget = int(getattr(self, "token_budget", 0))
  if token_budget <= 0:
    token_budget = seq_length
  token_budget = min(token_budget, seq_length)

  #print(token_budget)
  selected_token_idx, selected_token_valid = quest_select_token_indices_from_chunks(
    chunk_scores=chunk_scores,
    token_budget=token_budget,
    chunk_size=self.chunk_size,
    seq_length=seq_length,
  )

  global layerID
  layerid=layerID
  chunk_size=self.chunk_size

  layerID+=1
  if layerID==29:
    layerID=1


  # )


  selected_token_idx_2d = selected_token_idx.squeeze(2)
  gather_index = selected_token_idx_2d.unsqueeze(-1).expand(
    -1,
    -1,
    -1,
    self.head_dim,
  )

  selected_key_states = torch.gather(
    key_states,
    dim=2,
    index=gather_index,
  )
  selected_value_states = torch.gather(
    value_states,
    dim=2,
    index=gather_index,
  )

  attn_weights = torch.matmul(
    query_states,
    selected_key_states.transpose(2, 3),
  ) / math.sqrt(self.head_dim)

  min_value = torch.finfo(attn_weights.dtype).min
  attn_weights = attn_weights.masked_fill(~selected_token_valid, min_value)

  if attention_mask is not None:
    expanded_mask = attention_mask.expand(
      bsz,
      self.num_heads,
      q_len,
      seq_length,
    )
    selected_attn_mask = torch.gather(
      expanded_mask,
      dim=-1,
      index=selected_token_idx,
    )
    attn_weights = attn_weights + selected_attn_mask
    attn_weights = torch.max(
      attn_weights,
      torch.tensor(min_value, device=attn_weights.device, dtype=attn_weights.dtype),
    )

  attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

  # =========================

  # =========================
  if hasattr(self, "save_path") and self.save_path is not None:
    with torch.no_grad():
      quest_mask = torch.zeros(
        bsz, self.num_heads, 1, seq_length,
        device=selected_token_idx.device, dtype=torch.bool,
      )
      quest_mask.scatter_(-1, selected_token_idx, True)

      full_scores = torch.matmul(
        query_states, key_states.transpose(2, 3)
      ) / math.sqrt(self.head_dim)

      if attention_mask is not None:
        _exp_mask = attention_mask.expand(bsz, self.num_heads, q_len, seq_length)
        full_scores = full_scores + _exp_mask
        full_scores = torch.max(
          full_scores,
          torch.tensor(torch.finfo(full_scores.dtype).min,
                 device=full_scores.device, dtype=full_scores.dtype),
        )

      _task = getattr(self, "task", None)
      calculate_recall_from_mask(
        token_pos=int(seq_length),
        layerid=self.layer_id,
        chunk_size=self.chunk_size,
        token_budget=token_budget,
        scores_full=full_scores,
        quest_mask=quest_mask,
        save_path=self.save_path,
        task_name=_task,
      )

      calculate_topkrate(
        token_pos=int(seq_length),
        layerid=self.layer_id,
        chunk_size=self.chunk_size,
        token_budget=token_budget,
        scores_full=full_scores,
        quest_mask=quest_mask,
        save_path=self.save_path,
        task_name=_task,
      )

  attn_output = torch.matmul(attn_weights, selected_value_states)

  attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
  attn_output = self.o_proj(attn_output)

  if not output_attentions:
    attn_weights = None

  return attn_output, attn_weights, past_key_value

global layer_id
layer_id = 28



def enable_quest_qwen_attention_eval(model, args):
  if Qwen2Attention is None:
    raise ImportError(
      "Failed to import Qwen2 attention classes from transformers. "
      "Please upgrade transformers. Original error: %r" % (_IMPORT_ERROR,)
    )

  n_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
  if n_layers is None:
    n_layers = getattr(getattr(model, "config", None), "n_layer", None)
  if not isinstance(n_layers, int) or n_layers <= 0:
    n_layers = 28 # fallback

  next_layer_id = n_layers

  def _patch_module(m):
    nonlocal next_layer_id

    for name, child in reversed(m._modules.items()):
      if len(list(child.children())) > 0:
        _patch_module(child)

      if isinstance(child, (Qwen2Attention, Qwen2FlashAttention2, Qwen2SdpaAttention)):
        if not hasattr(child, "flash_forward"):
          child.flash_forward = child.forward

        child.forward = types.MethodType(forward, child)

        next_layer_id -= 1
        child.layer_id = next_layer_id
        child.token_budget = int(getattr(args, "token_budget", 0))
        child.chunk_size = int(getattr(args, "chunk_size", 16))
        child.save_path = getattr(args, "save_path", None)
        child.task = getattr(args, "task", None)

  _patch_module(model)
  return model

enable_quest_attention_eval = enable_quest_qwen_attention_eval
