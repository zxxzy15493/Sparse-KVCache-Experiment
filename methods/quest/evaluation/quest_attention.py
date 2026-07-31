import math
import numpy as np
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast

import types
import threading

from transformers.models.llama.modeling_llama import (
  LlamaAttention,
  apply_rotary_pos_emb,
  repeat_kv,
)




from transformers.cache_utils import DynamicCache
from transformers.models.mistral.modeling_mistral import MistralAttention


_file_lock = threading.Lock()

layerID=1


def local_heavy_hitter_mask(attn_weights, token_budget, chunk_size):


  seq_length = attn_weights.shape[-1]
  padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)

  attn_weights = torch.cat(  #
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
      )
      * torch.tensor(torch.finfo(attn_weights.dtype).min),
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


  _, topk = chunk_attn_weights.topk(
    k=min(max(3, math.ceil(token_budget / chunk_size)), chunk_attn_weights.size(-1)), dim=-1
  )
  topk = topk.unsqueeze(-1).repeat(
    1, 1, 1, 1, chunk_size
  ) * chunk_size + torch.arange(chunk_size, device=topk.device)
  topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
  
  mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
  mask_bottom.scatter_(-1, topk, True)
 
  mask_bottom = mask_bottom[:, :, :, :seq_length]

  return mask_bottom



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

  recall_per_head = hit.float().mean(dim=-1).squeeze(0).tolist() # (n_heads,)


  k_100 = min(100, seq)
  gt_idx_100 = scores_full.topk(k_100, dim=-1).indices
  hit_100 = quest_mask.gather(-1, gt_idx_100)
  recall_100 = float(hit_100.float().mean())
  recall_100_per_head = hit_100.float().mean(dim=-1).squeeze(0).tolist()

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
  ).indices # (bsz, num_heads, q_len, num_select_chunks)

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



def forward(
  self,
  hidden_states: torch.Tensor,
  attention_mask: Optional[torch.Tensor] = None,
  position_ids: Optional[torch.LongTensor] = None,
  past_key_value: Optional[Tuple[torch.Tensor]] = None,
  output_attentions: bool = False,
  use_cache: bool = False,
  **kwargs,
):
  bsz, q_len, _ = hidden_states.size()




  if q_len > 1 or self.layer_id < 2:
    return self.flash_forward(
      hidden_states,
      attention_mask,
      position_ids,
      past_key_value,
      output_attentions,
      use_cache,
      **kwargs,
    )
  query_states = (
    self.q_proj(hidden_states)
    .view(bsz, q_len, self.num_heads, self.head_dim)
    .transpose(1, 2)
  )

  key_states = (
    self.k_proj(hidden_states)
    .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    .transpose(1, 2)
  )

  value_states = (
    self.v_proj(hidden_states)
    .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    .transpose(1, 2)
  )

  
  # =========================
  # =========================
  cos, sin = self.rotary_emb(
    value_states,
    position_ids.to(value_states.device),
  )

  query_states, key_states = apply_rotary_pos_emb(
    query_states,
    key_states,
    cos,
    sin,
    position_ids,
  )

  # =========================
  # =========================
  if isinstance(past_key_value, DynamicCache):
    if use_cache:
      key_states, value_states = past_key_value.update(
        key_states,
        value_states,
        layer_idx=self.layer_idx,
      )
  else:
    if past_key_value is not None:
      key_states = torch.cat([past_key_value[0], key_states], dim=2)
      value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

  key_states = repeat_kv(key_states, self.num_key_value_groups)
  value_states = repeat_kv(value_states, self.num_key_value_groups)

  kv_seq_len = key_states.shape[-2]

  if attention_mask is not None:
    if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
      raise ValueError(
        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
        f"but is {attention_mask.size()}"
      )

  token_budget = min(kv_seq_len, self.token_budget)

  #print(token_budget)
  sign = torch.where(
    query_states > 0,
    torch.ones_like(query_states),
    -torch.ones_like(query_states),
  )

  positive_query = query_states * sign



  signed_key_states = key_states * sign

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

    signed_key_states_pad = torch.cat(
      [signed_key_states, pad_key],
      dim=-2,
    )
  else:
    signed_key_states_pad = signed_key_states

  num_chunks = signed_key_states_pad.shape[-2] // self.chunk_size

  chunk_key = signed_key_states_pad.reshape(
    bsz,
    self.num_heads,
    num_chunks,
    self.chunk_size,
    self.head_dim,
  )


  chunk_max_key = chunk_key.amax(dim=-2)



  chunk_scores = torch.matmul(
    positive_query.float(),
    chunk_max_key.transpose(2, 3).float(),
  )



  if attention_mask is not None:
    expanded_mask = attention_mask.expand(
      bsz,
      self.num_heads,
      q_len,
      kv_seq_len,
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

      expanded_mask_pad = torch.cat(
        [expanded_mask, pad_mask],
        dim=-1,
      )
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

  # =========================

  # =========================
  selected_token_idx, selected_token_valid = quest_select_token_indices_from_chunks(
    chunk_scores=chunk_scores,
    token_budget=token_budget,
    chunk_size=self.chunk_size,
    seq_length=kv_seq_len,
  )

  # =========================

  # =========================
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

  # =========================

  # =========================
  attn_weights = torch.matmul(
    query_states,
    selected_key_states.transpose(2, 3),
  ) / math.sqrt(self.head_dim)

  min_value = torch.finfo(attn_weights.dtype).min


  attn_weights = attn_weights.masked_fill(
    ~selected_token_valid,
    min_value,
  )


  if attention_mask is not None:
    expanded_mask = attention_mask.expand(
      bsz,
      self.num_heads,
      q_len,
      kv_seq_len,
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

  attn_weights = nn.functional.softmax(
    attn_weights,
    dim=-1,
    dtype=torch.float32,
  ).to(query_states.dtype)

  # =========================

  # =========================
  if hasattr(self, "save_path") and self.save_path is not None:
    with torch.no_grad():
      quest_mask = torch.zeros(
        bsz, self.num_heads, 1, kv_seq_len,
        device=selected_token_idx.device, dtype=torch.bool,
      )
      quest_mask.scatter_(-1, selected_token_idx, True)

      full_scores = torch.matmul(
        query_states, key_states.transpose(2, 3)
      ) / math.sqrt(self.head_dim)

      if attention_mask is not None:
        _exp_mask = attention_mask.expand(bsz, self.num_heads, q_len, kv_seq_len)
        full_scores = full_scores + _exp_mask
        full_scores = torch.max(
          full_scores,
          torch.tensor(torch.finfo(full_scores.dtype).min,
                 device=full_scores.device, dtype=full_scores.dtype),
        )

      _task = getattr(self, "task", None)
      calculate_recall_from_mask(
        token_pos=int(kv_seq_len),
        layerid=self.layer_id,
        chunk_size=self.chunk_size,
        token_budget=token_budget,
        scores_full=full_scores,
        quest_mask=quest_mask,
        save_path=self.save_path,
        task_name=_task,
      )

      calculate_topkrate(
        token_pos=int(kv_seq_len),
        layerid=self.layer_id,
        chunk_size=self.chunk_size,
        token_budget=token_budget,
        scores_full=full_scores,
        quest_mask=quest_mask,
        save_path=self.save_path,
        task_name=_task,
      )

  attn_output = torch.matmul(
    attn_weights,
    selected_value_states,
  )

  if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
    raise ValueError(
      f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, "
      f"but is {attn_output.size()}"
    )

  attn_output = attn_output.transpose(1, 2).contiguous()
  attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
  attn_output = self.o_proj(attn_output)



  if output_attentions:
    full_attn_weights = torch.zeros(
      bsz,
      self.num_heads,
      q_len,
      kv_seq_len,
      device=attn_weights.device,
      dtype=attn_weights.dtype,
    )

    full_attn_weights.scatter_(
      dim=-1,
      index=selected_token_idx,
      src=attn_weights,
    )

    attn_weights_to_return = full_attn_weights
  else:
    attn_weights_to_return = None

  return attn_output, attn_weights_to_return, past_key_value


global layer_id
layer_id = 32


def enable_quest_attention_eval(model, args):
  for name, module in reversed(model._modules.items()):
    if len(list(module.children())) > 0:
      enable_quest_attention_eval(
        module,
        args,
      )

    global layer_id
    if isinstance(module, (LlamaAttention, MistralAttention)):
      layer_id -= 1
      model._modules[name].layer_id = layer_id

      model._modules[name].flash_forward = model._modules[name].forward
      model._modules[name].forward = types.MethodType(
        forward, model._modules[name]
      )

      model._modules[name].token_budget = args.token_budget
      model._modules[name].chunk_size = args.chunk_size
      model._modules[name].save_path = getattr(args, "save_path", None)
      model._modules[name].task = getattr(args, "task", None)
