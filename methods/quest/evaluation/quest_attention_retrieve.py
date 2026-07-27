import math
import numpy as np
from contextlib import nullcontext
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast

import types

try:
  from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
  flash_attn_func = None
  flash_attn_varlen_func = None

from transformers.models.llama.modeling_llama import (
  LlamaAttention,
  apply_rotary_pos_emb,
  repeat_kv,
)




from transformers.cache_utils import DynamicCache
from transformers.models.mistral.modeling_mistral import MistralAttention

_QUEST_BREAKDOWN_TIME_MANAGER = None


def _breakdown_measure(component: str, *, pre_attn: bool = False):
  timing_manager = _QUEST_BREAKDOWN_TIME_MANAGER
  if timing_manager is None:
    return nullcontext()
  if pre_attn and hasattr(timing_manager, "measure_pre_attn_component"):
    return timing_manager.measure_pre_attn_component(component)
  return timing_manager.measure(component)


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


def quest_select_token_indices_from_scores(attn_weights, token_budget, chunk_size):
  seq_length = attn_weights.shape[-1]
  padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)
  if padding_length > 0:
    padding = torch.full(
      (*attn_weights.shape[:-1], padding_length),
      torch.finfo(attn_weights.dtype).min,
      device=attn_weights.device,
      dtype=attn_weights.dtype,
    )
    attn_weights = torch.cat([attn_weights, padding], dim=-1)

  chunk_scores = attn_weights.reshape(
    *attn_weights.shape[:-1],
    attn_weights.shape[-1] // chunk_size,
    chunk_size,
  ).amax(dim=-1)

  num_chunks = chunk_scores.size(-1)
  num_select_chunks = min(max(3, math.ceil(token_budget / chunk_size)), num_chunks)
  top_chunks = chunk_scores.topk(k=num_select_chunks, dim=-1).indices
  offsets = torch.arange(chunk_size, device=top_chunks.device, dtype=top_chunks.dtype)
  selected_token_idx = top_chunks.unsqueeze(-1) * chunk_size + offsets
  selected_token_idx = selected_token_idx.reshape(
    *top_chunks.shape[:-1],
    num_select_chunks * chunk_size,
  )
  selected_token_valid = selected_token_idx < seq_length
  selected_token_idx = selected_token_idx.clamp(max=seq_length - 1)
  return selected_token_idx, selected_token_valid


def _selected_attention_fallback(
  query_states,
  selected_key_states,
  selected_value_states,
  selected_token_idx,
  selected_token_valid,
  kv_seq_len,
  output_attentions,
):
  attn_weights = torch.matmul(
    query_states,
    selected_key_states.transpose(2, 3),
  ) / math.sqrt(query_states.shape[-1])

  min_value = torch.finfo(attn_weights.dtype).min
  attn_weights = attn_weights.masked_fill(~selected_token_valid, min_value)
  attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
    query_states.dtype
  )
  attn_output = torch.matmul(attn_weights, selected_value_states)

  if output_attentions:
    full_attn_weights = torch.zeros(
      *attn_weights.shape[:-1],
      kv_seq_len,
      device=attn_weights.device,
      dtype=attn_weights.dtype,
    )
    valid_attn_weights = attn_weights * selected_token_valid.to(attn_weights.dtype)
    attn_weights = full_attn_weights.scatter_add_(
      -1,
      selected_token_idx,
      valid_attn_weights,
    )
  else:
    attn_weights = None

  return attn_output, attn_weights


def _flash_attention_selected_tokens(
  query_states,
  selected_key_states,
  selected_value_states,
  selected_token_valid,
):
  if query_states.device.type != "cuda":
    return None
  if query_states.dtype not in (torch.float16, torch.bfloat16):
    return None

  bsz, num_heads, q_len, head_dim = query_states.shape
  if q_len != 1:
    return None

  softmax_scale = 1.0 / math.sqrt(head_dim)
  selected_token_valid = selected_token_valid.squeeze(2)
  all_tokens_valid = bool(selected_token_valid.all().item())

  if all_tokens_valid:
    if flash_attn_func is None:
      return None
    with _breakdown_measure("attn"):
      attn_output = flash_attn_func(
        query_states.transpose(1, 2).contiguous(),
        selected_key_states.transpose(1, 2).contiguous(),
        selected_value_states.transpose(1, 2).contiguous(),
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
      )
    return attn_output.transpose(1, 2).contiguous()

  if flash_attn_varlen_func is None:
    return None

  selected_len = selected_key_states.shape[2]
  batch_heads = bsz * num_heads
  valid_counts = selected_token_valid.sum(dim=-1).reshape(-1)
  if bool((valid_counts == 0).any().item()):
    return None

  flat_valid = selected_token_valid.reshape(batch_heads, selected_len)
  flat_key_states = selected_key_states.reshape(batch_heads, selected_len, head_dim)
  flat_value_states = selected_value_states.reshape(batch_heads, selected_len, head_dim)

  query_varlen = query_states.reshape(batch_heads * q_len, 1, head_dim).contiguous()
  key_varlen = flat_key_states[flat_valid].unsqueeze(1).contiguous()
  value_varlen = flat_value_states[flat_valid].unsqueeze(1).contiguous()

  cu_seqlens_q = torch.arange(
    batch_heads * q_len + 1,
    device=query_states.device,
    dtype=torch.int32,
  )
  cu_seqlens_k = torch.empty(batch_heads + 1, device=query_states.device, dtype=torch.int32)
  cu_seqlens_k[0] = 0
  cu_seqlens_k[1:] = torch.cumsum(valid_counts.to(torch.int32), dim=0)

  with _breakdown_measure("attn"):
    attn_output = flash_attn_varlen_func(
      query_varlen,
      key_varlen,
      value_varlen,
      cu_seqlens_q=cu_seqlens_q,
      cu_seqlens_k=cu_seqlens_k,
      max_seqlen_q=1,
      max_seqlen_k=int(valid_counts.max().item()),
      dropout_p=0.0,
      softmax_scale=softmax_scale,
      causal=False,
    )
  return attn_output.reshape(bsz, num_heads, q_len, head_dim).contiguous()



def calculate_recall_from_mask(
  token_pos:int,
  layerid:int,
  chunk_size:int,
  token_budget: int,    #q=1  nheads=num of query head
  scores_full: torch.Tensor,
  quest_mask: torch.Tensor,
):
  

  seq = scores_full.shape[-1]
  k_eff = min(token_budget, seq)

  gt_idx = scores_full.topk(k_eff, dim=-1).indices




  hit = quest_mask.gather(-1, gt_idx)     # (bs, n_heads, q, k_eff) bool

  recall = hit.float().mean()


  from pathlib import Path
  import json

  model_name="Llama"
  suffix=f'{model_name}-chunk_size{chunk_size}-tokenbudget{token_budget}'
  outdir = Path("efficiency/recall-results")
  outdir.mkdir(parents=True, exist_ok=True)

  outpath = outdir / f"{suffix}.jsonl"
  outpath.parent.mkdir(parents=True, exist_ok=True)

  record = {
        "token_pos":token_pos,
        "layer": layerid,
        "k":token_budget,
        "recall": float(recall),
      }
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

  model_name="Llama"
  suffix=f'{model_name}-chunk_size{chunk_size}-tokenbudget{token_budget}'
  outdir = Path("efficiency/topkrate-results")
  outdir.mkdir(parents=True, exist_ok=True)

  outpath = outdir / f"{suffix}.jsonl"
  outpath.parent.mkdir(parents=True, exist_ok=True)
  record = {
        "token_pos":token_pos,
        "layer": layerid,
        "k":token_budget,
        "avg_captured_mass_all": float(out["mass_mean"].item()),
      }
  with open(outpath, "a", encoding="utf-8") as f:

    json.dump(record, f, ensure_ascii=False)
    f.write("\n")

  return out




def forward(
  self,
  hidden_states: torch.Tensor,
  attention_mask: Optional[torch.Tensor] = None,
  position_ids: Optional[torch.LongTensor] = None,
  past_key_value: Optional[Tuple[torch.Tensor]] = None,
  output_attentions: bool = False,
  use_cache: bool = False,
  **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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
  

  abs_pos = int(position_ids[0, -1].item())


  first_pos = getattr(self, "_first_decode_abs_pos", None)
  if first_pos is None:
    self._first_decode_abs_pos = abs_pos
    first_pos = abs_pos


  token_no = abs_pos - first_pos + 1


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
  

  if isinstance(past_key_value, DynamicCache):
    kv_seq_len = past_key_value.get_seq_length()
  else:
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
      assert isinstance(past_key_value, tuple)
      kv_seq_len += past_key_value[0].shape[-2]
  

  cos, sin = self.rotary_emb(value_states, position_ids.to(value_states.device))
  query_states, key_states = apply_rotary_pos_emb(
    query_states, key_states, cos, sin, position_ids
  )


  if isinstance(past_key_value, DynamicCache):
    if use_cache:
      with _breakdown_measure("write_cache"):
        key_states, value_states = past_key_value.update(key_states, value_states, layer_idx=self.layer_idx)
  else:
    if use_cache:
      with _breakdown_measure("write_cache"):
        if past_key_value is not None:
          key_states = torch.cat([past_key_value[0], key_states], dim=2)
          value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states)
    else:
      if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
      past_key_value = None


  key_states = repeat_kv(key_states, self.num_key_value_groups)
  value_states = repeat_kv(value_states, self.num_key_value_groups)


  with _breakdown_measure("index_build"):


    sign = (query_states > 0) + (~(query_states > 0)) * -1
    max_key = key_states * sign
    postive_query = query_states * sign


    seq_length = max_key.shape[-2]
    padding_length = self.chunk_size - ((seq_length - 1) % self.chunk_size + 1)

    max_key = torch.cat(
      [
        max_key,  
        torch.ones(
          (max_key.shape[0], max_key.shape[1], padding_length, max_key.shape[3]),
          device=max_key.device,
        )
        * torch.tensor(torch.finfo(max_key.dtype).min),
      ],
      dim=-2,
    )



    chunk_max_key = max_key.reshape(
      max_key.shape[0],
      max_key.shape[1],
      max_key.shape[2] // self.chunk_size,
      self.chunk_size,
      max_key.shape[3],
    ).amax(dim=-2)


    chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, self.chunk_size, 1)
    chunk_max_key = chunk_max_key.reshape(
      chunk_max_key.shape[0], chunk_max_key.shape[1], -1, chunk_max_key.shape[-1]
    )[:, :, :seq_length, :]

  with _breakdown_measure("retrieve"):

    quantized_weight = torch.matmul(
      postive_query.float(),
      chunk_max_key.transpose(2, 3),
    )

  if quantized_weight.size() != (bsz, self.num_heads, q_len, kv_seq_len):
    raise ValueError(
      f"Quantized attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
      f" {quantized_weight.size()}"
    )


  if attention_mask is not None:
    if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
      raise ValueError(
        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
      )
    with _breakdown_measure("retrieve"):
      quantized_weight = quantized_weight + attention_mask
      min_value = torch.tensor(
        torch.finfo(quantized_weight.dtype).min,
        device=quantized_weight.device,
        dtype=quantized_weight.dtype,
      )
      quantized_weight = torch.max(
        quantized_weight,
        min_value,
      )
    
  with _breakdown_measure("retrieve"):

    token_budget = min(kv_seq_len, self.token_budget)

    if token_budget > 0:
      selected_token_idx, selected_token_valid = quest_select_token_indices_from_scores(
        quantized_weight,
        token_budget,
        self.chunk_size,
      )
    else:
      selected_token_idx = torch.zeros(
        bsz,
        self.num_heads,
        q_len,
        1,
        device=query_states.device,
        dtype=torch.long,
      )
      selected_token_valid = torch.zeros_like(selected_token_idx, dtype=torch.bool)

    if position_ids is not None:
      causal_limit = position_ids[:, None, :, None].to(selected_token_idx.device)
      selected_token_valid = selected_token_valid & (selected_token_idx <= causal_limit)

    if attention_mask is not None:
      expanded_mask = attention_mask.expand(bsz, self.num_heads, q_len, kv_seq_len)
      selected_attn_mask = torch.gather(
        expanded_mask,
        dim=-1,
        index=selected_token_idx,
      )
      if selected_attn_mask.dtype == torch.bool:
        selected_token_valid = selected_token_valid & selected_attn_mask
      else:
        selected_token_valid = selected_token_valid & (
          selected_attn_mask > torch.finfo(selected_attn_mask.dtype).min / 2
        )
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

  global layerID
  layerid=layerID
  chunk_size=self.chunk_size

  layerID+=1
  if layerID==33:
    layerID=1


  # )

  attn_weights = None
  attn_output = None
  if not output_attentions:
    attn_output = _flash_attention_selected_tokens(
      query_states,
      selected_key_states,
      selected_value_states,
      selected_token_valid,
    )
    #   )



  if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
    raise ValueError(
      f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
      f" {attn_output.size()}"
    )

  attn_output = attn_output.transpose(1, 2)
  attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
  attn_output = self.o_proj(attn_output)

  if not output_attentions:
    attn_weights = None

  return attn_output, attn_weights, past_key_value


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
