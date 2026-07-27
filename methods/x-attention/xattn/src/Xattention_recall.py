from xattn.src.utils import *
import torch
import math
import torch.nn.functional as F
from xattn.src.kernels import (
  flat_group_gemm,
  softmax_fuse_block_sum,
  flat_group_gemm_fuse_reshape,
)

import sys
import os
from pathlib import Path
sys.path.insert(0, ".")
from block_sparse_attn import block_sparse_attn_func

blocks_num=[]


def xattn_estimate(
  query_states: torch.Tensor,
  key_states: torch.Tensor,
  block_size,
  stride,
  norm=1,
  softmax=True,
  threshold=0.9,
  chunk_size=16384,
  select_mode="inverse",
  use_triton=True,
  causal=True,
  kdb: int = 1,
  keep_sink=False,
  keep_recent=False,
) -> torch.Tensor:
  batch_size, num_kv_head, k_len, head_dim = key_states.shape
  batch_size, num_q_head, q_len, head_dim = query_states.shape
  assert num_q_head == num_kv_head

  k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
  q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
  k_chunk_num = (k_len + k_num_to_pad) // chunk_size
  k_block_num = (k_len + k_num_to_pad) // block_size
  q_chunk_num = (q_len + q_num_to_pad) // chunk_size
  q_block_num = (q_len + q_num_to_pad) // block_size
  assert k_chunk_num >= q_chunk_num
  offset_token_chunk_num = k_chunk_num - q_chunk_num

  if k_num_to_pad > 0:
    pad_key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad), value=0).to("cuda")
  else:
    pad_key_states = key_states
  if q_num_to_pad > 0:
    pad_query_states = F.pad(query_states, (0, 0, 0, q_num_to_pad), value=0).to(
      "cuda"
    )
  else:
    pad_query_states = query_states

  assert num_kv_head == num_q_head
  attn_sum_list = []
  simple_mask_list = []
  
  reshaped_chunk_size = chunk_size // stride

  reshaped_block_size = block_size // stride
  k_reshaped_num_to_pad = k_num_to_pad // stride
  k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
  q_reshaped_num_to_pad = q_num_to_pad // stride
  num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size
  if not use_triton:
    if select_mode == "random":
      perm_idx = torch.randperm(stride)
      reshaped_key = torch.cat(
        [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
      )
      reshaped_query = torch.cat(
        [
          pad_query_states[:, :, perm_idx[i] :: stride, :]
          for i in range(stride)
        ],
        dim=-1,
      )
    elif select_mode == "inverse" or select_mode == "":
      reshaped_key = torch.cat(     #reshaped_key (B, H, k_len/stride, D×stride)
        [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
      )
      reshaped_query = torch.cat(
        [
          (pad_query_states[:, :, (stride - 1 - q) :: (stride * kdb), :])
          for q in range(stride)
        ],
        dim=-1,
      )
    elif select_mode == "slash":
      reshaped_key = torch.cat(
        [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
      )
      reshaped_query = torch.cat(
        [(pad_query_states[:, :, q::stride, :]) for q in range(stride)], dim=-1
      )
    elif select_mode == "double":
      reshaped_key = torch.cat(
        [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
      )
      reshaped_key = reshaped_key + torch.cat(
        [reshaped_key[:, :, :, head_dim:], reshaped_key[:, :, :, 0:head_dim]],
        dim=-1,
      )
      reshaped_query = torch.cat(
        [
          (pad_query_states[:, :, (stride - 1 - q) :: stride, :])
          for q in range(stride)
        ],
        dim=-1,
      )
    elif select_mode == "triple":
      reshaped_key = torch.cat(
        [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
      )
      reshaped_key = reshaped_key + torch.cat(
        [reshaped_key[:, :, :, head_dim:], reshaped_key[:, :, :, 0:head_dim]],
        dim=-1,
      )
      reshaped_key = reshaped_key + torch.cat(
        [reshaped_key[:, :, :, -head_dim:], reshaped_key[:, :, :, 0:-head_dim]],
        dim=-1,
      )
      reshaped_query = torch.cat(
        [
          (pad_query_states[:, :, (stride - 1 - q) :: stride, :])
          for q in range(stride)
        ],
        dim=-1,
      )
    assert reshaped_key.shape[-2] == k_reshaped_seq_len

  for chunk_idx in range(q_chunk_num):

    if use_triton:
      if kdb != 1:
        raise ValueError("use_triton and kdb cannot be used together")
      attn_weights_slice = flat_group_gemm_fuse_reshape(
        pad_query_states[
          :,
          :,
          (chunk_idx * reshaped_chunk_size)
          * stride : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size)
          * stride,
          :,
        ],
        pad_key_states,
        stride,
        (k_block_num - q_block_num) * reshaped_block_size
        + chunk_idx * reshaped_chunk_size,
        (k_block_num - q_block_num) * reshaped_block_size
        + chunk_idx * reshaped_chunk_size
        + reshaped_chunk_size,
        is_causal=causal,
      )
      attn_sum = softmax_fuse_block_sum(
        attn_weights_slice,
        reshaped_block_size,
        min(4096, reshaped_block_size),
        (k_block_num - q_block_num) * reshaped_block_size
        + chunk_idx * reshaped_chunk_size,
        (k_block_num - q_block_num) * reshaped_block_size
        + chunk_idx * reshaped_chunk_size
        + reshaped_chunk_size,
        k_reshaped_seq_len - k_reshaped_num_to_pad,
        1.4426950408889634 / math.sqrt(head_dim) / stride / norm,
        is_causal=causal,
      )
    else:
      chunked_query = reshaped_query[#  chunked_query: (B, H, reshaped_chunk_size/kdb, D×stride)
        :,             #reshaped_key^T: (B, H, D×stride, k_reshaped_seq_len)
        :,             #
        (chunk_idx * reshaped_chunk_size)
        // kdb : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size)
        // kdb,
        :,
      ]      #attn_weights_slice: (B, H, reshaped_chunk_size/kdb, k_reshaped_seq_len)

      attn_weights_slice = torch.matmul(
        chunked_query, 
        reshaped_key.transpose(2, 3),
      ).to("cuda")


      attn_weights_slice = (
        attn_weights_slice / math.sqrt(head_dim) / stride / norm
      )

      if causal:
        causal_mask = torch.zeros(
          (
            batch_size,
            num_q_head,
            reshaped_chunk_size,
            reshaped_chunk_size * k_chunk_num,
          ),
          device=key_states.device,
        )
        causal_mask[:, :, :, (-k_reshaped_num_to_pad):] = float("-inf")
        chunk_start = (chunk_idx + offset_token_chunk_num) * reshaped_chunk_size
        chunk_end = chunk_start + reshaped_chunk_size
        causal_mask[:, :, :, chunk_start:chunk_end] = torch.triu(
          torch.ones(
            1,
            num_q_head,
            reshaped_chunk_size,
            reshaped_chunk_size,
            device=key_states.device,
          )
          * float("-inf"),
          diagonal=1,
        )

        if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
          causal_mask[:, :, (-(q_reshaped_num_to_pad // kdb)) :, :] = float(
            "-inf"
          )

        causal_mask[:, :, :, chunk_end:] = float("-inf")
        causal_mask = causal_mask[:, :, kdb - 1 :: kdb, :]
        attn_weights_slice = attn_weights_slice + causal_mask.to(
          attn_weights_slice.device
        )

      if softmax:
        attn_weights_slice = F.softmax(
          attn_weights_slice, dim=-1, dtype=torch.float32
        ).to(pad_query_states.dtype)
      else:
        attn_weights_slice = torch.exp(attn_weights_slice).to(
          pad_query_states.dtype
        )
      attn_weights_slice = F.dropout(attn_weights_slice, p=0, training=False)

      if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
        attn_weights_slice[:, :, (-(q_reshaped_num_to_pad // kdb)) :, :] = 0

      attn_sum = (
        attn_weights_slice.view(  #attn_weights_slice  B H 512 512
          batch_size,
          num_kv_head,
          num_blocks_per_chunk,
          reshaped_block_size // kdb, 
          -1,
          reshaped_block_size,
        )
        .sum(dim=-1)
        .sum(dim=-2)
        .to("cuda")
      )
      del chunked_query
    


    simple_mask = find_blocks_chunked(
      attn_sum,
      k_block_num - q_block_num + chunk_idx * num_blocks_per_chunk,
      threshold,
      None,
      decoding=False,
      mode="prefill",
      causal=causal,
    )

    attn_sum_list.append(attn_sum)
    simple_mask_list.append(simple_mask)

    del attn_weights_slice

  if not use_triton:
    del reshaped_query, reshaped_key
  attn_sums = torch.cat(attn_sum_list, dim=-2)
  simple_masks = torch.cat(simple_mask_list, dim=-2)

  if causal:
    simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
      torch.tril(
        torch.ones(
          q_block_num, q_block_num, dtype=bool, device=key_states.device
        ),
        diagonal=0,
      ),
      simple_masks[:, :, -q_block_num:, -q_block_num:],
      False,
    )
  if keep_sink:
    simple_masks[:, :, 0, :] = True
  if keep_recent:
    eye_matrix = torch.eye(q_block_num, device=simple_masks.device, dtype=bool)
    eye_matrix_expanded = (
      eye_matrix.unsqueeze(0)
      .unsqueeze(0)
      .expand(1, num_kv_head, q_block_num, q_block_num)
    )
    simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
      eye_matrix_expanded, True, simple_masks[:, :, -q_block_num:, -q_block_num:]
    )



  return attn_sums, simple_masks


def caculate_recall(
    


):
  return 1



@torch.no_grad()
def topk_recall_from_approx_simple_mask(
  stride:int,
  model_name:str,
  layer_id:int,
  query_states: torch.Tensor,     # (B,H,q_len,D)
  key_states: torch.Tensor,      # (B,H,k_len,D)
  approx_simple_mask: torch.Tensor,  # (B,H,q_blk,k_blk) bool
  block_size: int = 128,
  causal: bool = True,
  offset: int | None = None,
  save_path: str = " ",
):
  device = query_states.device
  B, H, q_len, D = query_states.shape
  _, _, k_len, _ = key_states.shape

  q_block_num = (q_len + block_size - 1) // block_size
  k_block_num = (k_len + block_size - 1) // block_size


  if offset is None:
    offset = max(0, k_len - q_len)


  blk_mask = approx_simple_mask[:, :, :q_block_num, :k_block_num].to(torch.bool)


  recall_bhq = torch.empty((B, H, q_len), device=device, dtype=torch.float32)

  inv_sqrt_d = 1.0 / math.sqrt(D)

  for b in range(B):
    for h in range(H):

      keys_h = key_states[b, h].to(torch.float32) # (k_len, D)

      for qb in range(q_block_num):
        q_start = qb * block_size
        q_end = min((qb + 1) * block_size, q_len)
        q_len_blk = q_end - q_start
        if q_len_blk <= 0:
          continue

        q_blk = query_states[b, h, q_start:q_end].to(torch.float32) # (q_len_blk,D)



        token_pos = torch.arange(q_start, q_end, device=device) # (q_len_blk,)
        if causal:

          key_ends = (offset + token_pos).clamp(min=0, max=k_len - 1) # (q_len_blk,)

          allowed_counts = key_ends + 1

          max_end = int(key_ends.max().item())
        else:
          key_ends = None
          allowed_counts = torch.full((q_len_blk,), k_len, device=device, dtype=torch.long)
          max_end = k_len - 1



        keys_slice = keys_h[: max_end + 1] # (max_end+1, D)



        scores = (q_blk @ keys_slice.T) * inv_sqrt_d # (q_len_blk, max_end+1)
        
        if causal:
          kk = torch.arange(max_end + 1, device=device)


          scores = scores.masked_fill(kk.unsqueeze(0) > key_ends.unsqueeze(1), float("-inf"))


        sel_blocks = blk_mask[b, h, qb].nonzero(as_tuple=False).flatten() # (n_sel,)
        if sel_blocks.numel() == 0:
          recall_bhq[b, h, q_start:q_end] = 0.0
          continue



        starts = sel_blocks * block_size            # (n_sel,)
        ends = (starts + block_size - 1).clamp(max=k_len - 1)  # (n_sel,)
        if causal:

          eff_end = torch.minimum(ends.unsqueeze(0), key_ends.unsqueeze(1))  # (q_len_blk,n_sel)
        else:
          eff_end = ends.unsqueeze(0).expand(q_len_blk, -1)


        counts = (eff_end - starts.unsqueeze(0) + 1).clamp(min=0) # (q_len_blk,n_sel)

        kqs = counts.sum(dim=1).to(torch.long)           # (q_len_blk,)




        recall_blk = torch.zeros((q_len_blk,), device=device, dtype=torch.float32)



        full_cover = (kqs >= allowed_counts) 
        recall_blk[full_cover] = 1.0


        need = (kqs > 0) & (~full_cover)
        if need.any():

          max_k_need = int(kqs[need].max().item())



          topk_idx = torch.topk(scores[need], k=max_k_need, dim=-1).indices # (n_need, max_k_need)

          topk_blocks = topk_idx // block_size                # (n_need, max_k_need)


          sel_blk_mask_1d = blk_mask[b, h, qb] # (k_block_num,)


          in_sel = sel_blk_mask_1d[topk_blocks] # (n_need, max_k_need) bool


          row_k = kqs[need] # (n_need,)
          pos = torch.arange(max_k_need, device=device).unsqueeze(0) # (1,max_k_need)

          valid = pos < row_k.unsqueeze(1)              # (n_need,max_k_need)

          hits = (in_sel & valid).sum(dim=1).to(torch.float32)    # (n_need,)
          recall_blk[need] = hits / row_k.to(torch.float32)

        recall_bhq[b, h, q_start:q_end] = recall_blk



  per_query_recall = recall_bhq.mean(dim=1).mean(dim=0)  # (q_len,)
  per_head_recall = recall_bhq.mean(dim=2).mean(dim=0)

  mean_recall = per_query_recall.mean()          # scalar

  from pathlib import Path
  import json

  if save_path != " ":
    outdir = Path(save_path)
  else:
    outdir = Path("efficiency/recall")
  outdir.mkdir(parents=True, exist_ok=True)

  outpath_pre_head = outdir / f"{model_name}-stride{stride}-pre-head.jsonl"
  outpath_pre_layer = outdir / f"{model_name}-stride{stride}-layer.jsonl"

  outpath_pre_head.parent.mkdir(parents=True, exist_ok=True)
  outpath_pre_layer.parent.mkdir(parents=True, exist_ok=True)

  with open(outpath_pre_head, "a", encoding="utf-8") as f:

    for h in range(H):
      record = {
        "layer": layer_id,
        "head_num": int(h),
        "avg_recall_pre_head": float(per_head_recall[h].item()),
        "q_len": int(q_len),
      }
      json.dump(record, f, ensure_ascii=False)
      f.write("\n")

  with open(outpath_pre_layer, "a", encoding="utf-8") as f:

    record = {
      "layer": layer_id,
      "avg_recall_pre_layer": float(mean_recall.item()),
      "q_len": int(q_len),
    }
    json.dump(record, f, ensure_ascii=False)
    f.write("\n")

  return per_query_recall, mean_recall, recall_bhq




@torch.no_grad()
def selected_attn_mass_from_blockmask(
  stride,
  model_name,
  layer_id,
  query_states,     # (B,Hq,q_len,D)
  key_states,
  approx_simple_mask,  # (B,Hq,q_blk,k_blk) bool/0-1
  block_size=128,
  causal=True,
  offset=None,
  save_path=" ",
):
  device = query_states.device
  B, Hq, q_len, D = query_states.shape
  _, Hk, k_len, _ = key_states.shape
  H = Hq

  if Hk != Hq:
    assert Hq % Hk == 0
    key_states = key_states.repeat_interleave(Hq // Hk, dim=1)

  if offset is None:
    offset = max(0, k_len - q_len)
  

  q_blk = (q_len + block_size - 1) // block_size
  k_blk = (k_len + block_size - 1) // block_size
  blk_mask = (approx_simple_mask[:, :, :q_blk, :k_blk] > 0).to(torch.bool)

  inv_sqrt_d = 1.0 / math.sqrt(D)

  mass_bhq = torch.empty((B, Hq, q_len), device=device, dtype=torch.float32)
  recall100_bhq = torch.zeros((B, Hq, q_len), device=device, dtype=torch.float32)

  for b in range(B):
    for h in range(Hq):

      K = key_states[b, h].to(torch.float32) # (k_len,D)

      for qb in range(q_blk):
        qs = qb * block_size
        qe = min((qb + 1) * block_size, q_len)
        if qe <= qs:
          continue

        Q = query_states[b, h, qs:qe].to(torch.float32) # (q_len_blk,D)
        q_len_blk = qe - qs


        pos = torch.arange(qs, qe, device=device)
        if causal:

          key_ends = (offset + pos).clamp(0, k_len - 1)   # (q_len_blk,)
          max_end = int(key_ends.max().item())
        else:
          key_ends = None
          max_end = k_len - 1

        Ks = K[:max_end + 1] # (max_end+1,D)

        scores = (Q @ Ks.T) * inv_sqrt_d # (q_len_blk, max_end+1)

        if causal:
          kk = torch.arange(max_end + 1, device=device)
          scores = scores.masked_fill(kk.unsqueeze(0) > key_ends.unsqueeze(1), float("-inf"))


        attn = torch.softmax(scores, dim=-1) # (q_len_blk, max_end+1)



        sel_blocks = blk_mask[b, h, qb].nonzero(as_tuple=False).flatten()
        if sel_blocks.numel() == 0:
          mass_bhq[b, h, qs:qe] = 0.0
          continue
        

        sel_cols = torch.zeros((max_end + 1,), device=device, dtype=torch.bool)
        for blk in sel_blocks.tolist():
          s = blk * block_size
          e = min((blk + 1) * block_size, max_end + 1)
          if s < e:
            sel_cols[s:e] = True
        


        mass = attn[:, sel_cols].sum(dim=-1) # (q_len_blk,)
        mass_bhq[b, h, qs:qe] = mass


        recall100_blk = torch.zeros((q_len_blk,), device=device, dtype=torch.float32)
        recall100_need = pos >= 100
        if recall100_need.any():
          if causal:
            topk_counts = torch.minimum(key_ends[recall100_need] + 1, torch.full_like(key_ends[recall100_need], 100))
          else:
            topk_counts = torch.full((int(recall100_need.sum().item()),), min(100, k_len), device=device, dtype=torch.long)

          max_topk = int(topk_counts.max().item())
          if max_topk > 0:
            topk_idx = torch.topk(scores[recall100_need], k=max_topk, dim=-1).indices
            topk_blocks = topk_idx // block_size
            in_sel = blk_mask[b, h, qb][topk_blocks]
            topk_pos = torch.arange(max_topk, device=device).unsqueeze(0)
            valid_topk = topk_pos < topk_counts.unsqueeze(1)
            hits = (in_sel & valid_topk).sum(dim=1).to(torch.float32)
            recall100_blk[recall100_need] = hits / topk_counts.to(torch.float32)
        recall100_bhq[b, h, qs:qe] = recall100_blk
  

  per_query_mass = mass_bhq.mean(dim=1).mean(dim=0) # (q_len,)
  mean_mass = per_query_mass.mean()         # scalar
  if q_len > 100:
    recall100_per_head = recall100_bhq[:, :, 100:].mean(dim=(0, 2))
    recall100 = recall100_per_head.mean()
  else:
    recall100_per_head = torch.zeros((Hq,), device=device, dtype=torch.float32)
    recall100 = torch.tensor(0.0, device=device, dtype=torch.float32)
  from pathlib import Path
  import json

  if save_path != " ":
    path = Path(f"{save_path}.jsonl")
  else:
    path = Path("efficiency/recall") / f"{model_name}-stride{stride}-attn-mass.jsonl"
  path.parent.mkdir(parents=True, exist_ok=True)

  attn_score_per_head = mass_bhq.mean(dim=(0, 2)).detach().cpu().tolist() # (H,)

  with open(path, "a", encoding="utf-8") as f:
    record = {
      "layer": int(layer_id),
      "attn_score_per_head": attn_score_per_head,
      "mean_attn_mass": float(mean_mass.item()),
      "recall100": float(recall100.item()),
      "recall100_per_head": [float(recall100_per_head[h].item()) for h in range(H)],
      "q_len": int(q_len),
      "num_heads": int(H),
    }
    json.dump(record, f, ensure_ascii=False)
    f.write("\n")



  return per_query_mass, mean_mass, mass_bhq






# )

def record_block(model_name,task,k_len,block_size):
  
  avg=sum(blocks_num)/len(blocks_num)
  
  block_num=math.ceil(k_len/block_size)

  blocks = block_num * (block_num + 1) // 2


  from pathlib import Path
  import json

  outdir = Path("efficiency/blocks")
  outdir.mkdir(parents=True, exist_ok=True)

  outpath_pre_head = outdir / f"{model_name}_{task}_block_num.jsonl"
  outpath_pre_head.parent.mkdir(parents=True, exist_ok=True)



  with open(outpath_pre_head, "a", encoding="utf-8") as f:

    record = {
      "avg_blocks_nums": float(avg),
      "blocks": int(blocks),
      "seq_len": int(k_len),
    }
    json.dump(record, f, ensure_ascii=False)
    f.write("\n")



def Xattention_prefill(
  query_states: torch.Tensor,
  key_states: torch.Tensor,
  value_states: torch.Tensor,
  stride,
  type=" ",
  model_name=" ",
  task=" ",
  save_path=" ",
  layer_id=0,
  norm=1,
  threshold=0.8,
  block_size=128,
  use_triton=True,
  causal=True,
  kdb=1,
  chunk_size=None,
  keep_sink=False,
  keep_recent=False,
):
  batch_size, num_heads, k_len, head_dim = key_states.shape
  _, _, q_len, _ = query_states.shape

  q_block_num = (q_len + block_size - 1) // block_size
  k_block_num = (k_len + block_size - 1) // block_size
  if chunk_size is None:




    chunk_size = int(
      max(
        min(
          max(2048, 1 << (k_len - 1).bit_length()),
          128 * 1024 * 2048 // (1 << (k_len - 1).bit_length()),
        ),
        2048,
      )
    )



  attn_sums, approx_simple_mask = xattn_estimate(
    query_states,
    key_states,
    block_size=block_size,
    stride=stride,
    norm=norm,
    threshold=threshold,
    select_mode="inverse",
    use_triton=use_triton,
    causal=causal,
    chunk_size=chunk_size,
    kdb=kdb,
    keep_sink=keep_sink,
    keep_recent=keep_recent,
  )

  mask = approx_simple_mask.to(torch.bool)     # (B,H,q_blk,k_blk)
  selected_per_qblk = mask.sum(dim=-1)

  selected_per_qblk = selected_per_qblk.sum(dim=-1)
  avg_blocks_per_query = selected_per_qblk.float().mean()
  




  save_path=save_path+"/"+task


  selected_attn_mass_from_blockmask(stride,model_name,layer_id,query_states,key_states,approx_simple_mask,save_path=save_path)
  

  


  if query_states.device != value_states.device:
    value_states = value_states.to(query_states.device)
  if approx_simple_mask.device != query_states.device:
    approx_simple_mask = approx_simple_mask.to(query_states.device)

  ####################
  assert block_size == 128
  assert batch_size == 1
  query_states = query_states.transpose(1, 2).view(q_len, num_heads, head_dim)
  key_states = key_states.transpose(1, 2).view(k_len, num_heads, head_dim)
  value_states = value_states.transpose(1, 2).view(k_len, num_heads, head_dim)
  q_cu_seq_lens = torch.tensor(
    [0, q_len], dtype=torch.int32, device=query_states.device
  )
  k_cu_seq_lens = torch.tensor(
    [0, k_len], dtype=torch.int32, device=query_states.device
  )

  head_mask_type = torch.tensor(
    [1 for _ in range(num_heads)], device=query_states.device, dtype=torch.int32
  )
  assert head_mask_type.device == query_states.device
  assert q_cu_seq_lens.device == query_states.device
  assert k_cu_seq_lens.device == query_states.device
  assert key_states.device == query_states.device
  assert value_states.device == query_states.device
  assert approx_simple_mask.device == query_states.device


  attn_output = block_sparse_attn_func(
    query_states,
    key_states,
    value_states,
    q_cu_seq_lens,
    k_cu_seq_lens,
    head_mask_type,
    None,
    approx_simple_mask[:, :, :q_block_num, :k_block_num].contiguous(),
    q_len,
    k_len,
    p_dropout=0.0,
    deterministic=True,
    is_causal=causal,
  )
  attn_output = attn_output.view(batch_size, q_len, num_heads, head_dim).transpose(
    1, 2
  )
  ################################

  del query_states
  num_to_compute = (k_block_num + 1) * k_block_num / 2 * num_heads
  
  del approx_simple_mask, attn_sums
  return attn_output
