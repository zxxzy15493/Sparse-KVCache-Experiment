import ctypes
from pathlib import Path
from typing import Optional, Tuple, List
import math
import os
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import DynamicCache
from flash_attn import flash_attn_func 

# pylibraft otherwise picks an incompatible system CUDA library before the
# accuracy loader gets a chance to initialize the active Python environment.
_site_packages = Path(torch.__file__).resolve().parents[1]
for _relative in (
    "nvidia/cublas/lib/libcublas.so.12",
    "nvidia/cublas/lib/libcublasLt.so.12",
    "nvidia/cusolver/lib/libcusolver.so.11",
):
    _library = _site_packages / _relative
    if _library.is_file():
        ctypes.CDLL(str(_library), mode=ctypes.RTLD_GLOBAL)

from pylibraft.cluster import KMeansParams, fit
from pylibraft.neighbors import ivf_flat
import pylibraft.config
pylibraft.config.set_output_as("torch")
import rmm
import time
# torch.set_printoptions(profile="full")

from clusterkv._clusterkv_knl import search_indices
from .cluster_cache_simulator import CacheSimulator

CHECK_RECALL = eval(os.environ.get("CHECK_RECALL", "0"))

_recall_file = None
_recall_prompt_id = 0


def repeat_for_recall(a: torch.Tensor, size, dim_idx):
    shape = a.shape
    return a.unsqueeze(dim_idx + 1) \
        .expand(*shape[:dim_idx], shape[dim_idx], size, *shape[dim_idx + 1:]) \
        .reshape(*shape[:dim_idx], shape[dim_idx] * size, *shape[dim_idx + 1:])


def write_recall_prompt_separator(prefill_len=None):
    global _recall_prompt_id

    f = get_recall_file()
    if f is None:
        return None

    _recall_prompt_id += 1
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    f.write("\n")
    f.write(f"# ===== PROMPT {_recall_prompt_id} | {now} =====\n")
    if prefill_len is not None:
        f.write(f"# prefill_len: {prefill_len}\n")
    f.flush()

    return _recall_prompt_id


def get_recall_file():
    global _recall_file

    if _recall_file is not None:
        return _recall_file

    recall_name = os.environ.get("RECALL_NAME", "recall")
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = os.path.join("recall_list", recall_name)
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.join(save_dir, f"{recall_name}_{now}.csv")

    _recall_file = open(filename, "a", buffering=1)
    _recall_file.write("layer,head,recall,recall@100,selected_attn\n")

    return _recall_file


@torch.no_grad()
def calc_recall(query, key, dummy_topk_indices, num_kv_group, topk_size, layer_idx=0):
    """
    Write PQCache-compatible recall records.

    query:
        [bsz, n_heads, 1, head_dim]
    key:
        [bsz, kv_heads or n_heads, kv_seq_len, head_dim]
    dummy_topk_indices:
        [bsz, kv_heads or n_heads, 1, topk_size]
        or [bsz, kv_heads or n_heads, topk_size]
    """

    _, kv_head_num, kv_seq_len, dim = key.shape
    _, n_head, q_len, _ = query.shape

    assert q_len == 1, "recall logging only supports decode q_len == 1"
    assert topk_size <= kv_seq_len, f"topk_size={topk_size}, kv_seq_len={kv_seq_len}"

    if dummy_topk_indices.dim() == 3:
        dummy_topk_indices = dummy_topk_indices.unsqueeze(2)

    if key.shape[1] * num_kv_group == query.shape[1]:
        full_key = repeat_for_recall(key, num_kv_group, 1)
    elif key.shape[1] == query.shape[1]:
        full_key = key
    else:
        raise RuntimeError(
            f"Unsupported key/query shape for recall: "
            f"key={key.shape}, query={query.shape}, num_kv_group={num_kv_group}"
        )

    real_weight = query.float() @ full_key.float().transpose(2, 3)

    attn_scale = math.sqrt(dim)
    real_attn_score = torch.softmax(real_weight / attn_scale, dim=-1).to(query.dtype)

    real_topk_indices = real_weight.topk(k=topk_size, dim=-1, largest=True).indices
    real_top100_indices = real_weight.topk(k=min(100, kv_seq_len), dim=-1, largest=True).indices

    if dummy_topk_indices.shape[1] != real_topk_indices.shape[1]:
        dummy_topk_indices = repeat_for_recall(dummy_topk_indices, num_kv_group, 1)

    dummy_topk_indices = dummy_topk_indices.flatten(0, 1)
    real_topk_indices = real_topk_indices.flatten(0, 1)
    real_top100_indices = real_top100_indices.flatten(0, 1)

    f = get_recall_file()
    if layer_idx == 0:
        f.write("\n")
    f.flush()

    for h in range(n_head):
        dummy = dummy_topk_indices[h, 0, :].to(torch.long)
        real = real_topk_indices[h, 0, :]
        real100 = real_top100_indices[h, 0, :]

        assert dummy.numel() == torch.unique(dummy).numel(), (
            f"Recall indices contain duplicates at layer={layer_idx}, head={h}"
        )

        comparison = torch.isin(dummy, real, assume_unique=True)
        hit_cnt = torch.sum(comparison.int()).item()
        recall_rate = hit_cnt / topk_size

        comparison100 = torch.isin(dummy, real100, assume_unique=True)
        hit100_cnt = torch.sum(comparison100.int()).item()
        recall100_rate = hit100_cnt / min(100, kv_seq_len)

        head_real_attn = real_attn_score[0, h, 0, :]
        selected_attn = head_real_attn[dummy].sum().item()

        f.write(
            f"{layer_idx:3d},{h:3d},"
            f"{recall_rate:7.4f},{recall100_rate:7.4f},{selected_attn:7.4f}\n"
        )

# Use this function as the metadata only has 2-dim
def repeat_metadata(metadata: torch.Tensor, n_rep: int) -> torch.Tensor:
    num_key_value_heads, slen = metadata.shape
    if n_rep == 1:
        return metadata
    metadata = metadata[:, None, :].expand(num_key_value_heads, n_rep, slen)
    return metadata.reshape(num_key_value_heads * n_rep, slen)

# def build_cluster(prefill_key, nlist, balance, cluster_params, 
#                   num_key_value_groups, gqa_policy):
#     _, num_kv_heads, prefill_len, head_dim = prefill_key.shape
    
#     tmp_prefill_key = prefill_key.to(torch.float32)
#     nlist_range = torch.arange(nlist, device=prefill_key.device).reshape(nlist, 1)
#     cluster_key_indices = torch.empty((num_kv_heads, prefill_len), dtype=torch.int64,
#                                             device=prefill_key.device)
#     cluster_key_ptr = torch.empty((num_kv_heads, prefill_len), dtype=torch.int16,
#                                         device=prefill_key.device)
#     cluster_key_size = torch.empty((num_kv_heads, nlist), dtype=torch.int32,
#                                         device=prefill_key.device)
#     for h in range(num_kv_heads):
#         # head_keys = prefill_key[0, h].to(torch.float32)
#         head_keys = tmp_prefill_key[0, h]
#         if balance:
#             flat_index = ivf_flat.build(cluster_params, head_keys)
#             head_centroids = flat_index.centers
#         else:
#             head_centroids, _, _ = fit(cluster_params, head_keys)
#         head_centroids = head_centroids.to(prefill_key.dtype)
#         # centoid_indices: (prefill_len,)
#         _, centoid_indices = torch.max(torch.mm(F.normalize(prefill_key[0, h], p=2, dim=-1), 
#                                                 F.normalize(head_centroids, p=2, dim=-1).t()), 
#                                                 dim=-1)
#         # if centoid_indices is like [3, 1, 1, 2]
#         # cluster_key_ptr is [1, 1, 2, 3], cluster_key_indices is [1, 2, 3, 0]
#         cluster_key_ptr[h], cluster_key_indices[h] \
#             = torch.where(centoid_indices==nlist_range)
#         cluster_key_size[h] = torch.bincount(cluster_key_ptr[h], minlength=nlist)
#         # self.cluster_key[0, h] = prefill_key[0, h, self.cluster_key_indices[h], :]
        
#         # if self.layer_id == 10 and h == 1:
#         #     print(centoid_indices)
#         #     print(self.cluster_key_indices[h], self.cluster_key_ptr[h])
#         #     print(self.cluster_key_size[h])
#         head_centroids = head_centroids.unsqueeze(0)
#         if h == 0:
#             key_centroids = head_centroids
#         else:
#             key_centroids = torch.cat([key_centroids, head_centroids], dim=0)
#     # (num_kv_heads, nlist)
#     cluster_key_size_ps = torch.cumsum(cluster_key_size, dim=-1)
#     # if self.layer_id == 10:
#     #     print(self.cluster_key_size_ps[1])
#     key_centroids = key_centroids.unsqueeze(0)
#     # self.key_centroids: (1, num_kv_heads, nlist, head_dim)
#     if gqa_policy is None:
#         key_centroids = repeat_kv(key_centroids, num_key_value_groups)
#         cluster_key_ptr = repeat_metadata(cluster_key_ptr, num_key_value_groups)
#         cluster_key_size = repeat_metadata(cluster_key_size, num_key_value_groups)
#         cluster_key_size_ps = repeat_metadata(cluster_key_size_ps, num_key_value_groups)

#     return key_centroids, cluster_key_indices, cluster_key_ptr, cluster_key_size, cluster_key_size_ps
@torch.no_grad()
def _batched_kmeans_pytorch(
    keys: torch.Tensor,          # [num_kv_heads, seq_len, head_dim]
    centroids: torch.Tensor,     # [num_kv_heads, nlist, head_dim]
    labels: torch.Tensor,        # [num_kv_heads, seq_len]
    cluster_size: torch.Tensor,  # [num_kv_heads, nlist]
    niter: int,
):
    """
    Batched PyTorch kmeans.

    keys:      [num_kv_heads, seq_len, head_dim]
    centroids: [num_kv_heads, nlist, head_dim]
    labels:    [num_kv_heads, seq_len]

    Uses cosine similarity for assignment.
    """
    num_kv_heads, seq_len, head_dim = keys.shape
    nlist = centroids.shape[1]
    eps = 1e-8

    # keys stay constant across kmeans iterations; normalize only once
    keys_normalized = keys / (
        torch.linalg.norm(keys, dim=-1, keepdim=True) + eps
    )

    # preallocate to avoid repeated GPU memory allocations per iteration
    cluster_sum = torch.empty_like(centroids)

    cluster_size_tmp = torch.empty(
        num_kv_heads,
        nlist,
        dtype=torch.int32,
        device=keys.device,
    )

    ones = torch.ones(
        num_kv_heads,
        seq_len,
        dtype=torch.int32,
        device=keys.device,
    )

    for _ in range(niter):
        centroids_normalized = centroids / (
            torch.linalg.norm(centroids, dim=-1, keepdim=True) + eps
        )

        # [num_kv_heads, seq_len, nlist]
        cos_sim = torch.matmul(
            keys_normalized,
            centroids_normalized.transpose(1, 2),
        )

        # [num_kv_heads, seq_len]
        new_labels = cos_sim.argmax(dim=-1)
        labels.copy_(new_labels)

        # compute sum per cluster
        cluster_sum.zero_()

        scatter_index = new_labels.unsqueeze(-1).expand(
            num_kv_heads,
            seq_len,
            head_dim,
        )

        cluster_sum.scatter_add_(
            dim=1,
            index=scatter_index,
            src=keys,
        )

        # compute size per cluster
        cluster_size_tmp.zero_()

        cluster_size_tmp.scatter_add_(
            dim=1,
            index=new_labels,
            src=ones,
        )

        # empty clusters keep their previous centroid
        non_empty = cluster_size_tmp > 0

        new_centroids = centroids.clone()
        new_centroids[non_empty] = (
            cluster_sum[non_empty]
            / cluster_size_tmp[non_empty].to(keys.dtype).unsqueeze(-1)
        )

        centroids.copy_(new_centroids)

    # after the final centroid update, reassign labels once more
    centroids_normalized = centroids / (
        torch.linalg.norm(centroids, dim=-1, keepdim=True) + eps
    )

    cos_sim = torch.matmul(
        keys_normalized,
        centroids_normalized.transpose(1, 2),
    )

    final_labels = cos_sim.argmax(dim=-1)
    labels.copy_(final_labels)

    # recompute cluster_size from the final labels
    cluster_size.zero_()

    cluster_size.scatter_add_(
        dim=1,
        index=final_labels,
        src=ones,
    )

@torch.no_grad()
def build_cluster(
    prefill_key,
    nlist,
    balance,
    cluster_params,
    num_key_value_groups,
    gqa_policy,
):
    """
    New batched build_cluster.

    Inputs:
        prefill_key: [1, num_kv_heads, prefill_len, head_dim]

    Outputs:
        key_centroids:       [1, num_heads or num_kv_heads, nlist, head_dim]
        cluster_key_indices: [num_kv_heads, prefill_len]
        cluster_key_ptr:     [num_heads or num_kv_heads, prefill_len]
        cluster_key_size:    [num_heads or num_kv_heads, nlist]
        cluster_key_size_ps: [num_heads or num_kv_heads, nlist]
    """
    _, num_kv_heads, prefill_len, head_dim = prefill_key.shape
    device = prefill_key.device

    # [num_kv_heads, prefill_len, head_dim]
    keys = prefill_key[0].to(torch.float32).contiguous()

    # ------------------------------------------------------------
    # read niter
    # ------------------------------------------------------------
    if hasattr(cluster_params, "niter"):
        niter = cluster_params.niter
    elif isinstance(cluster_params, dict) and "niter" in cluster_params:
        niter = cluster_params["niter"]
    else:
        # fall back to a default when niter is not provided
        niter = 10

    # ------------------------------------------------------------
    # initialize centroids
    # following the new version: fixed seed, randomly pick nlist tokens from seq_len
    # ------------------------------------------------------------
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)

    init_idx = torch.randint(
        low=0,
        high=prefill_len,
        size=(nlist,),
        generator=gen,
    ).to(device)

    # [num_kv_heads, nlist, head_dim]
    centroids = keys[:, init_idx, :].contiguous()

    # ------------------------------------------------------------
    # labels / metadata buffer
    # ------------------------------------------------------------
    labels = torch.empty(
        num_kv_heads,
        prefill_len,
        dtype=torch.int64,
        device=device,
    )

    cluster_key_size = torch.empty(
        num_kv_heads,
        nlist,
        dtype=torch.int32,
        device=device,
    )

    # ------------------------------------------------------------
    # Batched kmeans
    # Note: balance is not used here for now.
    # If balance=True previously relied on ivf_flat.build's balanced behavior,
    # this version is not strictly equivalent.
    # ------------------------------------------------------------
    _batched_kmeans_pytorch(
        keys=keys,
        centroids=centroids,
        labels=labels,
        cluster_size=cluster_key_size,
        niter=niter,
    )

    # ------------------------------------------------------------
    # build cluster_key_ptr / cluster_key_indices
    #
    # Original logic:
    #   cluster_key_ptr[h], cluster_key_indices[h] =
    #       torch.where(centoid_indices == nlist_range)
    #
    # Equivalent replacement:
    #   sorted_labels, sorted_indices = torch.sort(labels)
    #
    # e.g. labels = [3, 1, 1, 2]
    # sorted_labels  = [1, 1, 2, 3]
    # sorted_indices = [1, 2, 3, 0]
    # ------------------------------------------------------------
    sorted_labels, sorted_indices = torch.sort(labels, dim=-1)

    cluster_key_ptr = sorted_labels.to(torch.int16)
    cluster_key_indices = sorted_indices.to(torch.int64)

    # [num_kv_heads, nlist]
    cluster_key_size_ps = torch.cumsum(cluster_key_size, dim=-1)

    # [1, num_kv_heads, nlist, head_dim]
    key_centroids = centroids.to(prefill_key.dtype).unsqueeze(0)

    # ------------------------------------------------------------
    # GQA repeat
    # keep consistent with the original version
    # ------------------------------------------------------------
    if gqa_policy is None:
        key_centroids = repeat_kv(key_centroids, num_key_value_groups)

        cluster_key_ptr = repeat_metadata(
            cluster_key_ptr,
            num_key_value_groups,
        )

        cluster_key_size = repeat_metadata(
            cluster_key_size,
            num_key_value_groups,
        )

        cluster_key_size_ps = repeat_metadata(
            cluster_key_size_ps,
            num_key_value_groups,
        )

    return (
        key_centroids,
        cluster_key_indices,
        cluster_key_ptr,
        cluster_key_size,
        cluster_key_size_ps,
    )

def stat_topk(layer_id, indices, q, prefill_keys, name):
    _, num_heads, k = indices.shape
    attn_weights = torch.matmul(q, prefill_keys.transpose(2, 3))    # [1, num_heads, 1, seq_len]
    _, topk_indices = attn_weights.topk(k, dim=-1)
    topk_indices = topk_indices.squeeze(2)      # [1, 32, k]
    hit_rate = []
    for h in range(num_heads):
        truth = topk_indices[0, h].cpu()
        pred = indices[0, h].cpu()
        hit_rate.append(len( set(truth.numpy()) & set(pred.numpy()) ) / k)
    avg_hit_rate = sum(hit_rate) / len(hit_rate)
    with open(f'topk_stat/top{k}-{name}.csv', 'a') as f:
        f.write(f'layer {layer_id}, {avg_hit_rate}\n')
        
def cluster_attn_out(
    query_states,
    key_states,
    value_states,
    attention_mask,
    prompt_len,
    key_centroids,
    cluster_key_indices,
    cluster_key_size,
    cluster_key_size_ps,
    num_key_value_groups,
    layer_id,
    token_budget,
    sink,
    head_sel,
    cluster_cache,
    topk_stat=False,
    cluster_params=None,
    max_search_clusters=None,
):
    """
    Optimized cluster attention for decode.

    query_states:        [bsz, num_heads, q_len, head_dim]
    key_states:          [bsz, num_heads, kv_seq_len, head_dim]
                         Note: this version assumes key_states / value_states have
                         already been repeat_kv'd to num_heads.
                         To further optimize, remove repeat_kv in forward_cluster,
                         but that requires a separate gather-logic change.
    value_states:        [bsz, num_heads, kv_seq_len, head_dim]
    key_centroids:       [1, num_heads, nlist, head_dim]
    cluster_key_indices: [num_kv_heads, prefill_len_without_sink]
                         or [num_heads, prefill_len_without_sink]
    cluster_key_size:    [num_heads, nlist]
    cluster_key_size_ps: [num_heads, nlist]

    Returns:
        attn_output: [bsz, q_len, hidden_size]
    """

    bsz, num_kv_or_heads, kv_seq_len, head_dim = key_states.shape
    _, num_heads, q_len, _ = query_states.shape

    assert bsz == 1
    assert q_len == 1, "cluster_attn_out optimized version is intended for decode q_len == 1"

    hidden_size = num_heads * head_dim
    device = query_states.device

    # ------------------------------------------------------------
    # 0. basic budget
    # ------------------------------------------------------------
    need_tokens = token_budget - sink
    if need_tokens < 0:
        need_tokens = 0

    nlist = key_centroids.shape[-2]

    if max_search_clusters is None:
        # default: keep semantics close to the original (scan all clusters)
        # for speed, pass 32 / 64 / 128 at the call site.
        max_search_clusters = nlist
    else:
        max_search_clusters = min(max_search_clusters, nlist)

    # ------------------------------------------------------------
    # 1. score query against centroids
    #
    # Original:
    #   c_dist = q @ centroid.T
    #   sort all nlist
    #
    # New:
    #   topk, take only the top max_search_clusters clusters
    # ------------------------------------------------------------
    # c_dist: [1, num_heads, 1, nlist]
    c_dist = torch.matmul(
        query_states,
        key_centroids.transpose(2, 3),
    )

    # c_neighbor: [1, num_heads, 1, max_search_clusters]
    _, c_neighbor = torch.topk(
        c_dist,
        k=max_search_clusters,
        dim=-1,
        largest=True,
        sorted=True,
    )

    # [num_heads, max_search_clusters]
    c_neighbor = c_neighbor.squeeze(0).squeeze(-2)

    # ------------------------------------------------------------
    # 2. get size / start / end of each selected cluster
    # ------------------------------------------------------------
    # [num_heads, max_search_clusters]
    sel_cluster_size = torch.gather(
        cluster_key_size,
        dim=-1,
        index=c_neighbor,
    )

    # prefix sum over selected clusters
    # [num_heads, max_search_clusters]
    sel_cluster_size_ps = torch.cumsum(sel_cluster_size, dim=-1)

    # end / start inside the original sorted cluster_key_indices
    # [num_heads, max_search_clusters]
    sel_cluster_key_end = torch.gather(
        cluster_key_size_ps,
        dim=-1,
        index=c_neighbor,
    )

    sel_cluster_key_start = sel_cluster_key_end - sel_cluster_size

    # cache simulator: keep the original logic
    if cluster_cache is not None:
        assert cluster_params is not None
        cluster_params.update(c_neighbor)

    # ------------------------------------------------------------
    # 3. vectorized generation of selected token indices
    #
    # Original flow:
    #   max_num_indices = torch.sum(...).max()
    #   torch.full(dynamic shape)
    #   search_indices(...)
    #   slice to token_budget - sink
    #
    # New version generates a fixed [num_heads, need_tokens] tensor,
    # avoiding .item() and dynamic shapes.
    # ------------------------------------------------------------
    if need_tokens > 0:
        # pos: [num_heads, need_tokens]
        pos = torch.arange(
            need_tokens,
            device=device,
            dtype=sel_cluster_size_ps.dtype,
        ).unsqueeze(0).expand(num_heads, -1)

        # total selected tokens from max_search_clusters clusters
        # [num_heads, 1]
        total_selected = sel_cluster_size_ps[:, -1:].clamp_min(0)

        # valid position: pos < total_selected
        # if max_search_clusters cannot cover need_tokens, tail positions are masked out.
        selected_valid = pos < total_selected

        # for each pos, find which selected cluster it falls into
        #
        # boundaries: sel_cluster_size_ps[h] = [s0, s0+s1, ...]
        # pos=0..need_tokens-1
        #
        # searchsorted(..., right=True):
        #   pos < s0          -> cluster_rank 0
        #   s0 <= pos < s0+s1 -> cluster_rank 1
        #
        # [num_heads, need_tokens]
        cluster_rank = torch.searchsorted(
            sel_cluster_size_ps.contiguous(),
            pos.contiguous(),
            right=True,
        )

        # guard against cluster_rank == max_search_clusters (OOB) for invalid pos
        cluster_rank_safe = cluster_rank.clamp_max(max_search_clusters - 1)

        # prev prefix sum
        prev_rank = (cluster_rank_safe - 1).clamp_min(0)

        prev_ps = torch.gather(
            sel_cluster_size_ps,
            dim=1,
            index=prev_rank,
        )

        prev_ps = torch.where(
            cluster_rank_safe > 0,
            prev_ps,
            torch.zeros_like(prev_ps),
        )

        # offset of current token inside the cluster
        offset_in_cluster = pos - prev_ps

        # position inside the sorted cluster_key_indices array
        # [num_heads, need_tokens]
        src_pos = torch.gather(
            sel_cluster_key_start,
            dim=1,
            index=cluster_rank_safe,
        ) + offset_in_cluster

        # clamp invalid positions to 0 first; they are masked out later
        src_pos = torch.where(
            selected_valid,
            src_pos,
            torch.zeros_like(src_pos),
        )

        src_pos = src_pos.to(torch.long)

        # --------------------------------------------------------
        # cluster_key_indices is typically [num_kv_heads, prefill_len_without_sink]
        # but cluster_key_size / centroids may already be repeated to num_heads.
        #
        # if cluster_key_indices is already num_heads, just use the head id directly.
        # otherwise map h -> h // num_key_value_groups per GQA.
        # --------------------------------------------------------
        if cluster_key_indices.shape[0] == num_heads:
            gather_head_ids = torch.arange(
                num_heads,
                device=device,
                dtype=torch.long,
            )
        else:
            gather_head_ids = (
                torch.arange(num_heads, device=device, dtype=torch.long)
                // num_key_value_groups
            )

        gather_head_ids = gather_head_ids.unsqueeze(1).expand(
            num_heads,
            need_tokens,
        )

        # [num_heads, need_tokens]
        sel_key_indices = cluster_key_indices[
            gather_head_ids,
            src_pos,
        ]

        # clusters are built on prefill_key[..., sink:, :], so add the sink offset back
        sel_key_indices = sel_key_indices + sink

        # defensive clamp
        sel_key_indices = sel_key_indices.clamp(min=0, max=max(kv_seq_len - 1, 0))

    else:
        sel_key_indices = torch.empty(
            num_heads,
            0,
            dtype=torch.long,
            device=device,
        )
        selected_valid = torch.empty(
            num_heads,
            0,
            dtype=torch.bool,
            device=device,
        )

    # ------------------------------------------------------------
    # 4. assemble full attention indices:
    #    [sink tokens] + [selected prefill tokens] + [decode/current window tokens]
    #
    # Original: gather selected K/V first, then concat K/V.
    # New: concat indices first, then do a single K/V gather.
    # ------------------------------------------------------------
    if sink > 0:
        sink_indices = torch.arange(
            sink,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(num_heads, -1)

        sink_valid = torch.ones(
            num_heads,
            sink,
            dtype=torch.bool,
            device=device,
        )
    else:
        sink_indices = torch.empty(
            num_heads,
            0,
            dtype=torch.long,
            device=device,
        )
        sink_valid = torch.empty(
            num_heads,
            0,
            dtype=torch.bool,
            device=device,
        )

    # current decode window:
    # key_states is already updated, so prompt_len:kv_seq_len covers new decode tokens.
    if kv_seq_len > prompt_len:
        cur_indices = torch.arange(
            prompt_len,
            kv_seq_len,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(num_heads, -1)

        cur_valid = torch.ones(
            num_heads,
            kv_seq_len - prompt_len,
            dtype=torch.bool,
            device=device,
        )
    else:
        cur_indices = torch.empty(
            num_heads,
            0,
            dtype=torch.long,
            device=device,
        )
        cur_valid = torch.empty(
            num_heads,
            0,
            dtype=torch.bool,
            device=device,
        )

    # [num_heads, total_selected_len]
    full_indices = torch.cat(
        [
            sink_indices,
            sel_key_indices,
            cur_indices,
        ],
        dim=-1,
    )

    # [num_heads, total_selected_len]
    full_valid = torch.cat(
        [
            sink_valid,
            selected_valid,
            cur_valid,
        ],
        dim=-1,
    )

    # defensive clamp to avoid OOB in gather
    full_indices = full_indices.clamp(min=0, max=max(kv_seq_len - 1, 0))

    total_k = full_indices.shape[-1]

    # ------------------------------------------------------------
    # 5. single gather of K/V
    # ------------------------------------------------------------
    gather_index = full_indices.view(
        1,
        num_heads,
        total_k,
        1,
    ).expand(
        bsz,
        num_heads,
        total_k,
        head_dim,
    )

    sel_key_states = key_states.gather(
        dim=2,
        index=gather_index,
    )

    sel_value_states = value_states.gather(
        dim=2,
        index=gather_index,
    )

    # ------------------------------------------------------------
    # 6. attention with flash_attn
    # ------------------------------------------------------------
    # flash_attn_func expects:
    #   q: [bsz, q_len, num_heads, head_dim]
    #   k: [bsz, total_k, num_heads, head_dim]
    #   v: [bsz, total_k, num_heads, head_dim]
    #
    # current tensors are:
    #   query_states:   [bsz, num_heads, q_len, head_dim]
    #   sel_key_states: [bsz, num_heads, total_k, head_dim]
    #   sel_value_states: [bsz, num_heads, total_k, head_dim]

    # Note:
    # 1. flash_attn_func cannot consume an additive attention_mask directly.
    # 2. full_valid may contain invalid selected tokens.
    #    If max_search_clusters fully covers need_tokens, full_valid is usually all True.
    #    Otherwise keep the original attention logic or ensure invalid tokens never appear.
    if not torch.all(full_valid):
        # fallback: keep the original masked attention to skip invalid tokens
        attn_weights = torch.matmul(
            query_states,
            sel_key_states.transpose(2, 3),
        ) / math.sqrt(head_dim)

        invalid_mask = ~full_valid.view(1, num_heads, 1, total_k)

        attn_weights = attn_weights.masked_fill(
            invalid_mask,
            torch.finfo(attn_weights.dtype).min,
        )

        if attention_mask is not None:
            if attention_mask.shape[-1] >= kv_seq_len:
                mask_src = attention_mask[:, :, :, :kv_seq_len]

                if mask_src.shape[1] == 1:
                    mask_src = mask_src.expand(bsz, num_heads, q_len, kv_seq_len)

                mask_index = full_indices.view(
                    1,
                    num_heads,
                    1,
                    total_k,
                ).expand(
                    bsz,
                    num_heads,
                    q_len,
                    total_k,
                )

                gathered_mask = mask_src.gather(
                    dim=-1,
                    index=mask_index,
                )

                attn_weights = attn_weights + gathered_mask

        attn_weights = F.softmax(
            attn_weights,
            dim=-1,
            dtype=torch.float32,
        ).to(query_states.dtype)

        attn_output = torch.matmul(
            attn_weights,
            sel_value_states,
        )

    else:
        attn_output = flash_attn_func(
            query_states.transpose(1, 2),        # [bsz, q_len, num_heads, head_dim]
            sel_key_states.transpose(1, 2),      # [bsz, total_k, num_heads, head_dim]
            sel_value_states.transpose(1, 2),    # [bsz, total_k, num_heads, head_dim]
            causal=True,
        ).transpose(1, 2)                        # [bsz, num_heads, q_len, head_dim]

        if attn_output.size() != (bsz, num_heads, q_len, head_dim):
            raise ValueError(
                f"`attn_output` should be of size "
                f"{(bsz, num_heads, q_len, head_dim)}, "
                f"but is {attn_output.size()}"
            )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)

    # ------------------------------------------------------------
    # 7. PQCache-compatible recall logging
    # ------------------------------------------------------------
    if CHECK_RECALL:
        # calc_recall expects [bsz, head, q_len, selected_len].
        # If max_search_clusters is too small, invalid positions are padded
        # with duplicate indices, which would break the uniqueness check.
        # The default max_search_clusters=nlist should make all positions valid.
        if torch.all(full_valid):
            recall_indices = full_indices.view(1, num_heads, 1, total_k)
            calc_recall(
                query_states,
                key_states,
                recall_indices,
                num_key_value_groups,
                recall_indices.shape[-1],
                layer_idx=layer_id,
            )

    return attn_output
# def cluster_attn_out(query_states, key_states, value_states, attention_mask, prompt_len,
#                     key_centroids, cluster_key_indices, cluster_key_size, cluster_key_size_ps,
#                     num_key_value_groups, layer_id, token_budget, sink, head_sel, 
#                     cluster_cache, topk_stat=False, cluster_params=None):
#     bsz, num_kv_heads, kv_seq_len, head_dim = key_states.shape
#     _, num_heads, q_len, _ = query_states.shape
#     hidden_size = num_heads * head_dim
# 	# c_dist: (1, num_heads, 1, nlist)
#     c_dist = torch.matmul(query_states, key_centroids.transpose(2, 3))
#     _, c_neighbor = torch.sort(c_dist, dim=-1, descending=True)
#     # (num_heads, nlist)
#     c_neighbor = c_neighbor.squeeze(0).squeeze(-2)
#     neighbor_cluster_size = torch.gather(cluster_key_size, -1, c_neighbor)
#     neighbor_cluster_key_size_ps = torch.cumsum(neighbor_cluster_size, dim=-1)
#     # get the number of needed clusters by mask smaller and get min
#     neighbor_cluster_key_size_ps[neighbor_cluster_key_size_ps < (token_budget-sink)] = 10000000
#     # num_need_clusters: (num_kv_heads)
#     _, num_need_clusters = torch.min(neighbor_cluster_key_size_ps, dim=-1)
#     num_need_clusters += 1
#     # now we select same number of clusters for all heads
#     max_num_need_clusters = torch.max(num_need_clusters).item()

# 	# (num_heads, max_num_need_clusters)
#     sel_cluster_indices = c_neighbor[:, :max_num_need_clusters]
#     sel_cluster_size = neighbor_cluster_size[:, :max_num_need_clusters]
#     # not use neighbor_cluster_key_size_ps[, :max_num_need_clusters] as it has be modified
#     sel_cluster_size_ps = torch.cumsum(sel_cluster_size, dim=-1)
#     sel_cluster_key_end = torch.gather(cluster_key_size_ps, -1, sel_cluster_indices)
#     sel_cluster_key_start = sel_cluster_key_end - sel_cluster_size

#     if cluster_cache is not None:
#         cluster_params.update(sel_cluster_indices)
    
#     # if self.layer_id == 10:
#     #     print(neighbor_cluster_size[1])
#     #     print(neighbor_cluster_key_size_ps[1])
#     #     print(num_need_clusters[1])
#     #     print(max_num_need_clusters)
#     #     print(sel_cluster_indices[1])
#     #     print(sel_cluster_size[1])
#     #     print(sel_cluster_key_end[1])
#     #     print(sel_cluster_key_start[1])
#     #     print()
#     use_search_kernel = True

#     if use_search_kernel:
#         max_num_indices = torch.sum(sel_cluster_size, dim=-1).max()
#         sel_key_indices = torch.full((num_heads, max_num_indices), kv_seq_len, 
#                                      dtype=torch.int64, device='cuda')
#         search_indices(num_need_clusters,
#                     sel_cluster_size_ps,
#                     sel_cluster_key_start,
#                     sel_cluster_key_end,
#                     cluster_key_indices,
#                     sel_key_indices)
#         sel_key_indices = sel_key_indices[:, :token_budget-sink]
#     else:
#         sel_key_indices = []
#         for h in range(num_heads):
#             kv_h = h // num_key_value_groups
#             head_num_need_clusters = num_need_clusters[h]
#             head_sel_key_indices = []
#             for i in range(head_num_need_clusters):
#                 head_sel_key_indices.append(cluster_key_indices[
#                     kv_h, sel_cluster_key_start[h, i]: sel_cluster_key_end[h, i]
#                 ])
#             head_sel_key_indices = torch.cat(head_sel_key_indices)
#             # if self.layer_id == 10:
#             #     print(head_sel_key_indices.shape)
#             sel_key_indices.append(head_sel_key_indices)
#         # sel_key_indices: (1, num_heads, token_budget) or 
#         if head_sel == "pad":
#             sel_key_indices = pad_sequence(sel_key_indices, batch_first=True, padding_value=kv_seq_len)
#         elif head_sel == "truc":
#             sel_key_indices = torch.stack([ind[:token_budget-sink] for ind in sel_key_indices])
#         else:
#             assert False
    
#     sel_key_indices = sel_key_indices.unsqueeze(0)
#     sel_key_indices += sink
#     sel_key_indices[sel_key_indices > kv_seq_len] = kv_seq_len

#     if topk_stat:
#         sink_indices = torch.arange(sink, device=sel_key_indices.device).repeat(1, num_heads, 1)
#         full_sel_key_indices = torch.cat([sink_indices, sel_key_indices], dim=-1)
#         nlist = c_dist.shape[-1]
#         assert cluster_params is not None
#         stat_topk(layer_id, full_sel_key_indices, query_states, 
#                   key_states[:, :, :prompt_len, :], 
#                   f'nc{nlist}-fi{cluster_params.max_iter}')

#     # sel_key_indices: (1, num_heads, token_budget, head_dim)
#     sel_key_indices = sel_key_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
#     if head_sel == "truc":
#         sel_key_states = key_states.gather(dim=2, index=sel_key_indices)
#         sel_value_states = value_states.gather(dim=2, index=sel_key_indices)
#     elif head_sel == "pad":
#         kpad = torch.ones((key_states.shape[0], key_states.shape[1], 
#                                 1, key_states.shape[3]), dtype=key_states.dtype, 
#                                 device=key_states.device) * torch.finfo(key_states.dtype).min
#         vpad = torch.zeros((value_states.shape[0], value_states.shape[1], 
#                                 1, value_states.shape[3]), dtype=value_states.dtype, 
#                                 device=value_states.device)
#         sign = (query_states > 0) + (~(query_states > 0)) * -1
#         sel_key_states = torch.cat([key_states, kpad*sign], dim=2).gather(dim=2, index=sel_key_indices)
#         sel_value_states = torch.cat([value_states, vpad], dim=2).gather(dim=2, index=sel_key_indices)

#     sel_key_states = torch.cat([key_states[:, :, :sink, :], sel_key_states, 
#                                 key_states[:, :, prompt_len:, :]], dim=2)
#     sel_value_states = torch.cat([value_states[:, :, :sink, :], sel_value_states, 
#                                   value_states[:, :, prompt_len:, :]], dim=2)

#     # if self.layer_id == 10:
#     #     print(sel_key_states.shape, sel_value_states.shape)
#     attn_weights = torch.matmul(query_states, sel_key_states.transpose(2, 3)) / math.sqrt(head_dim)

#     if attention_mask is not None:  # no matter the length, we just slice it
#         causal_mask = attention_mask[:, :, :, : sel_key_states.shape[-2]]
#         attn_weights = attn_weights + causal_mask

#     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#     # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
#     attn_output = torch.matmul(attn_weights, sel_value_states)

#     if attn_output.size() != (bsz, num_heads, q_len, head_dim):
#         raise ValueError(
#             f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
#             f" {attn_output.size()}"
#         )

#     attn_output = attn_output.transpose(1, 2).contiguous()
#     attn_output = attn_output.reshape(bsz, q_len, hidden_size)
#     return attn_output

def forward_cluster(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[DynamicCache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    assert bsz == 1

    if self.layer_id < 2 or q_len > 1 \
        or (self.prompt_len == 0 and q_len < self.token_budget) \
        or (self.prompt_len > 0 and self.prompt_len+q_len < self.token_budget) :   # for first several tokens of ppl_eval
        if q_len > 1:
            if CHECK_RECALL and self.layer_id == 0:
                write_recall_prompt_separator(prefill_len=q_len)

            self.prompt_len = q_len
            # reset cache for each request
            if self.cache_steps > 0 and self.layer_id >= 2:
                self.cluster_cache = CacheSimulator(self.layer_id, self.cache_steps+1)
        return self.flash_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            **kwargs,
        )
    
    sink = self.sink
    prefill_key = past_key_value[self.layer_id][0]
    assert prefill_key.shape[-2] > sink
    prefill_key = prefill_key[..., sink:, :]
    # clustering for prefilled keys
    # st = time.perf_counter()
    if self.key_centroids is None:
        self.key_centroids, self.cluster_key_indices, \
        self.cluster_key_ptr, self.cluster_key_size, self.cluster_key_size_ps = \
		build_cluster(prefill_key, self.nlist, self.balance, self.cluster_params,
                    self.num_key_value_groups, self.gqa_policy)
    # se = time.perf_counter()
    # print(f"build_time: {se - st}")

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

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[self.layer_id][0].shape[-2]
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_id)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    token_budget = min(self.prompt_len, self.token_budget)
    attn_output = cluster_attn_out(
        query_states, key_states, value_states, attention_mask, 
        self.prompt_len, self.key_centroids, self.cluster_key_indices, 
        self.cluster_key_size, self.cluster_key_size_ps,
        self.num_key_value_groups, self.layer_id, token_budget, 
        sink, self.head_sel, self.cluster_cache, self.topk_stat, self.cluster_params
    )
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def split_tensor_along_last_dim(
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

@torch.jit.script
def glm_apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [b, np, sq, hn]
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
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

def forward_cluster_glm(
    self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
):
    bsz, q_len, _ = hidden_states.size()
    assert bsz == 1

    if q_len > 1 or self.layer_number < 3 \
        or (self.prompt_len == 0 and q_len < self.token_budget) \
        or (self.prompt_len > 0 and self.prompt_len+q_len < self.token_budget) :   # for first several tokens of ppl_eval
        if q_len > 1:
            # ChatGLM-style layer_number is usually 1-based.
            # If your implementation is 0-based, change this condition to == 0.
            if CHECK_RECALL and self.layer_number == 0:
                write_recall_prompt_separator(prefill_len=q_len)

            self.prompt_len = q_len
            if self.cache_steps > 0 and self.layer_number>= 2:
                self.cluster_cache = CacheSimulator(self.layer_number, self.cache_steps+1)
        return self.flash_forward(
            hidden_states,
            attention_mask,
            rotary_pos_emb,
            kv_cache,
            use_cache
        )

    sink = self.sink
    prefill_key = kv_cache[0]
    prefill_key = prefill_key[..., sink:, :]
    num_key_value_group = self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition
    # clustering for prefilled keys
    if self.key_centroids is None:
        self.key_centroids, self.cluster_key_indices, \
        self.cluster_key_ptr, self.cluster_key_size, self.cluster_key_size_ps = \
		build_cluster(prefill_key, self.nlist, self.balance, self.cluster_params, 
                    num_key_value_group, self.gqa_policy)
    # hidden_states: [b, sq, h]

    # =================================================
    # Pre-allocate memory for key-values for inference.
    # =================================================
    # =====================
    # Query, Key, and Value
    # =====================

    # Attention heads [b, sq, h] --> [b, sq, (np * 3 * hn)]
    mixed_x_layer = self.query_key_value(hidden_states)

    if self.multi_query_attention:
        (query_layer, key_layer, value_layer) = mixed_x_layer.split(
            [
                self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
            ],
            dim=-1,
        )
        query_layer = query_layer.view(
            query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        )
        key_layer = key_layer.view(
            key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
        value_layer = value_layer.view(
            value_layer.size()[:-1]
            + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
    else:
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
                            (self.num_attention_heads_per_partition,
                            3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

    # [b, sq, np, hn] -> [b, np, sq, hn]
    query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

    # apply relative positional encoding (rotary embedding)
    if rotary_pos_emb is not None:
        query_layer = glm_apply_rotary_pos_emb(query_layer, rotary_pos_emb)
        key_layer = glm_apply_rotary_pos_emb(key_layer, rotary_pos_emb)

    # adjust key and value for inference
    if kv_cache is not None:
        cache_k, cache_v = kv_cache
        key_layer = torch.cat((cache_k, key_layer), dim=2)
        value_layer = torch.cat((cache_v, value_layer), dim=2)
    if use_cache:
        if kv_cache is None:
            kv_cache = torch.cat((key_layer.unsqueeze(0).unsqueeze(0), value_layer.unsqueeze(0).unsqueeze(0)),
                                    dim=1)
        else:
            kv_cache = (key_layer, value_layer)
    else:
        kv_cache = None

    if self.multi_query_attention:
        key_layer = key_layer.unsqueeze(2)
        key_layer = key_layer.expand(
            -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
        )
        key_layer = key_layer.contiguous().view(
            key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
        )
        value_layer = value_layer.unsqueeze(2)
        value_layer = value_layer.expand(
            -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
        )
        value_layer = value_layer.contiguous().view(
            value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
        )

    # ==================================
    # core attention computation
    # ==================================

    # context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
    token_budget = min(self.prompt_len, self.token_budget)
    context_layer = cluster_attn_out(
        query_layer, key_layer, value_layer, attention_mask, 
        self.prompt_len, self.key_centroids, self.cluster_key_indices, 
        self.cluster_key_size, self.cluster_key_size_ps,
        num_key_value_group, self.layer_number, token_budget, 
        sink, self.head_sel, self.cluster_cache, self.topk_stat, self.cluster_params
    )

    # =================
    # Output. [sq, b, h]
    # =================

    output = self.dense(context_layer)

    return output, kv_cache

MAX_POOL_SIZE = 1*1024**3
def cluster_reset(model):
    if isinstance(model, PreTrainedModel):
        # rmm.reinitialize(pool_allocator=True, initial_pool_size=MAX_POOL_SIZE, maximum_pool_size=MAX_POOL_SIZE)
        torch.cuda.empty_cache()
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            cluster_reset(module)
        module.key_centroids = None
        module.cluster_key_indices = None
        module.cluster_key_ptr = None
        module.cluster_key_size = None
        module.cluster_key_size_ps = None

def apply_cluster_config(module, args):
    nlist = args.nlist
    module.nlist = nlist
    module.head_sel = args.head_sel
    module.balance = True if args.balance else False
    module.sink = args.sink
    module.gqa_policy = args.gqa_policy
    if args.balance:
        module.cluster_params = ivf_flat.IndexParams(
        n_lists=nlist, metric='inner_product', kmeans_n_iters=args.fit_iter,
        kmeans_trainset_fraction=1, add_data_on_build=False)
    else:
        module.cluster_params = KMeansParams(
            n_clusters=nlist, max_iter=args.fit_iter, metric=args.dist_t)
