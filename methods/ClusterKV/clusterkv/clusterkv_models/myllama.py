# FlashInfer-free ClusterKV implementation for Llama.
#
# Goals
# -----
# - Remove flashinfer begin/forward/end calls from the attention path.
# - Keep the original ClusterKV controller, cache layout, append/offload,
#   clustering, retrieval, and recall kernels.
# - Use flash_attn for dense and materialized sparse attention.
#
# Out of scope
# ------------
# - Beam search (not supported by ClusterKV at all).
# - bsz != 1 (matches the original `assert bsz == 1`).
#
# Reference files (NOT modified):
#   clusterkv/clusterkv_models/llama.py
#   clusterkv/clusterkv_models/ClusterKVAttention.py
#   clusterkv/clusterkv_utils/clusterkv_controller.py
#   clusterkv/clusterkv_utils/decode_wrapper.py
#   clusterkv/clusterkv_utils/__init__.py

""" LLaMA model using ClusterKV native cache/kernels and flash-attn. """

import math
import os
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import clusterkv._clusterkv_knl as _kernels
from flash_attn import flash_attn_func
from torch import nn
from clusterkv.clusterkv_utils import append_kv, build_cluster
from clusterkv.clusterkv_utils.clusterkv_controller import ClusterKVController
from clusterkv.clusterkv_utils.global_timer import stage_begin, stage_end

from transformers import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from clusterkv.clusterkv_models.ClusterKVAttention import ClusterKVAttention


# ---------------------------------------------------------------------------
# Module-level helpers (pure PyTorch replacements for the CUDA kernels)
# ---------------------------------------------------------------------------


def pytorch_kmeans(
    keys: torch.Tensor,
    nlist: int,
    niter: int,
    init_idx: Optional[torch.Tensor] = None,
    epsilon: float = 1e-8,
):
    """Cosine-similarity k-means.

    Replaces `_kernels.update_centroids` + the surrounding cluster init in
    `clusterkv/clusterkv_utils/__init__.py:build_cluster`.

    Args:
        keys: [num_kv_heads, seq_len, head_dim].
        nlist: number of clusters.
        niter: number of Lloyd iterations.
        init_idx: optional [nlist] long tensor of seed positions. If None,
            samples uniformly without replacement via randint (matching the
            original).

    Returns:
        centroids: [num_kv_heads, nlist, head_dim] (not L2-normalized).
        labels:    [num_kv_heads, seq_len]        int64.
    """
    num_kv_heads, seq_len, head_dim = keys.shape
    device = keys.device
    dtype = keys.dtype

    if init_idx is None:
        # Match the original's `torch.randint(0, seq_len, (nlist,))` seeding.
        # When seq_len < nlist, fall back to modulo to avoid index errors.
        if seq_len >= nlist:
            init_idx = torch.randint(0, seq_len, (nlist,), device=device)
        else:
            init_idx = torch.randint(0, seq_len, (nlist,), device=device) % seq_len
    centroids = keys[:, init_idx, :].clone()  # [Hk, nlist, D]

    # Allocate labels once and reuse.
    labels = torch.zeros(num_kv_heads, seq_len, dtype=torch.int64, device=device)

    for _ in range(niter):
        c_norm = F.normalize(centroids, dim=-1, eps=epsilon)
        k_norm = F.normalize(keys, dim=-1, eps=epsilon)
        # [Hk, S, nlist]
        cos_sim = torch.matmul(k_norm, c_norm.transpose(1, 2))
        labels = cos_sim.argmax(dim=-1)

        # Update centroids: scatter_add keys into the cluster slots.
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(num_kv_heads, nlist, device=device, dtype=dtype)
        # Expand labels to match the head_dim for scatter.
        labels_exp = labels.unsqueeze(-1).expand(-1, -1, head_dim)
        new_centroids.scatter_add_(1, labels_exp, keys)
        # Counts: scatter_add ones into cluster slots.
        ones = torch.ones(num_kv_heads, seq_len, device=device, dtype=dtype)
        counts.scatter_add_(1, labels, ones)
        # Avoid div-by-zero for empty clusters (keeps old centroid in place).
        new_centroids = new_centroids / counts.clamp_min(1.0).unsqueeze(-1)
        centroids = new_centroids

    return centroids, labels


def pytorch_count_and_sort(
    labels: torch.Tensor, nlist: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Count labels, prefix-sum, and sort keys by cluster.

    Replaces `_kernels.count_labels` and the surrounding `argsort` in
    `clusterkv/clusterkv_utils/__init__.py:build_cluster`.

    Args:
        labels: [num_kv_heads, seq_len] int.
        nlist: number of clusters.

    Returns:
        cluster_size:     [num_kv_heads, nlist]       int32.
        cluster_size_ps:  [num_kv_heads, nlist]       int32 (cumsum).
        sorted_idx:       [num_kv_heads, seq_len]     int32 (argsort of labels).
    """
    num_kv_heads, seq_len = labels.shape
    device = labels.device
    dtype_i = torch.int32

    # bincount per head: one-hot scatter_add into a [Hk, nlist] buffer.
    cluster_size = torch.zeros(num_kv_heads, nlist, device=device, dtype=dtype_i)
    ones = torch.ones(num_kv_heads, seq_len, device=device, dtype=dtype_i)
    cluster_size.scatter_add_(1, labels.to(dtype_i), ones)

    cluster_size_ps = torch.cumsum(cluster_size, dim=-1).to(dtype_i)
    sorted_idx = torch.argsort(labels, dim=-1).to(dtype_i)

    return cluster_size, cluster_size_ps, sorted_idx


def pytorch_topk_cluster_selection(
    query: torch.Tensor,
    centroids: torch.Tensor,
    cluster_size: torch.Tensor,
    cluster_key_indices: torch.Tensor,
    num_kv_groups: int,
    sink: int,
    sel_budget: int,
    max_seq_len: int,
) -> torch.Tensor:
    """Select `sel_budget` token indices per query head.

    Replaces `_kernels.get_neigh_c` + `_kernels.get_sel_indices` +
    the `sel_token_indices += controller.sink` line in
    `clusterkv/clusterkv_utils/__init__.py:update_sel_indices`.

    Strategy: for each query head, score all clusters via cosine sim, take
    the top-k clusters, and gather the tokens of those clusters into a flat
    `[num_heads, sel_budget]` buffer. Tokens are absolute cache positions
    in `[sink, max_seq_len)` once we add the sink offset at the end.

    Args:
        query:                [1, num_heads, 1, head_dim]  (already RoPE'd).
        centroids:            [num_kv_heads, total_nlist, head_dim].
        cluster_size:         [num_kv_heads, total_nlist]  int32.
        cluster_key_indices:  [num_kv_heads, max_seq_len]  int32.
        num_kv_groups:        num_heads // num_kv_heads.
        sink:                 sink size (added to all selected indices).
        sel_budget:           token_budget - 1.
        max_seq_len:          max sequence length (used as the padding sentinel).

    Returns:
        sel_token_indices: [num_heads, sel_budget]  int32, padded with
            `max_seq_len + 1` for unused slots.
    """
    num_heads, head_dim = query.shape[1], query.shape[-1]
    num_kv_heads, total_nlist = centroids.shape[0], centroids.shape[1]
    device = query.device
    eps = 1e-8

    # Per-query-head cluster slice via GQA mapping.
    kv_idx_for_head = (
        torch.arange(num_heads, device=device, dtype=torch.int64) // num_kv_groups
    )
    # Broadcast kv-head fields to [num_heads, ...]
    centroids_for_head = centroids[kv_idx_for_head]  # [num_heads, total_nlist, D]
    cluster_size_for_head = cluster_size[kv_idx_for_head]  # [num_heads, total_nlist]
    cluster_key_indices_for_head = cluster_key_indices[
        kv_idx_for_head
    ]  # [num_heads, max_seq_len]

    # Cosine similarity per head.
    q_flat = query.squeeze(0).squeeze(1)  # [num_heads, D]
    q_norm = F.normalize(q_flat.float(), dim=-1, eps=eps)  # [num_heads, D]
    c_norm = F.normalize(centroids_for_head.float(), dim=-1, eps=eps)  # [num_heads, nlist, D]
    # Per-head matmul: q_norm.unsqueeze(1) is [num_heads, 1, D] and broadcasts
    # against c_norm.transpose(1, 2) which is [num_heads, D, nlist] ->
    # result is [num_heads, 1, nlist] -> squeeze to [num_heads, nlist].
    scores = torch.matmul(q_norm.unsqueeze(1), c_norm.transpose(1, 2)).squeeze(1)  # [num_heads, total_nlist]

    # Pick the smallest k such that cumulative size >= sel_budget. To keep
    # things simple and match the original's "take top clusters until budget
    # is filled" we just take the top-k=sel_budget clusters; the gather
    # stage will pad with sentinel where clusters are short.
    k = min(sel_budget, total_nlist)
    topk_clusters = scores.topk(k=k, dim=-1).indices.to(torch.int32)  # [num_heads, k]

    # Per-cluster start/end offsets in `cluster_key_indices_for_head`.
    cluster_size_ps_for_head = torch.cumsum(cluster_size_for_head, dim=-1)
    starts = cluster_size_ps_for_head.gather(1, topk_clusters) - cluster_size_for_head.gather(
        1, topk_clusters
    )
    ends = cluster_size_ps_for_head.gather(1, topk_clusters)
    cluster_sizes = (ends - starts).clamp_min(0)  # [num_heads, k]

    # Pack into a fixed [num_heads, sel_budget] buffer independently per head.
    # This avoids sentinel holes when the selected clusters have different
    # sizes on different heads.
    sel = torch.full(
        (num_heads, sel_budget), max_seq_len + 1, dtype=torch.int32, device=device
    )
    for h in range(num_heads):
        out_idx = 0
        for j in range(k):
            if out_idx >= sel_budget:
                break
            size_j = int(cluster_sizes[h, j].item())
            if size_j <= 0:
                continue
            start_j = int(starts[h, j].item())
            take = min(size_j, sel_budget - out_idx)
            sel[h, out_idx:out_idx + take] = cluster_key_indices_for_head[
                h, start_j:start_j + take
            ].to(torch.int32)
            out_idx += take

        if out_idx == 0:
            sel[h].fill_(0)
        elif out_idx < sel_budget:
            sel[h, out_idx:].fill_(sel[h, out_idx - 1])

    # Add sink offset so indices are absolute cache positions.
    sel = sel + sink
    return sel


def pytorch_full_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    is_causal: bool = True,
) -> torch.Tensor:
    """FlashAttention-2 over a (full) cache.

    Replaces `flashinfer.single_prefill_with_kv_cache` and the full-attention
    branch of `BatchDecodeWithPagedKVCacheWrapper.forward`.

    Args:
        query_states:  [bsz, q_len, num_heads, head_dim]   (BSND).
        key_states:    [bsz, kv_len, num_kv_heads, head_dim] (BSND).
        value_states:  same shape as key_states.
        is_causal:     whether to apply a causal mask.

    Returns:
        attn_output: [bsz, q_len, num_heads, head_dim].
    """
    # flash_attn 2.7's flash_attn_func requires num_kv_heads == num_heads
    # (no native GQA), so we expand the kv heads via repeat_interleave.
    num_heads = query_states.shape[2]
    num_kv_heads = key_states.shape[2]
    n_rep = num_heads // num_kv_heads
    # if n_rep > 1:
    #     key_states = key_states.repeat_interleave(n_rep, dim=2)
    #     value_states = value_states.repeat_interleave(n_rep, dim=2)
    return flash_attn_func(
        query_states,
        key_states,
        value_states,
        causal=is_causal,
    )


def cache_nhd_to_bsnd(cache_states: torch.Tensor) -> torch.Tensor:
    """Convert native cache layout [num_kv_heads, seq_len, head_dim] to BSND."""
    return cache_states.transpose(0, 1).unsqueeze(0).contiguous()


def pytorch_sparse_decode_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    """FlashAttention-2 over a pre-selected K/V block.

    Replaces the `decode_sparse_attn(..., topk_indices)` path.

    Args:
        query_states:  [bsz, 1, num_heads, head_dim]            (BSND).
        key_states:    [bsz, sel_len, num_heads, head_dim]      (BSND, sink|sel|win).
        value_states:  same shape as key_states.

    Returns:
        attn_output: [bsz, 1, num_heads, head_dim].
    """
    # `is_causal=False` because the only query token attends to the entire
    # selected sequence (no future tokens to mask — we already chose them).
    return flash_attn_func(
        query_states,
        key_states,
        value_states,
        causal=False,
    )


# ---------------------------------------------------------------------------
# Native cache (grows via torch.cat, replaces the pre-allocated GPU pool)
# ---------------------------------------------------------------------------


class NativeKVCache:
    """A simple growing K/V cache: per-layer [num_kv_heads, seq_len, head_dim]."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        # We use empty zero-length tensors so the first append does a single
        # cat from 0 -> q_len (cheap; no large allocation up front).
        self.key_cache: List[torch.Tensor] = [
            torch.empty(0, num_kv_heads, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.value_cache: List[torch.Tensor] = [
            torch.empty(0, num_kv_heads, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

    def append(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """new_k, new_v: [num_kv_heads, q_len, head_dim]."""
        if self.key_cache[layer_idx].numel() == 0:
            # First append: just assign (avoids cat with a 0-length tensor,
            # which torch.cat cannot infer the shape of correctly).
            self.key_cache[layer_idx] = new_k.contiguous()
            self.value_cache[layer_idx] = new_v.contiguous()
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], new_k], dim=1
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], new_v], dim=1
            )

    def get_seq_len(self, layer_idx: int = 0) -> int:
        return int(self.key_cache[layer_idx].shape[1])

    def get_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_window(
        self, layer_idx: int, start: int, end: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache[layer_idx][:, start:end, :], self.value_cache[layer_idx][
            :, start:end, :
        ]

    def gather(
        self,
        layer_idx: int,
        sel_idx_per_head: torch.Tensor,
        num_heads: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather the selected K/V per head and broadcast to num_heads.

        Args:
            sel_idx_per_head: [num_heads, K] int32 (absolute cache positions).

        Returns:
            K_g, V_g: [1, K, num_heads, head_dim]  (BSND).
        """
        k, v = self.get_kv(layer_idx)  # [Hk, S, D]
        n_rep = num_heads // self.num_kv_heads
        if n_rep > 1:
            # sel_idx_per_head is per-query-head; map to per-kv-head slice
            # then expand back via repeat_interleave.
            kv_idx = sel_idx_per_head[::n_rep]  # [Hk, K]
            kv_idx_long = kv_idx.to(torch.int64).unsqueeze(-1).expand(
                -1, -1, self.head_dim
            )
            # Result: [Hk, K, D] -> repeat to [num_heads, K, D].
            k_g = k.gather(1, kv_idx_long).repeat_interleave(n_rep, dim=0)
            v_g = v.gather(1, kv_idx_long).repeat_interleave(n_rep, dim=0)
        else:
            idx_long = sel_idx_per_head.to(torch.int64).unsqueeze(-1).expand(
                -1, -1, self.head_dim
            )
            k_g = k.gather(1, idx_long)
            v_g = v.gather(1, idx_long)
        # Add batch dim and reorder to BSND: [1, K, num_heads, D].
        return (
            k_g.transpose(0, 1).unsqueeze(0).contiguous(),
            v_g.transpose(0, 1).unsqueeze(0).contiguous(),
        )

    def clear(self) -> None:
        for i in range(self.num_layers):
            self.key_cache[i] = self.key_cache[i][:, :0, :]
            self.value_cache[i] = self.value_cache[i][:, :0, :]


class OriginalKVCacheAdapter:
    """Adapter over ClusterKVController.kv_cache using the original append kernels."""

    def __init__(self, controller: ClusterKVController):
        self.controller = controller
        self.num_kv_heads = controller.num_kv_heads
        self.num_heads = controller.num_heads
        self.head_dim = controller.head_dim

    def append(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        # myllama produces [Hk, q_len, D]; original append_kv expects [q_len, Hk, D].
        append_kv(
            new_k.transpose(0, 1).contiguous(),
            new_v.transpose(0, 1).contiguous(),
            self.controller,
            layer_idx,
        )

    def get_seq_len(self, layer_idx: int = 0) -> int:
        return self.controller.kv_seqlen

    def get_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        k, v = self.controller.get_kv(layer_idx)  # [S, 1, Hk, D]
        return (
            k.squeeze(1).transpose(0, 1).contiguous(),
            v.squeeze(1).transpose(0, 1).contiguous(),
        )

    def get_window(self, layer_idx: int, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        k = self.controller.kv_cache[layer_idx][start:end, 0, 0, ...]
        v = self.controller.kv_cache[layer_idx][start:end, 1, 0, ...]
        return k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous()

    def gather(
        self,
        layer_idx: int,
        sel_idx_per_head: torch.Tensor,
        num_heads: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k, v = self.get_kv(layer_idx)
        n_rep = num_heads // self.num_kv_heads
        if n_rep > 1:
            kv_idx = sel_idx_per_head[::n_rep]
            kv_idx_long = kv_idx.to(torch.int64).unsqueeze(-1).expand(-1, -1, self.head_dim)
            k_g = k.gather(1, kv_idx_long).repeat_interleave(n_rep, dim=0)
            v_g = v.gather(1, kv_idx_long).repeat_interleave(n_rep, dim=0)
        else:
            idx_long = sel_idx_per_head.to(torch.int64).unsqueeze(-1).expand(-1, -1, self.head_dim)
            k_g = k.gather(1, idx_long)
            v_g = v.gather(1, idx_long)
        return (
            k_g.transpose(0, 1).unsqueeze(0).contiguous(),
            v_g.transpose(0, 1).unsqueeze(0).contiguous(),
        )

    def clear(self) -> None:
        self.controller.clean_states()


# ---------------------------------------------------------------------------
# Clustering state (pure PyTorch; replaces the metadata on ClusterKVController)
# ---------------------------------------------------------------------------


class PyTorchClusterState:
    """Per-layer clustering metadata + the working buffers for the sliding
    window and top-k selection.

    Mirrors the relevant fields of `clusterkv.clusterkv_utils.clusterkv_controller.ClusterKVController`.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        nlist: int,
        token_budget: int,
        sink: int,
        window: int,
        window_nlist: int,
        device: torch.device,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.nlist = nlist
        self.token_budget = token_budget
        self.sink = sink
        self.window = window
        self.window_nlist = window_nlist
        self.device = device

        # Per-layer cluster metadata (lazy until first build).
        self.centroids: List[Optional[torch.Tensor]] = [None] * num_layers
        self.cluster_size: List[Optional[torch.Tensor]] = [None] * num_layers
        self.cluster_size_ps: List[Optional[torch.Tensor]] = [None] * num_layers
        self.cluster_key_indices: List[Optional[torch.Tensor]] = [None] * num_layers

        # Working buffers.
        self.labels = torch.full(
            (num_kv_heads, max_seq_len), -1, dtype=torch.int32, device=device
        )
        self.sel_token_indices = torch.full(
            (num_heads, max(1, token_budget - 1)),
            max_seq_len + 1,
            dtype=torch.int32,
            device=device,
        )
        self.neigh_c = torch.zeros((num_heads, nlist), dtype=torch.int32, device=device)
        self.neigh_c_size = torch.zeros((num_heads, nlist), dtype=torch.int32, device=device)
        # The last position in the cache (incremented by `prepare_metadata`).
        self.kv_seqlen = 0
        # The prompt length (set on prefill).
        self.prompt_len = 0
        # The sliding window cursor (absolute position of the first token
        # in the current window).
        self.win_offset = 0

    @property
    def generated_len(self) -> int:
        return self.kv_seqlen - self.prompt_len

    @property
    def cur_win_size(self) -> int:
        if self.generated_len == 0:
            return 0
        return (self.generated_len % self.window) or self.window

    def reset(self) -> None:
        """Clear all clustering state — used by clusterkv_clear()."""
        self.centroids = [None] * self.num_layers
        self.cluster_size = [None] * self.num_layers
        self.cluster_size_ps = [None] * self.num_layers
        self.cluster_key_indices = [None] * self.num_layers
        self.labels.fill_(-1)
        self.sel_token_indices.fill_(self.max_seq_len + 1)
        self.neigh_c = torch.zeros(
            (self.num_heads, self.nlist), dtype=torch.int32, device=self.device
        )
        self.neigh_c_size = torch.zeros(
            (self.num_heads, self.nlist), dtype=torch.int32, device=self.device
        )
        self.kv_seqlen = 0
        self.prompt_len = 0
        self.win_offset = 0


# ---------------------------------------------------------------------------
# Decoder layer (QKV+RoPE+flash-attn inline so we can splice in cache + sparse path)
# ---------------------------------------------------------------------------


class MyLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.num_heads
        )
        self.layer_idx = layer_idx
        self.is_causal = True

        self.self_attn = ClusterKVAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def _mlp_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        chunk_size = int(os.environ.get("CLUSTERKV_MLP_CHUNK_SIZE", "8192"))
        if self.training or chunk_size <= 0 or hidden_states.shape[1] <= chunk_size:
            return self.mlp(hidden_states)
        return torch.cat(
            [
                self.mlp(hidden_states[:, start:start + chunk_size, :])
                for start in range(0, hidden_states.shape[1], chunk_size)
            ],
            dim=1,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        model_ref: "MyLlamaModel",
        attention_mask: Optional[torch.Tensor] = None,  # accepted, ignored
        past_key_value=None,  # accepted, ignored (we use model_ref.cache)
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        bsz, q_len, _ = hidden_states.shape
        assert bsz == 1, "MyLlamaDecoderLayer only supports batch size 1."
        prefix = "prefill" if q_len > 1 else "decode"

        residual = hidden_states
        stage_begin(f"{prefix}_pre_ffn", self.layer_idx)
        hidden_states = self.input_layernorm(hidden_states)

        attn_output, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            controller=model_ref.controller,
            use_flash_attn_clusterkv=True,
            skip_layer=model_ref._skip_layer,
            token_budget=model_ref._token_budget,
            full_mode=model_ref._full,
        )

        hidden_states = residual + attn_output

        # --- MLP ---
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self._mlp_forward(hidden_states)
        hidden_states = residual + hidden_states
        stage_end(f"{prefix}_post_ffn", self.layer_idx)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

    def _build_clusters(
        self,
        model_ref: "MyLlamaModel",
        new_k: torch.Tensor,
        prefill: bool,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Run k-means on the new keys and update cluster state for this layer.

        Args:
            new_k: [num_kv_heads, q_len, head_dim] (already RoPE'd).
            prefill: if True, cluster the entire new range. If False (decode),
                cluster only the last `window` tokens and use `window_nlist`
                clusters, with the key_offset shifted so absolute positions
                are preserved.
        """
        state = model_ref._cluster
        sink = state.sink
        window = state.window

        if prefill:
            keys_for_cluster = new_k[:, sink:, :].transpose(0, 1).contiguous()
            key_offset = 0
            nlist = model_ref._nlist
        else:
            if state.offload and self.layer_idx >= 2:
                keys_for_cluster = state.get_app_k_clustering(self.layer_idx).contiguous()
                key_offset = state.kv_seqlen - sink - window
            else:
                k_full, _ = model_ref.cache.get_kv(self.layer_idx)
                start = state.kv_seqlen - 1 - window  # current position is 1 token past end of window
                if start < sink:
                    start = sink
                keys_for_cluster = k_full[:, start:state.kv_seqlen - 1, :].transpose(0, 1).contiguous()
                key_offset = start - sink
            nlist = model_ref._window_nlist

        if keys_for_cluster.shape[0] == 0:
            return

        if stream is None:
            stream = torch.cuda.current_stream(keys_for_cluster.device)

        build_cluster(
            state,
            self.layer_idx,
            keys_for_cluster,
            key_offset,
            nlist,
            stream,
        )

    def _select_sparse_indices(
        self,
        q: torch.Tensor,
        state: ClusterKVController,
    ) -> torch.Tensor:
        """Run ClusterKV's native selection kernels and return absolute ids."""
        state.sel_token_indices.fill_(state.max_seq_len + 1)
        q_for_sel = q.squeeze(0)  # [1, num_heads, D], original kernel layout.
        _kernels.get_neigh_c(
            q_for_sel,
            state.centroids[self.layer_idx],
            state.cluster_size[self.layer_idx],
            state.neigh_c,
            state.neigh_c_size,
        )
        _kernels.get_sel_indices(
            state.neigh_c,
            state.neigh_c_size,
            state.cluster_size_ps[self.layer_idx],
            state.cluster_key_indices[self.layer_idx],
            state.sel_token_indices,
        )
        sel = state.sel_token_indices + state.sink
        invalid = (sel < 0) | (sel >= state.kv_seqlen)
        if os.environ.get("CLUSTERKV_DEBUG_RECALL") == "1":
            sorted_sel = torch.sort(sel, dim=1).values
            dup_count = (sorted_sel[:, 1:] == sorted_sel[:, :-1]).sum().item()
            print(
                "[clusterkv-debug] "
                f"layer={self.layer_idx} generated={state.generated_len} "
                f"kv={state.kv_seqlen} sel_min={sel.min().item()} "
                f"sel_max={sel.max().item()} invalid={invalid.sum().item()} "
                f"dup={dup_count}",
                flush=True,
            )
        if state.offload and self.layer_idx >= 2:
            replacement = state.g2c[self.layer_idx]
        else:
            replacement = torch.zeros_like(sel)
        return torch.where(invalid, replacement, sel).to(torch.int32).contiguous()

    def _gather_with_sink_window(
        self,
        model_ref: "MyLlamaModel",
        sel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the K/V tensor for sparse decode: [sink | sel | window].

        Args:
            sel: [num_heads, B-1] int32, absolute cache positions (with sink
                already added).

        Returns:
            K_g, V_g: [1, sink + (B-1) + window, num_heads, head_dim]  (BSND).
        """
        state = model_ref._cluster
        sink = state.sink
        seq_len = state.kv_seqlen

        if state.offload and self.layer_idx >= 2:
            # Compact offload cache layout matches ClusterKVAttention:
            # [sink | token_budget compact slots | current window].
            total_len = state.sink + state._token_budget + state.cur_win_size
            k_compact = state.kv_cache[self.layer_idx][:total_len, 0, 0, ...]
            v_compact = state.kv_cache[self.layer_idx][:total_len, 1, 0, ...]
            return (
                k_compact.unsqueeze(0).contiguous(),
                v_compact.unsqueeze(0).contiguous(),
            )

        # 1. Selected tokens: gather from cache, shape [1, B-1, num_heads, D].
        k_sel, v_sel = model_ref.cache.gather(
            self.layer_idx, sel, num_heads=self.num_heads
        )
        # k_sel, v_sel are [1, B-1, num_heads, D] in BSND layout already.

        # 2. Sink: [Hk, sink, D] -> [1, sink, num_heads, D] (BSND).
        k_sink, v_sink = model_ref.cache.get_window(self.layer_idx, 0, sink)
        k_sink = cache_nhd_to_bsnd(k_sink).repeat_interleave(
            self.num_key_value_groups, dim=2
        )
        v_sink = cache_nhd_to_bsnd(v_sink).repeat_interleave(
            self.num_key_value_groups, dim=2
        )

        # 3. Sliding window: match ClusterKVController.cur_win_indices length.
        cur_win_size = state.cur_win_size
        win_start = max(sink, seq_len - cur_win_size)
        k_win, v_win = model_ref.cache.get_window(self.layer_idx, win_start, seq_len)
        k_win = cache_nhd_to_bsnd(k_win).repeat_interleave(
            self.num_key_value_groups, dim=2
        )
        v_win = cache_nhd_to_bsnd(v_win).repeat_interleave(
            self.num_key_value_groups, dim=2
        )

        k_full = torch.cat([k_sink, k_sel, k_win], dim=1)  # cat along seq dim
        v_full = torch.cat([v_sink, v_sel, v_win], dim=1)
        return k_full, v_full


# ---------------------------------------------------------------------------
# LlamaModel: owns the shared rotary embedding, the cache, and the layer loop
# ---------------------------------------------------------------------------


class MyLlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // self.num_heads
        )

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [MyLlamaDecoderLayer(config, i) for i in range(self.num_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Shared rotary embedding (one instance, used by every layer).
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        # The original ClusterKV controller/cache — allocated by clusterkv_init.
        self.controller: Optional[ClusterKVController] = None
        self.cache: Optional[OriginalKVCacheAdapter] = None
        self._cluster: Optional[ClusterKVController] = None

        # Defaults (overwritten by clusterkv_init).
        self._skip_layer: int = 0
        self._token_budget: int = 0
        self._nlist: int = 0
        self._niter: int = 0
        self._window_nlist: int = 0
        self._full: bool = False

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing.
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.cache is None:
            raise RuntimeError(
                "NativeKVCache not initialized. Call clusterkv_init() first."
            )

        past_seen_tokens = self.cache.get_seq_len(0)
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + seq_length,
                dtype=torch.long,
                device=(input_ids.device if input_ids is not None else inputs_embeds.device),
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Compute RoPE (cos, sin) once and share with all layers.
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        assert self.controller is not None, "Please init ClusterKVController first."
        self.controller.prepare_metadata(seq_length)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        if self._skip_layer > 0:
            self.controller.set_token_budget(self.controller._max_page_limit)
            self.controller.prepare_append_metadata(seq_length)
        else:
            self.controller.set_token_budget(self._token_budget)
            self.controller.prepare_append_metadata(seq_length)

        for idx, decoder_layer in enumerate(self.layers):
            if idx == self._skip_layer:
                self.controller.set_token_budget(self._token_budget)
                self.controller.prepare_append_metadata(seq_length, updateTensor=(idx == 0))

            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                model_ref=self,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # ClusterKV owns the cache internally. Returning HF-style past_key_values
        # would duplicate the full/compact KV tensors every generation step.
        pkv = None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, pkv, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=pkv,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# ---------------------------------------------------------------------------
# LlamaForCausalLM (entry point with the same external API as the original)
# ---------------------------------------------------------------------------


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = MyLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing.
        self.post_init()

    # ---------- ClusterKV-style init/clear (identical signature) ----------

    def clusterkv_init(
        self,
        nlist: int,
        niter: int,
        max_seq_len: int,
        token_budget: int = 512,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda:0"),
        full: bool = False,
        sink: int = 16,
        window: int = 320,
        window_nlist: int = 8,
        offload: bool = False,
    ) -> None:
        """Allocate the native KV cache and the cluster state. Call this
        once after `from_pretrained` and before the first `forward`."""
        self.model._nlist = nlist
        self.model._niter = niter
        self.model._window_nlist = window_nlist
        self.model._token_budget = token_budget
        self.model._full = full
        # Skip-first-2-layers when in sparse mode; if full=True, skip none.
        self.model._skip_layer = 0 if full else 2

        controller = ClusterKVController(
            num_layers=self.model.num_layers,
            num_heads=self.model.num_heads,
            num_kv_heads=self.model.num_key_value_heads,
            head_dim=self.model.head_dim,
            nlist=nlist,
            niter=niter,
            token_budget=token_budget,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=device,
            full=full,
            sink=sink,
            window=window,
            window_nlist=window_nlist,
            offload=offload,
        )
        self.model.controller = controller
        self.model._cluster = controller
        self.model.cache = OriginalKVCacheAdapter(controller)

        print(
            f"MyLlamaForCausalLM: ClusterKVController cache allocated (max_seq_len={max_seq_len}, "
            f"token_budget={token_budget}, full={full}, offload={offload}, "
            f"skip_layer={self.model._skip_layer})"
        )

    def clusterkv_clear(self) -> None:
        """Reset the cache and cluster state for a new conversation."""
        assert self.model.cache is not None, "Must be called after clusterkv_init()."
        self.model.cache.clear()

    # ---------- Standard HF API ----------

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute logits for the last `num_logits_to_keep` tokens. We
        # default to 1 (the last token), matching the original clusterkv
        # LlamaForCausalLM and matching what bench_textgen.py expects.
        if num_logits_to_keep == 0:
            num_logits_to_keep = 1
        slice_indices = slice(-num_logits_to_keep, None) if num_logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :]).float()

        loss = None
        if labels is not None:
            # Upcast to float for the loss.
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.to(shift_logits.device).view(-1)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        has_past = (
            self.model.cache is not None
            and self.model.cache.get_seq_len(0) > 0
        )

        # For generation, slice only after ClusterKV has seen the prompt.
        if has_past:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if has_past:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        if cache_position is not None and has_past:
            cache_position = cache_position[-input_ids.shape[1]:]

        if inputs_embeds is not None and not has_past:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # Beam search is not supported (matches the original).
        raise NotImplementedError(
            "Beam search is not supported by MyLlamaForCausalLM."
        )
