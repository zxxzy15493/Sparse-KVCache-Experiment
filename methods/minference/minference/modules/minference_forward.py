# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

from ..ops.pit_sparse_flash_attention_v2 import vertical_slash_sparse_attention
# from .kvcompression import *
import torch


LAST_Q_MASKS = {}


def _last_q_mask(device, length):
    """Return a causal mask without pinning imports to cuda:0.

    Qwen3-32B is often sharded over multiple GPUs.  Constructing this tensor
    eagerly on cuda:0 made the module import device-dependent and could cause
    cross-device errors for layers placed on another GPU.
    """
    key = (device, length)
    mask = LAST_Q_MASKS.get(key)
    if mask is None:
        positions = torch.arange(length, device=device)
        mask = positions[None, None, :, None] >= positions[None, None, None, :]
        LAST_Q_MASKS[key] = mask
    return mask

def sum_all_diagonal_matrix(mat: torch.Tensor):
    b, h, n, m = mat.shape
    zero_mat = mat.new_zeros((b, h, n, n)) # Zero matrix used for padding
    mat_padded =  torch.cat((zero_mat, mat, zero_mat), -1) # pads the matrix on left and right
    stride_b, stride_h, stride_row, stride_col = mat_padded.stride()
    mat_strided = mat_padded.as_strided(
        (b, h, n, n + m),
        (stride_b, stride_h, stride_row + stride_col, stride_col),
    )
    sum_diags = torch.sum(mat_strided, 2) # Sums the resulting matrix's columns
    return sum_diags[:,:,1:]

def minference_prefill_kernel(
    q, k, v, head_id, layer_idx,
    best_pattern, seqlens=None,
):
    head_dim = q.size(-1)
    def vertical_and_slash_kernel(q, k, v, vertical_size, slash_size):
        vertical_size, slash_size  = min(q_len, max(vertical_size, 30)), min(q_len, max(slash_size, 50))
        last_q = min(64, q_len)
        if seqlens is None:
            qk = torch.einsum('bhmk,bhnk->bhmn', q[:, :, -last_q:, :], k) / math.sqrt(head_dim)
            qk[:, :, :, -last_q:] = torch.where(
                _last_q_mask(q.device, last_q), qk[:, :, :, -last_q:], -torch.inf
            )
        else:
            # Q/K/V have their valid tokens at the start of each row. Select
            # each sample's last queries, rather than queries from its padding.
            lengths = seqlens.to(torch.long).view(-1, 1, 1, 1)
            query_positions = (lengths - last_q).clamp_min(0) + torch.arange(last_q, device=q.device).view(1, 1, -1, 1)
            last_queries = q.gather(2, query_positions.clamp_max(q_len - 1).expand(-1, q.size(1), -1, head_dim))
            key_positions = torch.arange(q_len, device=q.device).view(1, 1, 1, -1)
            qk = torch.einsum('bhmk,bhnk->bhmn', last_queries, k) / math.sqrt(head_dim)
            qk.masked_fill_((key_positions >= lengths) | (key_positions > query_positions), -torch.inf)
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
        if seqlens is not None:
            # This also clears NaNs from completely empty samples.
            qk.masked_fill_(query_positions >= lengths, 0)
        vertical = qk.sum(-2, keepdim=True)
        vertical[...,:30] = torch.inf
        if seqlens is not None:
            vertical.masked_fill_(key_positions >= lengths, -torch.inf)
        vertical_topk = torch.topk(vertical, vertical_size, -1).indices

        diagonals = sum_all_diagonal_matrix(qk)
        if seqlens is None:
            slash = diagonals[..., :q_len]
            slash[..., -100:] = torch.inf
        else:
            positions = key_positions.squeeze(-2)
            row_lengths = lengths.squeeze(-2)
            # When a sample has fewer than last_q queries, remove the extra
            # leading diagonals introduced by the common query-window size.
            diagonal_positions = (positions + (last_q - row_lengths).clamp_min(0)).clamp_max(diagonals.size(-1) - 1)
            slash = diagonals.gather(-1, diagonal_positions.expand(-1, q.size(1), -1))
            slash.masked_fill_(positions >= (row_lengths - 100).clamp_min(0), torch.inf)
            slash.masked_fill_(positions >= row_lengths, -torch.inf)
        slash_size = min(slash_size, slash.shape[-1])
        slash_topk = torch.topk(slash, slash_size, -1).indices
        if seqlens is None:
            slash = (q_len - 1) - slash_topk
        else:
            # The converters take rectangular index tensors. Extra vertical
            # entries must be beyond every query block; duplicate slash zero
            # entries merge into the same causal block in the converters.
            vertical_topk.masked_fill_(vertical_topk >= lengths, torch.iinfo(torch.int32).max)
            slash = (row_lengths - 1) - slash_topk
            slash.masked_fill_(slash_topk >= row_lengths, 0)

        return vertical_slash_sparse_attention(q, k, v, vertical_topk, slash, seqlens=seqlens)


    q_len = q.shape[2]
    ty, vertical_size, slash_size, _ = best_pattern[layer_idx].get(str(head_id), ("vertical_and_slash", 1000, 6096, 1))

 
    return vertical_and_slash_kernel(q, k, v, vertical_size, slash_size)


def minference_prefill_forward(
    query_states, key_states, value_states, layer_idx,
    best_pattern, attention_mask=None,
):
    """Run each head over the whole batch, including unequal left-padded rows."""
    bsz, head_num, q_len, head_dim = query_states.shape
    kv_head_num = key_states.shape[1]
    kv_group_size = head_num // kv_head_num
    output = torch.empty_like(query_states)
    if q_len == 0:
        return output
    seqlens = None
    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape[0] != bsz or attention_mask.shape[1] < q_len:
            raise ValueError("MInference prefill requires a 2D left-padding attention mask")
        mask = attention_mask[:, -q_len:].to(device=query_states.device, dtype=torch.bool)
        seqlens = mask.sum(-1, dtype=torch.int32)
        positions = torch.arange(q_len, device=query_states.device).view(1, 1, -1, 1)
        starts = (q_len - seqlens.to(torch.long)).view(-1, 1, 1, 1)
        compact_indices = (positions + starts).clamp_max(q_len - 1).expand(-1, 1, -1, head_dim)
        restore_indices = (positions - starts).clamp_min(0).expand(-1, 1, -1, head_dim)

    for head in range(head_num):
        group = head // kv_group_size
        q = query_states[:, head:head + 1]
        k = key_states[:, group:group + 1]
        v = value_states[:, group:group + 1]
        if seqlens is not None:
            q = q.gather(2, compact_indices)
            k = k.gather(2, compact_indices)
            v = v.gather(2, compact_indices)
        attn_output = minference_prefill_kernel(q, k, v, head, layer_idx, best_pattern, seqlens=seqlens)
        if seqlens is not None:
            attn_output = attn_output.gather(2, restore_indices)
            attn_output.masked_fill_(~mask[:, None, :, None], 0)
        output[:, head:head + 1] = attn_output
    return output
