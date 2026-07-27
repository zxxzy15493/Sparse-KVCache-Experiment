# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

from ..ops.pit_sparse_flash_attention_v2 import vertical_slash_sparse_attention
# from .kvcompression import *
import torch



last_q = 64
arange = torch.arange(last_q, device="cuda:0")
LAST_Q_MASK = arange[None, None, :, None] >= arange[None, None, None, :]

def sum_all_diagonal_matrix(mat: torch.tensor):
    b, h, n, m = mat.shape
    zero_mat = torch.zeros((b, h, n, n)).to(mat.device) # Zero matrix used for padding
    mat_padded =  torch.cat((zero_mat, mat, zero_mat), -1) # pads the matrix on left and right
    mat_strided = mat_padded.as_strided((1, 1, n, n + m), (1, n * (2 * n + m), 2 * n + m + 1, 1)) # Change the strides
    sum_diags = torch.sum(mat_strided, 2) # Sums the resulting matrix's columns
    return sum_diags[:,:,1:]

def minference_prefill_kernel(
    q, k, v, head_id, layer_idx,
    best_pattern
):
    head_dim = q.size(-1)
    def vertical_and_slash_kernel(q, k, v, vertical_size, slash_size):
        vertical_size, slash_size  = min(q_len, max(vertical_size, 30)), min(q_len, max(slash_size, 50))
        last_q = min(64, q_len)
        qk = torch.einsum(f'bhmk, bhnk -> bhmn', q[:,:,-last_q:,:], k) / math.sqrt(head_dim)
        qk[:, :, :, -last_q:] = torch.where(LAST_Q_MASK[...,-last_q:,-last_q:].to(q.device), qk[:, :, :, -last_q:], -torch.inf)
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
        vertical = qk.sum(-2, keepdim=True)
        vertical[...,:30] = torch.inf
        vertical_topk = torch.topk(vertical, vertical_size, -1).indices

        slash = sum_all_diagonal_matrix(qk)[...,:-last_q + 1]
        slash[...,-100:] = torch.inf
        slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices

        return vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)


    q_len = q.shape[2]
    ty, vertical_size, slash_size, _ = best_pattern[layer_idx].get(str(head_id), ("vertical_and_slash", 1000, 6096, 1))

 
    return vertical_and_slash_kernel(q, k, v, vertical_size, slash_size)

def minference_prefill_forward(
    query_states, key_states, value_states, layer_idx,
    best_pattern, 
):
    bsz, head_num, q_len, head_dim = query_states.shape
    kv_head_num = key_states.shape[1]
    kv_group_size = head_num // kv_head_num
    output = torch.empty_like(query_states)
    # [bsz, head_num, q_len, head_dim] = query_states.shape
    # print(f"key_states shape: {key_states.shape}, query_states shape: {query_states.shape}")

    for bdx in range(bsz):
        for head in range(query_states.size(1)):
            group = head // kv_group_size
            q = query_states[bdx:bdx+1, head, :, :].unsqueeze(1)    # [1, 1, q_len, head_dim]
            k = key_states[bdx:bdx+1, group, :, :].unsqueeze(1)     # [1, 1, q_len, head_dim]
            v = value_states[bdx:bdx+1, group, :, :].unsqueeze(1)   # [1, 1, q_len, head_dim]

            attn_output = minference_prefill_kernel(q, k, v, head, layer_idx, best_pattern)
            output[bdx, head:head+1, :, :] = attn_output

    return output


