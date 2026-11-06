"""Model-agnostic de-kernelized decode attention for AdaKV / HeadKV.

Replaces the varlen flash-attention kernel (``flash_attn_varlen_func``) used at
decode time with native PyTorch ``scaled_dot_product_attention``.

The algorithm (which KV tokens each head keeps, per-head length ``head_lens``)
is *unchanged*.  The flattened per-head cache (``DynamicCacheSplitHeadFlatten``)
stores one contiguous ``[total_tokens, head_dim]`` tensor, where head ``h`` owns
rows ``cu_klen[h] : cu_klen[h+1]``.  We rebuild a dense ``[num_heads, max_len,
head_dim]`` tensor (zero-padded every head to the longest head) and pass an
attention mask that only lets each head attend to the first ``head_lens[h]``
positions.  The output is elementwise-identical to the varlen kernel's; only the
wasted FLOPs differ (which is exactly what the ablation exposes).

This module is intentionally independent of any specific model family
(Qwen2/Qwen3/Llama/ChatGLM); the model hijacks only need to call
``sdpa_decode_attention`` instead of ``flash_attn_varlen_func``.
"""

import torch
import torch.nn.functional as F


def sdpa_decode_attention(query_states, key_states, value_states,
                          head_lens, cu_klen, bsz, num_heads, q_len, head_dim):
    """Dense SDPA replacement of the decode-phase varlen flash attention.

    Args:
        query_states: ``[bsz * num_heads, 1, head_dim]`` (the single new token).
        key_states:   ``[total_tokens, 1, head_dim]`` flattened per-head cache.
        value_states: same layout as key_states.
        head_lens: int32 tensor ``[bsz * num_heads]``; tokens kept per head.
        cu_klen:   int32 tensor ``[bsz * num_heads + 1]``; cumulative boundaries.
        bsz, num_heads, q_len, head_dim: standard attention dimensions.

    Returns:
        attn_output of shape ``[bsz, q_len, num_heads * head_dim]``.
    """
    flat_heads = int(head_lens.numel())
    max_len = int(head_lens.max().item())
    device = key_states.device
    dtype = key_states.dtype

    # flatten cache rows back out: [total, head_dim]
    k_flat = key_states.reshape(-1, head_dim)
    v_flat = value_states.reshape(-1, head_dim)

    # --- rebuild dense per-head KV, right-zero-padded to max_len ---
    dense_k = torch.zeros((flat_heads, max_len, head_dim), dtype=dtype, device=device)
    dense_v = torch.zeros_like(dense_k)

    cu = cu_klen.to(device=device).tolist()
    hl = head_lens.to(device=device).tolist()
    for h in range(flat_heads):
        l = int(hl[h])
        if l <= 0:
            continue
        s = int(cu[h])
        e = int(cu[h + 1])
        dense_k[h, :l] = k_flat[s:e]
        dense_v[h, :l] = v_flat[s:e]

    # --- boolean mask: head h may attend only to positions [0, head_lens[h]) ---
    allowed = (
        torch.arange(max_len, device=device).unsqueeze(0)
        < head_lens.to(device=device).unsqueeze(1)
    )
    attn_mask = allowed.unsqueeze(0).unsqueeze(2)  # [1, flat_heads, 1, max_len]

    # --- native dense SDPA ---
    q = query_states.reshape(flat_heads, 1, head_dim).unsqueeze(0)        # [1, flat_heads, 1, dim]
    k = dense_k.unsqueeze(0)                                              # [1, flat_heads, max_len, dim]
    v = dense_v.unsqueeze(0)
    attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
    attn = attn.squeeze(0)                                                # [flat_heads, 1, dim]

    # match the original output shape convention: [bsz, q_len, num_heads * head_dim]
    # flatten ordering is [batch, head] (row-major), so reshape(bsz, num_heads, ...)
    # then transpose(1, 2) keeps each batch's heads together in the hidden dim.
    return (
        attn.reshape(bsz, num_heads, q_len, head_dim)
        .transpose(1, 2)
        .reshape(bsz, q_len, num_heads * head_dim)
    )
