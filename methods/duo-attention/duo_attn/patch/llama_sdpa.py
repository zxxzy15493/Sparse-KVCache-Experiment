"""DuoAttention kernel-free variant (native SDPA comparison) — LLaMA model.

Algorithmically identical to ``llama_duo_attention_forward_one_way_reordered``:
same full/streaming head split, sink+recent window, tuple cache (full_KV, streaming_KV).
The only difference is replacing ``flash_attn_func`` / streaming-specific operators
with native ``F.scaled_dot_product_attention``.

- full heads:     full-causal SDPA, k/v from full cache (no truncation).
- streaming heads: full-causal SDPA, k/v from streaming cache (pre-truncated by sink+recent window).
- prefill:        full-block full-causal SDPA.

LLaMA uses GQA (num_key_value_heads < num_heads); the original flash_attn natively
handles grouped broadcasting.  SDPA requires expanding k/v heads to match query
heads (repeat_kv), producing element-wise equivalent results.
"""

from typing import List, Optional, Tuple, Union
import os
import torch
import torch.nn.functional as F
from torch import nn
import types

from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .utils import (
    reorder_linear_weights,
    reorder_full_attn_heads,
)
from .tuple_kv_cache import enable_tuple_kv_cache_for_llama


def _sdpa_prepare(q, k, v, num_query_heads):
    """Convert [bsz, seq, h, dim] to SDPA [bsz, h, seq, dim] layout, broadcast k/v heads by kv_groups.

    num_kv_heads must evenly divide num_query_heads (GQA).
    """
    # k/v: [bsz, seq, num_kv_heads, dim]
    kv_groups = num_query_heads // k.shape[2]
    k = repeat_kv(k.transpose(1, 2), kv_groups).contiguous()  # [bsz, num_kv_heads, seq, dim]
    v = repeat_kv(v.transpose(1, 2), kv_groups).contiguous()
    q = q.transpose(1, 2).contiguous()                        # [bsz, num_query_heads, q_seq, dim]
    return q, k, v


def sdpa_attn(query_states, key_states, value_states, num_query_heads, is_causal=True):
    """Model-agnostic kernel-free low-level helper: native SDPA attention.

    Accepts q/k/v in FlashAttention-style ``[bsz, seq, num_heads, dim]`` layout
    (fewer k/v heads for GQA), internally converts to SDPA layout with repeat_kv,
    returns ``[bsz, q_seq, num_query_heads, dim]``.
    Element-wise equivalent to ``flash_attn_func(..., causal=True)``; intended
    only for ablation (extra wasted FLOPs).
    """
    q, k, v = _sdpa_prepare(query_states, key_states, value_states, num_query_heads)
    attn_output = F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=is_causal
    )
    # [bsz, num_query_heads, q_seq, dim] -> [bsz, q_seq, num_query_heads, dim]
    return attn_output.transpose(1, 2)


def sdpa_attn_uniform_decode(
    query_states,
    full_key_states,
    full_value_states,
    streaming_key_states,
    streaming_value_states,
    num_full_query_head,
    num_streaming_query_head,
    sink_size,
    recent_size,
):
    """Decode-only uniform-form SDPA used for the non-uniform vs uniform ablation.

    The non-uniform DuoAttention form computes full heads over the whole sequence
    while streaming heads only compute over ``sink + recent`` tokens.  This
    function keeps exactly the same retained tokens but lays *every* head on the
    same full-length timeline: streaming K/V are placed with the sink at the head
    and the recent window at the tail, and the unused middle slots are masked
    out.  The math is unchanged (identical tokens, identical order), only the
    wasted FLOPs differ.  Padding is compute-only; the KV cache is untouched.

    Inputs use the FlashAttention layout ``[bsz, seq, num_kv_heads, head_dim]``
    for K/V and ``[bsz, seq, num_query_heads, head_dim]`` for Q.  Returns
    ``[bsz, q_len, num_query_heads, head_dim]``.
    """
    bsz = query_states.shape[0]
    full_len = full_key_states.shape[1]
    streaming_len = streaming_key_states.shape[1]
    num_streaming_kv_head = streaming_key_states.shape[2]
    device = query_states.device
    dtype = query_states.dtype

    # sink tokens live at the head of the timeline, recent tokens at the tail
    num_sink = min(int(sink_size), streaming_len)
    num_recent = streaming_len - num_sink

    padded_k = torch.zeros(
        (bsz, full_len, num_streaming_kv_head, streaming_key_states.shape[3]),
        dtype=dtype, device=device,
    )
    padded_v = torch.zeros(
        (bsz, full_len, num_streaming_kv_head, streaming_value_states.shape[3]),
        dtype=dtype, device=device,
    )
    if num_sink > 0:
        padded_k[:, :num_sink] = streaming_key_states[:, :num_sink]
        padded_v[:, :num_sink] = streaming_value_states[:, :num_sink]
    if num_recent > 0:
        padded_k[:, full_len - num_recent :] = streaming_key_states[:, num_sink:]
        padded_v[:, full_len - num_recent :] = streaming_value_states[:, num_sink:]

    key_all = torch.cat([full_key_states, padded_k], dim=2)
    value_all = torch.cat([full_value_states, padded_v], dim=2)

    # per-head keep mask: full heads see every slot, streaming heads only the
    # sink prefix and the recent suffix
    num_kv_head = key_all.shape[2]
    num_full_kv_head = num_kv_head - num_streaming_kv_head
    keep = torch.ones((bsz, num_kv_head, full_len), dtype=torch.bool, device=device)
    if num_streaming_kv_head > 0:
        keep[:, num_full_kv_head:, num_sink : full_len - num_recent] = False

    kv_groups = (
        num_full_query_head // num_full_kv_head
        if num_full_kv_head > 0
        else num_streaming_query_head // num_streaming_kv_head
    )
    q = query_states.transpose(1, 2).contiguous()
    k = repeat_kv(key_all.transpose(1, 2), kv_groups).contiguous()
    v = repeat_kv(value_all.transpose(1, 2), kv_groups).contiguous()
    if kv_groups > 1:
        keep = keep.repeat_interleave(kv_groups, dim=1)

    attn_output = F.scaled_dot_product_attention(
        q, k, v, attn_mask=keep[:, :, None, :], dropout_p=0.0, is_causal=False
    )
    return attn_output.transpose(1, 2)


def llama_duo_attention_forward_one_way_reordered_sdpa(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """DuoAttention kernel-free eval forward (native SDPA comparison variant).

    Algorithm identical to llama_duo_attention_forward_one_way_reordered,
    only replaces flash kernel with ``F.scaled_dot_product_attention`` (full-causal).
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

    # past_key_value = (full_KV, streaming_KV)
    # full_KV: (2 x bsz, num_full_key_value_heads, full_kv_seq_len, head_dim)
    # streaming_KV: (2 x bsz, num_streaming_key_value_heads, cache_size, head_dim)
    kv_seq_len = key_states.shape[1]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[2]

    cos, sin = self.rotary_emb(value_states, position_ids)
    if cos.device != query_states.device:
        cos, sin = cos.to(query_states.device), sin.to(query_states.device)
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        unsqueeze_dim=2,
    )

    if not hasattr(self, "full_attn_head_mask") or self.full_attn_head_mask is None:
        self.full_attn_head_mask = self.full_attention_heads > 0.5
        self.num_full_attn_head = self.full_attn_head_mask.sum().item()
        self.num_streaming_attn_head = self.num_key_value_heads - self.num_full_attn_head

        self.num_full_query_head = self.num_full_attn_head * self.num_key_value_groups
        self.num_streaming_query_head = self.num_heads - self.num_full_query_head

    full_key_states = key_states[:, :, : self.num_full_attn_head, :]
    full_value_states = value_states[:, :, : self.num_full_attn_head, :]

    streaming_key_states = key_states[:, :, self.num_full_attn_head :, :]
    streaming_value_states = value_states[:, :, self.num_full_attn_head :, :]

    if past_key_value is not None:
        past_full_KV = past_key_value[0].transpose(1, 2)
        past_streaming_KV = past_key_value[1].transpose(1, 2)

        past_full_key_states = past_full_KV[:bsz]
        past_full_value_states = past_full_KV[bsz:]

        full_key_states = torch.cat([past_full_key_states, full_key_states], dim=1)
        full_value_states = torch.cat([past_full_value_states, full_value_states], dim=1)

        past_streaming_key_states = past_streaming_KV[:bsz]
        past_streaming_value_states = past_streaming_KV[bsz:]

        streaming_key_states = torch.cat(
            [past_streaming_key_states, streaming_key_states], dim=1
        )
        streaming_value_states = torch.cat(
            [past_streaming_value_states, streaming_value_states], dim=1
        )

    if q_len == kv_seq_len:
        # pre-filling: native SDPA (full-causal)
        q, k, v = _sdpa_prepare(query_states, key_states, value_states, self.num_heads)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
    else:
        # decoding or continuous filling
        if self.num_full_attn_head > 0:
            full_query_states = query_states[:, :, : self.num_full_query_head, :]
            q, k, v = _sdpa_prepare(
                full_query_states, full_key_states, full_value_states,
                self.num_full_query_head,
            )
            full_attn_output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True
            ).transpose(1, 2)  # [bsz, q_len, num_full_query_head, dim]
        else:
            full_attn_output = None

        if self.num_streaming_attn_head > 0:
            streaming_query_states = query_states[:, :, self.num_full_query_head :, :]
            q, k, v = _sdpa_prepare(
                streaming_query_states,
                streaming_key_states,
                streaming_value_states,
                self.num_streaming_query_head,
            )
            streaming_attn_output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True
            ).transpose(1, 2)
        else:
            streaming_attn_output = None

        if full_attn_output is None:
            attn_output = streaming_attn_output
        elif streaming_attn_output is None:
            attn_output = full_attn_output
        else:
            attn_output = torch.cat([full_attn_output, streaming_attn_output], dim=2)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    # streaming cache window truncation: identical to original
    if streaming_key_states.shape[1] > self.recent_size + self.sink_size:
        recent_key_states = streaming_key_states[:, -self.recent_size :, :, :].clone()
        streaming_key_states[
            :, self.sink_size : self.sink_size + self.recent_size, :, :
        ].copy_(recent_key_states)
        streaming_key_states = streaming_key_states[
            :, : self.sink_size + self.recent_size, :, :
        ]

        recent_value_states = streaming_value_states[
            :, -self.recent_size :, :, :
        ].clone()
        streaming_value_states[
            :, self.sink_size : self.sink_size + self.recent_size, :, :
        ].copy_(recent_value_states)
        streaming_value_states = streaming_value_states[
            :, : self.sink_size + self.recent_size, :, :
        ]

    past_key_value = (
        (
            torch.cat([full_key_states, full_value_states], dim=0).transpose(1, 2),
            torch.cat([streaming_key_states, streaming_value_states], dim=0).transpose(1, 2),
        )
        if use_cache
        else None
    )

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def enable_llama_duo_attention_eval_sdpa(
    model: LlamaForCausalLM,
    full_attention_heads,
    sink_size,
    recent_size,
):
    """Kernel-free comparison variant: weight reordering reuses enable_llama_duo_attention_eval,
    only replaces forward with SDPA implementation."""
    enable_tuple_kv_cache_for_llama(model)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        layer_full_attention_heads = torch.tensor(
            full_attention_heads[idx], device=device, dtype=dtype
        )

        module.forward = types.MethodType(
            llama_duo_attention_forward_one_way_reordered_sdpa, module
        )
        module.q_proj = reorder_linear_weights(
            module.q_proj,
            layer_full_attention_heads,
            module.num_key_value_groups * module.head_dim,
            "out",
        )
        module.k_proj = reorder_linear_weights(
            module.k_proj,
            layer_full_attention_heads,
            module.head_dim,
            "out",
        )
        module.v_proj = reorder_linear_weights(
            module.v_proj,
            layer_full_attention_heads,
            module.head_dim,
            "out",
        )
        module.o_proj = reorder_linear_weights(
            module.o_proj,
            layer_full_attention_heads,
            module.num_key_value_groups * module.head_dim,
            "in",
        )
        layer_full_attention_heads = reorder_full_attn_heads(layer_full_attention_heads)

        module.sink_size = sink_size
        module.recent_size = recent_size
        module.register_buffer(
            "full_attention_heads",
            layer_full_attention_heads,
        )
