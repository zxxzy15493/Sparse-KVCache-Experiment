"""
Native (uncompressed) Llama / Qwen2 model with per-component timing instrumentation.

This patches the original HuggingFace LlamaDecoderLayer / Qwen2DecoderLayer and
LlamaAttention / Qwen2Attention (sdpa / flash_attention_2 / eager) so that every
forward pass records per-layer stage timings via the global_timer module.

Supported model families: Llama (3.x, 2), Qwen2 (2.x, 2.5)

Stage boundaries EXACTLY match PQCache's instrumentation
(see vq_method/llama31_patch.py and vq_method/retrieval_based/pq_search.py):

  prefill_preAttn_ffn  / decode_preAttn_ffn   → input_layernorm + qkv + RoPE
  prefill_write_cache  / decode_write_cache   → native KV-cache update only
  prefill_attn         / decode_attn          → attention kernel only (no o_proj)
  prefill_postAttn_ffn / decode_postAttn_ffn  → residual add + post_attention_layernorm + mlp + residual add

Everything else (qkv proj, RoPE, o_proj, residual adds) is
**untimed overhead** — exactly as in the PQCache instrumented model.

Stages that don't apply to native models (index_build, pq, load, unload) are
never recorded — they will appear as 0 ms in the final report.
"""

import math
import os
import types
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaFlashAttention2,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2FlashAttention2,
    Qwen2ForCausalLM,
    Qwen2Model,
    Qwen2SdpaAttention,
    apply_rotary_pos_emb as apply_rotary_pos_emb_qwen2,
    repeat_kv as repeat_kv_qwen2,
)

# Same global timer used by PQCache
from vq_method.retrieval_based.global_timer import (
    SYNC_TEST_TIME as _SYNC_TEST_TIME,
    stage_begin,
    stage_end,
)

__all__ = [
    "NativeTimedLlamaForCausalLM",
    "NativeTimedQwen2ForCausalLM",
    "create_native_timed_model",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_prefill(position_ids: torch.Tensor) -> bool:
    """Prefill when position_ids has more than 1 element (batch=1 assumed)."""
    return position_ids.numel() != 1


def _prefill_decode_stage(base_name: str, first_time: bool) -> str:
    """Return e.g. 'prefill_attn' or 'decode_attn'."""
    prefix = "prefill" if first_time else "decode"
    return f"{prefix}_{base_name}"


def _end_pre_attn_stage(first_time: bool, layer_idx: int):
    pre_stage_name = _prefill_decode_stage("preAttn_ffn", first_time)
    if _SYNC_TEST_TIME:
        stage_end(pre_stage_name, layer_idx)


def _timed_cache_update(
    past_key_value,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    sin: torch.Tensor,
    cos: torch.Tensor,
    cache_position: Optional[torch.LongTensor],
    first_time: bool,
):
    if past_key_value is not None:
        stage_name = _prefill_decode_stage("write_cache", first_time)
        if _SYNC_TEST_TIME:
            stage_begin(stage_name, layer_idx)
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, layer_idx, cache_kwargs)
        if _SYNC_TEST_TIME:
            stage_end(stage_name, layer_idx)
    return key_states, value_states


# ---------------------------------------------------------------------------
# Common: qkv proj + RoPE + timed KV cache → returns (q, k, v) ready for attention
# ---------------------------------------------------------------------------

def _qkv_rope_cache(
    self,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_value,
    attention_mask,
    position_embeddings,
    cache_position,
    first_time: bool,
    layer_idx: int,
):
    """
    Run qkv projections, RoPE, and the timed KV cache update.
    Returns (query_states, key_states, value_states) ready for the attention
    kernel, before GQA repeat.
    """
    bsz, q_len, _ = hidden_states.size()

    # --- qkv projections ---
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # --- RoPE ---
    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # --- KV cache ---
    _end_pre_attn_stage(first_time, layer_idx)
    key_states, value_states = _timed_cache_update(
        past_key_value, key_states, value_states, self.layer_idx,
        sin, cos, cache_position, first_time,
    )

    return query_states, key_states, value_states


# ---------------------------------------------------------------------------
# Attention patches  (one factory per attn_implementation)
# ---------------------------------------------------------------------------

def _make_timed_sdpa_forward(attn: LlamaSdpaAttention, layer_idx: int):
    """
    SDPA path.  Time only the F.scaled_dot_product_attention call.
    o_proj runs outside the timer.
    """
    original_forward = attn.forward

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        if output_attentions:
            return original_forward(
                hidden_states=hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, past_key_value=past_key_value,
                output_attentions=True, use_cache=use_cache,
                cache_position=cache_position, position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()
        first_time = _is_prefill(position_ids)

        # === preAttn_ffn: qkv + RoPE; write_cache: native KV-cache update ===
        q, k, v = _qkv_rope_cache(
            self, hidden_states, position_ids, past_key_value,
            attention_mask, position_embeddings, cache_position,
            first_time, layer_idx,
        )

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # === timed: attention kernel ===
        stage_name = _prefill_decode_stage("attn", first_time)
        if _SYNC_TEST_TIME:
            stage_begin(stage_name, layer_idx)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, k.shape[-2]]

        if q.device.type == "cuda" and causal_mask is not None:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        if _SYNC_TEST_TIME:
            stage_end(stage_name, layer_idx)

        # === postAttn_ffn starts at output reshape + o_proj ===
        post_stage_name = _prefill_decode_stage("postAttn_ffn", first_time)
        attn_output = attn_output.transpose(1, 2).contiguous()
        if _SYNC_TEST_TIME:
            stage_begin(post_stage_name, layer_idx)
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    return forward


def _make_timed_flash_forward(attn: LlamaFlashAttention2, layer_idx: int):
    """
    Flash-Attention-2 path.  Time only the _flash_attention_forward call.
    """
    from transformers.models.llama.modeling_llama import _flash_attention_forward

    original_forward = attn.forward

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        from transformers.cache_utils import StaticCache
        if isinstance(past_key_value, StaticCache):
            raise ValueError("StaticCache not supported with flash_attention_2")

        bsz, q_len, _ = hidden_states.size()
        first_time = _is_prefill(position_ids)

        # === preAttn_ffn: qkv + RoPE; write_cache: native KV-cache update ===
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        _end_pre_attn_stage(first_time, layer_idx)
        key_states, value_states = _timed_cache_update(
            past_key_value, key_states, value_states, self.layer_idx,
            sin, cos, cache_position, first_time,
        )

        # Transpose for flash_attn layout
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # === timed: flash attention forward ===
        stage_name = _prefill_decode_stage("attn", first_time)
        if _SYNC_TEST_TIME:
            stage_begin(stage_name, layer_idx)

        attn_output = _flash_attention_forward(
            query_states, key_states, value_states, attention_mask,
            q_len, position_ids=position_ids, dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        if _SYNC_TEST_TIME:
            stage_end(stage_name, layer_idx)

        # === postAttn_ffn starts at reshape + o_proj ===
        post_stage_name = _prefill_decode_stage("postAttn_ffn", first_time)
        if _SYNC_TEST_TIME:
            stage_begin(post_stage_name, layer_idx)
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    return forward


def _make_timed_eager_forward(attn: LlamaAttention, layer_idx: int):
    """
    Eager (manual matmul) path.  Time the matmul + softmax + matmul block.
    """
    original_forward = attn.forward

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        first_time = _is_prefill(position_ids)

        # === preAttn_ffn: qkv + RoPE; write_cache: native KV-cache update ===
        q, k, v = _qkv_rope_cache(
            self, hidden_states, position_ids, past_key_value,
            attention_mask, position_embeddings, cache_position,
            first_time, layer_idx,
        )

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # === timed: attention kernel ===
        stage_name = _prefill_decode_stage("attn", first_time)
        if _SYNC_TEST_TIME:
            stage_begin(stage_name, layer_idx)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : k.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, v)

        if _SYNC_TEST_TIME:
            stage_end(stage_name, layer_idx)

        # === postAttn_ffn starts at reshape + o_proj ===
        post_stage_name = _prefill_decode_stage("postAttn_ffn", first_time)
        attn_output = attn_output.transpose(1, 2).contiguous()
        if _SYNC_TEST_TIME:
            stage_begin(post_stage_name, layer_idx)
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return forward


def _patch_attention(attn, layer_idx: int):
    """Patch an attention module's forward with the appropriate timed version."""
    if isinstance(attn, LlamaSdpaAttention):
        new_fwd = _make_timed_sdpa_forward(attn, layer_idx)
    elif isinstance(attn, LlamaFlashAttention2):
        new_fwd = _make_timed_flash_forward(attn, layer_idx)
    elif isinstance(attn, LlamaAttention):
        new_fwd = _make_timed_eager_forward(attn, layer_idx)
    else:
        raise TypeError(f"Unknown attention class: {type(attn)}")

    attn.forward = types.MethodType(new_fwd, attn)


# ---------------------------------------------------------------------------
# Qwen2 attention patches  (sdpa / flash_attention_2 / eager)
# ---------------------------------------------------------------------------

def _make_timed_qwen2_sdpa_forward(attn: Qwen2SdpaAttention, layer_idx: int):
    """
    Qwen2 SDPA path.  Time only the F.scaled_dot_product_attention call.
    o_proj runs outside the timer.
    """
    original_forward = attn.forward

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        if output_attentions:
            return original_forward(
                hidden_states=hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, past_key_value=past_key_value,
                output_attentions=True, use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()
        first_time = _is_prefill(position_ids)

        # === preAttn_ffn: qkv + RoPE; write_cache: native KV-cache update ===
        q, k, v = _qkv_rope_cache(
            self, hidden_states, position_ids, past_key_value,
            attention_mask, position_embeddings, cache_position,
            first_time, layer_idx,
        )

        k = repeat_kv_qwen2(k, self.num_key_value_groups)
        v = repeat_kv_qwen2(v, self.num_key_value_groups)

        # === timed: attention kernel ===
        stage_name = _prefill_decode_stage("attn", first_time)
        if _SYNC_TEST_TIME:
            stage_begin(stage_name, layer_idx)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, k.shape[-2]]

        if q.device.type == "cuda" and causal_mask is not None:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        if _SYNC_TEST_TIME:
            stage_end(stage_name, layer_idx)

        # === postAttn_ffn starts at output reshape + o_proj ===
        post_stage_name = _prefill_decode_stage("postAttn_ffn", first_time)
        attn_output = attn_output.transpose(1, 2).contiguous()
        if _SYNC_TEST_TIME:
            stage_begin(post_stage_name, layer_idx)
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    return forward


def _make_timed_qwen2_flash_forward(attn: Qwen2FlashAttention2, layer_idx: int):
    """
    Qwen2 Flash-Attention-2 path.
    Time only the _flash_attention_forward call.
    Includes Qwen2-specific sliding-window logic.
    """
    from transformers.models.qwen2.modeling_qwen2 import _flash_attention_forward

    original_forward = attn.forward

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        from transformers.cache_utils import StaticCache
        if isinstance(past_key_value, StaticCache):
            raise ValueError("StaticCache not supported with flash_attention_2")

        bsz, q_len, _ = hidden_states.size()
        first_time = _is_prefill(position_ids)

        # === untimed: qkv proj ===
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # --- RoPE ---
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_qwen2(query_states, key_states, cos, sin)

        _end_pre_attn_stage(first_time, layer_idx)

        # --- KV cache (Qwen2-specific: sliding window slicing) ---
        if past_key_value is not None:
            stage_name = _prefill_decode_stage("write_cache", first_time)
            if _SYNC_TEST_TIME:
                stage_begin(stage_name, layer_idx)
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            kv_seq_len = key_states.shape[-2] + cache_position[0]
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1
                    )

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
            if _SYNC_TEST_TIME:
                stage_end(stage_name, layer_idx)

        # --- GQA repeat ---
        # key_states = repeat_kv_qwen2(key_states, self.num_key_value_groups)
        # value_states = repeat_kv_qwen2(value_states, self.num_key_value_groups)

        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # --- dtype handling ---
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Transpose for flash_attn layout
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Sliding window
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        # === timed: flash attention forward ===
        stage_name = _prefill_decode_stage("attn", first_time)
        if _SYNC_TEST_TIME:
            stage_begin(stage_name, layer_idx)

        attn_output = _flash_attention_forward(
            query_states, key_states, value_states, attention_mask,
            q_len, position_ids=position_ids, dropout=dropout_rate,
            sliding_window=sliding_window,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        if _SYNC_TEST_TIME:
            stage_end(stage_name, layer_idx)

        # === postAttn_ffn starts at reshape + o_proj ===
        post_stage_name = _prefill_decode_stage("postAttn_ffn", first_time)
        if _SYNC_TEST_TIME:
            stage_begin(post_stage_name, layer_idx)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    return forward


def _make_timed_qwen2_eager_forward(attn: Qwen2Attention, layer_idx: int):
    """
    Qwen2 eager (manual matmul) path.  Time the matmul + softmax + matmul block.
    """
    original_forward = attn.forward

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        first_time = _is_prefill(position_ids)

        # === preAttn_ffn: qkv + RoPE; write_cache: native KV-cache update ===
        q, k, v = _qkv_rope_cache(
            self, hidden_states, position_ids, past_key_value,
            attention_mask, position_embeddings, cache_position,
            first_time, layer_idx,
        )

        k = repeat_kv_qwen2(k, self.num_key_value_groups)
        v = repeat_kv_qwen2(v, self.num_key_value_groups)

        # === timed: attention kernel ===
        stage_name = _prefill_decode_stage("attn", first_time)
        if _SYNC_TEST_TIME:
            stage_begin(stage_name, layer_idx)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : k.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, v)

        if _SYNC_TEST_TIME:
            stage_end(stage_name, layer_idx)

        # === postAttn_ffn starts at reshape + o_proj ===
        post_stage_name = _prefill_decode_stage("postAttn_ffn", first_time)
        attn_output = attn_output.transpose(1, 2).contiguous()
        if _SYNC_TEST_TIME:
            stage_begin(post_stage_name, layer_idx)
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return forward


def _patch_qwen2_attention(attn, layer_idx: int):
    """Patch a Qwen2 attention module's forward with the appropriate timed version."""
    if isinstance(attn, Qwen2SdpaAttention):
        new_fwd = _make_timed_qwen2_sdpa_forward(attn, layer_idx)
    elif isinstance(attn, Qwen2FlashAttention2):
        new_fwd = _make_timed_qwen2_flash_forward(attn, layer_idx)
    elif isinstance(attn, Qwen2Attention):
        new_fwd = _make_timed_qwen2_eager_forward(attn, layer_idx)
    else:
        raise TypeError(f"Unknown Qwen2 attention class: {type(attn)}")

    attn.forward = types.MethodType(new_fwd, attn)


# ---------------------------------------------------------------------------
# Patched LlamaDecoderLayer
# ---------------------------------------------------------------------------

def _make_timed_decoder_layer_forward(layer_idx: int):
    """
    Returns a patched `forward` that wraps:
      preAttn_ffn  → input_layernorm
      postAttn_ffn → post_attention_layernorm + mlp
    The attention timing is handled by the patched attention forward.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        first_time = _is_prefill(position_ids)

        residual = hidden_states

        # ---- preAttn_ffn: input_layernorm only ----
        stage_name = _prefill_decode_stage("preAttn_ffn", first_time)
        if _SYNC_TEST_TIME:
            stage_begin(stage_name, layer_idx)
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention. Attention-kernel timing is recorded inside self_attn.
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
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

        # ---- postAttn_ffn: norm + mlp ----
        stage_name = _prefill_decode_stage("postAttn_ffn", first_time)

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        if _SYNC_TEST_TIME:
            stage_end(stage_name, layer_idx)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs

    return forward


# ---------------------------------------------------------------------------
# Qwen2 DecoderLayer
# ---------------------------------------------------------------------------

def _make_timed_qwen2_decoder_layer_forward(layer_idx: int):
    """
    Returns a patched `forward` for Qwen2DecoderLayer that wraps:
      preAttn_ffn  → input_layernorm
      postAttn_ffn → post_attention_layernorm + mlp
    The attention timing is handled by the patched attention forward.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        first_time = _is_prefill(position_ids)

        residual = hidden_states

        # ---- preAttn_ffn: input_layernorm only ----
        stage_name = _prefill_decode_stage("preAttn_ffn", first_time)
        if _SYNC_TEST_TIME:
            stage_begin(stage_name, layer_idx)
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention. Attention-kernel timing is recorded inside self_attn.
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
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

        # ---- postAttn_ffn: norm + mlp ----
        stage_name = _prefill_decode_stage("postAttn_ffn", first_time)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        if _SYNC_TEST_TIME:
            stage_end(stage_name, layer_idx)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs

    return forward


# ---------------------------------------------------------------------------
# Build the timed models
# ---------------------------------------------------------------------------

def _patch_model_with_timing(model: LlamaModel):
    """Replace every Llama decoder layer's forward + attention forward with timed versions."""
    for layer_idx, decoder_layer in enumerate(model.layers):
        _patch_attention(decoder_layer.self_attn, layer_idx)
        decoder_layer.forward = types.MethodType(
            _make_timed_decoder_layer_forward(layer_idx), decoder_layer
        )
    return model


def _patch_qwen2_model_with_timing(model: Qwen2Model):
    """Replace every Qwen2 decoder layer's forward + attention forward with timed versions."""
    for layer_idx, decoder_layer in enumerate(model.layers):
        _patch_qwen2_attention(decoder_layer.self_attn, layer_idx)
        decoder_layer.forward = types.MethodType(
            _make_timed_qwen2_decoder_layer_forward(layer_idx), decoder_layer
        )
    return model


# ---------------------------------------------------------------------------
# Public wrapper classes
# ---------------------------------------------------------------------------

class NativeTimedLlamaForCausalLM(LlamaForCausalLM):
    """
    HuggingFace LlamaForCausalLM with per-component GPU timing instrumentation.

    Usage::

        from breakdown_test.llama_native_timed import NativeTimedLlamaForCausalLM

        model = NativeTimedLlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = model.eval().cuda()

        # Then use global_timer.init_timer / init_all_stages / set_recording_state
        # as usual.
    """

    def __init__(self, config):
        super().__init__(config)
        _patch_model_with_timing(self.model)


class NativeTimedQwen2ForCausalLM(Qwen2ForCausalLM):
    """
    HuggingFace Qwen2ForCausalLM with per-component GPU timing instrumentation.

    Usage::

        from breakdown_test.llama_native_timed import NativeTimedQwen2ForCausalLM

        model = NativeTimedQwen2ForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = model.eval().cuda()

        # Then use global_timer.init_timer / init_all_stages / set_recording_state
        # as usual.
    """

    def __init__(self, config):
        super().__init__(config)
        _patch_qwen2_model_with_timing(self.model)


# ---------------------------------------------------------------------------
# Convenience: auto-detect model family
# ---------------------------------------------------------------------------

def create_native_timed_model(model_name_or_path: str, **kwargs):
    """
    Auto-detect model family from the model name and return the appropriate
    timed model instance.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        **kwargs: Passed through to ``from_pretrained``
                  (e.g. torch_dtype, attn_implementation, trust_remote_code).

    Returns:
        ``NativeTimedLlamaForCausalLM`` or ``NativeTimedQwen2ForCausalLM``.

    Example::

        model = create_native_timed_model(
            "Qwen/Qwen2.5-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = model.eval().cuda()
    """
    name_lower = model_name_or_path.lower()
    if "qwen" in name_lower:
        return NativeTimedQwen2ForCausalLM.from_pretrained(
            model_name_or_path, **kwargs
        )
    else:
        return NativeTimedLlamaForCausalLM.from_pretrained(
            model_name_or_path, **kwargs
        )
