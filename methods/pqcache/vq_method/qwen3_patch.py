"""PQCache attention patch for Transformers 4.57 Qwen3 models.

This patch deliberately keeps the upstream Qwen3Model, Qwen3DecoderLayer and
generation code intact.  Only each Qwen3Attention.forward is replaced, so the
4.57 mutable Cache and two-value attention return convention are preserved.
"""

import math
import os
import types
from typing import Optional

import torch
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3ForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .baseline_compressor import *
from .retrieval_based.global_timer import SYNC_TEST_TIME, stage_begin, stage_end
from .retrieval_based.pq_search import PqBasedSearchCompressor
from .retrieval_based.sparq import SparQCompressor

from flash_attn import flash_attn_func


__all__ = ["VQQwen3ForCausalLM", "patch_qwen3_attention"]


def _layer_device(layer_idx: int, num_layers: int) -> torch.device:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_count = len(visible.split(",")) if visible else 1
    return torch.device(f"cuda:{layer_idx // math.ceil(num_layers / device_count)}")


def _build_compressor(config, layer_idx: int):
    name = config.compressor
    if name == "h2o":
        return KVCacheH2OOfficial(
            config.compress_ratio, config.important_ratio, config.recent_ratio, config.sink_size
        )
    if name == "no_drop_lb":
        return fullKVLimitBasedCompressor(
            config.compress_ratio, config.important_ratio, config.recent_ratio, config.gqa, config.sink_size,
            fixbudget=getattr(config, "fixbudget", True), budget=getattr(config, "budget", 0),
            recent_size=getattr(config, "recent_size", 32),
        )
    if name == "no_drop_lb_32":
        return fullKVLimitBasedCompressor32(
            config.compress_ratio, config.important_ratio, config.recent_ratio, config.gqa, config.sink_size,
            fixbudget=getattr(config, "fixbudget", True), budget=getattr(config, "budget", 0),
            recent_size=getattr(config, "recent_size", 32),
        )
    if name == "no_drop_lb_topp":
        return fullKVLimitBasedCompressorTOPP(config.fixthreshold, config.gqa)
    if name == "no_drop_lb_topp32":
        return fullKVLimitBasedCompressorTOPP32(config.fixthreshold, config.gqa)
    if name == "pq_search":
        return PqBasedSearchCompressor(
            config.compress_ratio, config.recent_ratio, config.n_subvec_per_head, config.n_subbits, config.gqa,
            config.sink_size, fixbudget=getattr(config, "fixbudget", False), budget=getattr(config, "budget", 0),
            recent_size=getattr(config, "recent_size", 32), layer_idx=layer_idx,
            cur_device=_layer_device(layer_idx, config.num_hidden_layers), max_iter=config.max_iter,
            kv_head=config.num_key_value_heads, dim=config.head_dim,
            num_layer_cnt=config.num_hidden_layers,
        )
    if name == "sparq_f":
        return SparQCompressor(
            config.compress_ratio, config.recent_ratio, config.sink_size, config.gqa, r=config.topr,
            idx=layer_idx, model_config=config, layer_idx=layer_idx,
            cur_device=_layer_device(layer_idx, config.num_hidden_layers),
            kv_head=config.num_key_value_heads, dim=config.head_dim,
        )
    if name == "original":
        return None
    raise ValueError(f"Unsupported PQCache compressor: {name}")


def patch_qwen3_attention(attn: Qwen3Attention, config, layer_idx: int) -> None:
    """Patch one Qwen3 attention module without changing the upstream model loop."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.shape
        if bsz != 1:
            raise ValueError("PQCache Qwen3 patch currently supports batch size 1 only")
        if cache_position is None:
            raise ValueError("Qwen3 PQCache patch requires cache_position")

        hidden_shape = (bsz, q_len, -1, self.head_dim)
        # Qwen3 differs from Qwen2.5: Q/K RMSNorm is mandatory before RoPE.
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        is_prefill = cache_position.numel() > 1
        pre_stage = "prefill_preAttn_ffn" if is_prefill else "decode_preAttn_ffn"
        if is_prefill:
            self.seq_cnt += 1
            self.fwd_cnt = 0

        if self.compressor == "original":
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx,
                    {"sin": sin, "cos": cos, "cache_position": cache_position},
                )
            if SYNC_TEST_TIME:
                stage_end(pre_stage, self.layer_idx)
                stage_begin("prefill_attn" if is_prefill else "decode_attn", self.layer_idx)
            attn_output = flash_attn_func(
                query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), causal=True
            ).transpose(1, 2)
            if SYNC_TEST_TIME:
                stage_end("prefill_attn" if is_prefill else "decode_attn", self.layer_idx)
        elif self.compressor in {"pq_search", "sparq_f"}:
            if SYNC_TEST_TIME:
                stage_end(pre_stage, self.layer_idx)
            if is_prefill:
                attn_output, _ = self.kvcache_quantizer.prefill_attn(query_states, (key_states, value_states))
            else:
                repeated_k = repeat_kv(key_states, self.num_key_value_groups)
                repeated_v = repeat_kv(value_states, self.num_key_value_groups)
                attn_output = self.kvcache_quantizer.decoding_attn(
                    self.num_key_value_groups, query_states, repeated_k, repeated_v
                ).to(query_states.dtype)
        else:
            if past_key_values is None:
                raise ValueError("PQCache compression requires use_cache=True")
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
            )
            if SYNC_TEST_TIME:
                stage_end(pre_stage, self.layer_idx)
            repeated_k = repeat_kv(key_states, self.num_key_value_groups)
            repeated_v = repeat_kv(value_states, self.num_key_value_groups)
            if is_prefill:
                attn_output = flash_attn_func(
                    query_states.transpose(1, 2), repeated_k.transpose(1, 2), repeated_v.transpose(1, 2), causal=True
                ).transpose(1, 2)
                self.kvcache_quantizer.apply(
                    (key_states, value_states), layer_idx=self.idx, query_states=query_states
                )
            else:
                if SYNC_TEST_TIME:
                    stage_begin("decode_attn", self.layer_idx)
                weights = torch.matmul(query_states.float(), repeated_k.float().transpose(2, 3)) * self.scaling
                weights = self.kvcache_quantizer.restore(
                    weights, self.num_key_value_groups, query_states=query_states,
                    key_states=unrepeat(repeated_k, self.num_key_value_groups, 1), layer_idx=self.idx,
                ).to(query_states.dtype)
                attn_output = torch.matmul(weights, repeated_v)
                if SYNC_TEST_TIME:
                    stage_end("decode_attn", self.layer_idx)

        if attn_output.shape != (bsz, query_states.shape[1], q_len, self.head_dim):
            raise ValueError(f"Unexpected Qwen3 attention output shape: {attn_output.shape}")
        post_stage = "prefill_postAttn_ffn" if is_prefill else "decode_postAttn_ffn"
        attn_output = self.o_proj(attn_output.transpose(1, 2).reshape(bsz, q_len, -1).contiguous())
        if SYNC_TEST_TIME:
            stage_begin(post_stage, self.layer_idx)
        self.fwd_cnt += 1
        return attn_output, None

    attn.forward = types.MethodType(forward, attn)
    attn.compressor = config.compressor
    attn.kvcache_quantizer = _build_compressor(config, layer_idx)
    attn.fwd_cnt = 0
    attn.seq_cnt = -1
    attn.idx = layer_idx


class VQQwen3ForCausalLM(Qwen3ForCausalLM):
    """Qwen3ForCausalLM that retains HF 4.57 generation and model code."""

    def patch(self, config=None):
        config = config or self.config
        for layer_idx, layer in enumerate(self.model.layers):
            patch_qwen3_attention(layer.self_attn, config, layer_idx)
        return self
