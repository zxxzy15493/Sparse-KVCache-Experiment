import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers import AutoConfig
import os

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")

import flashinfer
try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
import clusterkv._clusterkv_knl as _kernels

from clusterkv.clusterkv_utils import ClusterKVController, build_cluster, append_kv, prefill_forward, decode_sparse_attn, update_sel_indices
from clusterkv.clusterkv_utils.global_timer import stage_begin, stage_end
import clusterkv.utils


def _get_rotary_embedding_and_apply(model_type: str):
    """Pick the RoPE and apply_rotary_pos_emb implementation based on model_type."""
    if model_type == "llama":
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
        return LlamaRotaryEmbedding, apply_rotary_pos_emb
    elif model_type == "qwen2":
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding, apply_rotary_pos_emb
        return Qwen2RotaryEmbedding, apply_rotary_pos_emb
    else:
        # default to Llama
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
        return LlamaRotaryEmbedding, apply_rotary_pos_emb


class ClusterKVAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: AutoConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = getattr(config, "pretraining_tp", 1)
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.attention_bias = getattr(config, "attention_bias", config.model_type == "qwen2")
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.attention_bias)
        self._init_rope()

    def _init_rope(self):
        # pick the RoPE implementation based on model_type
        rotary_emb_class, apply_rotary_pos_emb_fn = _get_rotary_embedding_and_apply(self.config.model_type)
        self.rotary_emb = rotary_emb_class(config=self.config)
        self._apply_rotary_pos_emb = apply_rotary_pos_emb_fn

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _timer_enabled(self) -> bool:
        return getattr(self.config, "model_type", None) == "llama"

    def _timer_stage(self, q_len: int, name: str) -> str:
        prefix = "prefill" if q_len > 1 else "decode"
        return f"{prefix}_{name}"

    def _timer_begin(self, q_len: int, name: str) -> None:
        if self._timer_enabled():
            stage_begin(self._timer_stage(q_len, name), self.layer_idx)

    def _timer_end(self, q_len: int, name: str) -> None:
        if self._timer_enabled():
            stage_end(self._timer_stage(q_len, name), self.layer_idx)

    def _flash_full_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Run flash-attn on BSND tensors, expanding GQA KV heads if needed."""
        if flash_attn_func is None:
            raise ImportError("flash-attn is required for use_flash_attn_clusterkv=True.")
        # num_kv_heads = key_states.shape[2]
        # n_rep = self.num_heads // num_kv_heads
        # if n_rep > 1:
        #     key_states = key_states.repeat_interleave(n_rep, dim=2)
        #     value_states = value_states.repeat_interleave(n_rep, dim=2)
        return flash_attn_func(
            query_states,
            key_states,
            value_states,
            causal=is_causal,
        )

    def _flashinfer_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Run FlashInfer Python API on NHD tensors."""
        if query_states.shape[0] == 1:
            return flashinfer.single_decode_with_kv_cache(
                query_states.squeeze(0),
                key_states,
                value_states,
                kv_layout="NHD",
            ).unsqueeze(0)
        return flashinfer.single_prefill_with_kv_cache(
            query_states,
            key_states,
            value_states,
            causal=is_causal,
            kv_layout="NHD",
            use_fp16_qk_reduction=False,
        )

    def _cache_nhd_to_bsnd(self, cache_states: torch.Tensor) -> torch.Tensor:
        return cache_states.unsqueeze(0).contiguous()

    def _cache_kv_to_bsnd(
        self,
        controller: ClusterKVController,
        q_len: int,
        key_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states is not None and value_states is not None:
            return key_states.unsqueeze(0).contiguous(), value_states.unsqueeze(0).contiguous()

        k_full, v_full = controller.get_kv(self.layer_idx)
        k_full = k_full.squeeze(1).unsqueeze(0)
        v_full = v_full.squeeze(1).unsqueeze(0)
        if not (controller.offload and self.layer_idx >= 2 and q_len == 1):
            k_full = k_full.contiguous()
            v_full = v_full.contiguous()
        return k_full, v_full

    def _select_sparse_indices(
        self,
        query_states: torch.Tensor,
        controller: ClusterKVController,
    ) -> torch.Tensor:
        """Run ClusterKV selection kernels and sanitize out-of-cache ids."""
        controller.sel_token_indices.fill_(controller.max_seq_len + 1)
        _kernels.get_neigh_c(
            query_states,
            controller.centroids[self.layer_idx],
            controller.cluster_size[self.layer_idx],
            controller.neigh_c,
            controller.neigh_c_size,
        )
        _kernels.get_sel_indices(
            controller.neigh_c,
            controller.neigh_c_size,
            controller.cluster_size_ps[self.layer_idx],
            controller.cluster_key_indices[self.layer_idx],
            controller.sel_token_indices,
        )
        sel = controller.sel_token_indices + controller.sink
        invalid = (sel < 0) | (sel >= controller.kv_seqlen)
        if os.environ.get("CLUSTERKV_DEBUG_RECALL") == "1":
            sorted_sel = torch.sort(sel, dim=1).values
            dup_count = (sorted_sel[:, 1:] == sorted_sel[:, :-1]).sum().item()
            print(
                "[clusterkv-debug] "
                f"layer={self.layer_idx} generated={controller.generated_len} "
                f"kv={controller.kv_seqlen} sel_min={sel.min().item()} "
                f"sel_max={sel.max().item()} invalid={invalid.sum().item()} "
                f"dup={dup_count}",
                flush=True,
            )
        if controller.offload and self.layer_idx >= 2:
            replacement = controller.g2c[self.layer_idx]
        else:
            replacement = torch.zeros_like(sel)
        return torch.where(invalid, replacement, sel).to(torch.int32).contiguous()

    def _gather_with_sink_window(
        self,
        controller: ClusterKVController,
        sel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Materialize sparse decode K/V as BSND [sink | selected | window]."""
        if controller.offload and self.layer_idx >= 2:
            total_len = controller.sink + controller._token_budget + controller.cur_win_size
            k_compact = controller.kv_cache[self.layer_idx][:total_len, 0, 0, ...]
            v_compact = controller.kv_cache[self.layer_idx][:total_len, 1, 0, ...]
            return (
                k_compact.unsqueeze(0),
                v_compact.unsqueeze(0),
            )

        k_cache, v_cache = controller.get_kv(self.layer_idx)
        k_cache = k_cache.squeeze(1).contiguous()  # [S, Hkv, D]
        v_cache = v_cache.squeeze(1).contiguous()
        seq_len = controller.kv_seqlen
        sink = controller.sink

        n_rep = self.num_key_value_groups
        if n_rep > 1:
            # `sel` is per query head. For GQA, heads in the same KV group use
            # the same KV head; keep K/V compact and let flash-attn handle GQA.
            kv_idx = sel[::n_rep].to(torch.int64)
        else:
            kv_idx = sel.to(torch.int64)
        idx_long = kv_idx.unsqueeze(-1).expand(-1, -1, self.head_dim)
        k_cache_hsd = k_cache.transpose(0, 1)
        v_cache_hsd = v_cache.transpose(0, 1)
        k_hkd = k_cache_hsd.gather(1, idx_long)
        v_hkd = v_cache_hsd.gather(1, idx_long)
        k_sel = k_hkd.transpose(0, 1).unsqueeze(0)
        v_sel = v_hkd.transpose(0, 1).unsqueeze(0)

        k_sink = k_cache[:sink, ...].unsqueeze(0)
        v_sink = v_cache[:sink, ...].unsqueeze(0)

        cur_win_size = controller.cur_win_size
        win_start = max(sink, seq_len - cur_win_size)
        k_win = k_cache[win_start:seq_len, ...].unsqueeze(0)
        v_win = v_cache[win_start:seq_len, ...].unsqueeze(0)

        return torch.cat([k_sink, k_sel, k_win], dim=1), torch.cat([v_sink, v_sel, v_win], dim=1)

    def _forward_flash_attn_clusterkv(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        q_len: int,
        controller: ClusterKVController,
        skip_layer: int,
        token_budget: int,
        full_mode: bool,
    ) -> torch.Tensor:
        """Flash-attn implementation used by myllama.py/myqwen2.py."""
        if self.layer_idx >= 2 and not controller.full and q_len > 1:
            assert q_len > controller.sink
            torch.cuda.nvtx.range_push("build_cluster")
            self._timer_begin(q_len, "index_build")
            if controller.overlap_build:
                with torch.cuda.stream(controller.build_cluster_stream):
                    build_cluster(
                        controller,
                        self.layer_idx,
                        key_states[controller.sink:],
                        0,
                        controller.nlist,
                        controller.build_cluster_stream,
                    )
                    controller.build_cluster_events[self.layer_idx].record(controller.build_cluster_stream)
            else:
                build_cluster(
                    controller,
                    self.layer_idx,
                    key_states[controller.sink:],
                    0,
                    controller.nlist,
                    torch.cuda.default_stream(),
                )
            self._timer_end(q_len, "index_build")
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("append_kv")
        append_kv(key_states, value_states, controller, self.layer_idx)
        torch.cuda.nvtx.range_pop()

        if self.layer_idx >= 2 and not controller.full:
            if q_len == 1 and controller.generated_len % controller.window == 0:
                torch.cuda.nvtx.range_push("rebuild_cluster")
                self._timer_begin(q_len, "index_build")
                append_key_for_cluster = controller.get_app_k_clustering(self.layer_idx)
                build_cluster(
                    controller,
                    self.layer_idx,
                    append_key_for_cluster,
                    controller.kv_seqlen - controller.sink - controller.window,
                    controller.window_nlist,
                    torch.cuda.default_stream(),
                )
                if controller.offload and self.layer_idx >= 2:
                    controller.offload_window_kv(self.layer_idx)
                self._timer_end(q_len, "index_build")
                torch.cuda.nvtx.range_pop()

        q_bsnd = query_states.unsqueeze(0).contiguous()
        if self.layer_idx < skip_layer or full_mode:
            torch.cuda.nvtx.range_push("full_attn")
            k_full, v_full = self._cache_kv_to_bsnd(controller, q_len)
            self._timer_end(q_len, "pre_ffn")
            self._timer_begin(q_len, "attn")
            attn_output = self._flash_full_attention(q_bsnd, k_full, v_full, is_causal=True)
            self._timer_end(q_len, "attn")
            torch.cuda.nvtx.range_pop()
        elif q_len > 1:
            torch.cuda.nvtx.range_push("full_attn")
            if controller.offload and self.layer_idx >= 2:
                k_full, v_full = self._cache_kv_to_bsnd(controller, q_len, key_states, value_states)
            else:
                k_full, v_full = self._cache_kv_to_bsnd(controller, q_len)
            self._timer_end(q_len, "pre_ffn")
            self._timer_begin(q_len, "attn")
            attn_output = self._flash_full_attention(q_bsnd, k_full, v_full, is_causal=True)
            self._timer_end(q_len, "attn")
            torch.cuda.nvtx.range_pop()
            if controller.offload and self.layer_idx >= 2:
                self._timer_begin(q_len, "unload")
                controller.offload_prefill_kv(self.layer_idx, key_states, value_states)
                self._timer_end(q_len, "unload")
        else:
            if controller.centroids[self.layer_idx] is None or controller.kv_seqlen <= token_budget:
                torch.cuda.nvtx.range_push("full_attn")
                k_full, v_full = self._cache_kv_to_bsnd(controller, q_len)
                self._timer_end(q_len, "pre_ffn")
                self._timer_begin(q_len, "attn")
                attn_output = self._flash_full_attention(q_bsnd, k_full, v_full, is_causal=True)
                self._timer_end(q_len, "attn")
                torch.cuda.nvtx.range_pop()
            else:
                torch.cuda.nvtx.range_push("sel_indices")
                if not controller.build_cluster_finish[self.layer_idx]:
                    controller.build_cluster_events[self.layer_idx].wait(controller.build_cluster_stream)
                    controller.build_cluster_finish[self.layer_idx] = True
                self._timer_begin(q_len, "retrieve")
                sel = self._select_sparse_indices(query_states, controller)
                self._timer_end(q_len, "retrieve")
                if controller.offload and self.layer_idx >= 2:
                    self._timer_begin(q_len, "load")
                    _kernels.recall(
                        controller.kv_cache_mid(self.layer_idx),
                        controller.kv_cache_cpu[self.layer_idx],
                        sel,
                        controller.g2c[self.layer_idx],
                        controller.c2g,
                        controller.is_in_cache,
                        controller.is_in_topk,
                        controller.swap_out_indices,
                        controller.swap_in_indices,
                        controller.swap_out_count,
                        controller.swap_in_count,
                        controller.kv_seqlen,
                    )
                    self._timer_end(q_len, "load")
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("sparse_attn")
                if not (controller.offload and self.layer_idx >= 2):
                    self._timer_begin(q_len, "load")
                k_g, v_g = self._gather_with_sink_window(controller, sel)
                if not (controller.offload and self.layer_idx >= 2):
                    self._timer_end(q_len, "load")
                self._timer_end(q_len, "pre_ffn")
                self._timer_begin(q_len, "attn")
                attn_output = flash_attn_func(q_bsnd, k_g, v_g, causal=False)
                self._timer_end(q_len, "attn")
                torch.cuda.nvtx.range_pop()

        return attn_output.squeeze(0)

    def _forward_flashinfer_clusterkv(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        q_len: int,
        controller: ClusterKVController,
        skip_layer: int,
        token_budget: int,
        full_mode: bool,
    ) -> torch.Tensor:
        """FlashInfer Python implementation used by myllama2.py."""
        if self.layer_idx >= 2 and not controller.full and q_len > 1:
            assert q_len > controller.sink
            torch.cuda.nvtx.range_push("build_cluster")
            if controller.overlap_build:
                with torch.cuda.stream(controller.build_cluster_stream):
                    build_cluster(
                        controller,
                        self.layer_idx,
                        key_states[controller.sink:],
                        0,
                        controller.nlist,
                        controller.build_cluster_stream,
                    )
                    controller.build_cluster_events[self.layer_idx].record(controller.build_cluster_stream)
            else:
                build_cluster(
                    controller,
                    self.layer_idx,
                    key_states[controller.sink:],
                    0,
                    controller.nlist,
                    torch.cuda.default_stream(),
                )
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("append_kv")
        append_kv(key_states, value_states, controller, self.layer_idx)
        torch.cuda.nvtx.range_pop()

        if self.layer_idx >= 2 and not controller.full:
            if q_len == 1 and controller.generated_len % controller.window == 0:
                torch.cuda.nvtx.range_push("rebuild_cluster")
                append_key_for_cluster = controller.get_app_k_clustering(self.layer_idx)
                build_cluster(
                    controller,
                    self.layer_idx,
                    append_key_for_cluster,
                    controller.kv_seqlen - controller.sink - controller.window,
                    controller.window_nlist,
                    torch.cuda.default_stream(),
                )
                if controller.offload and self.layer_idx >= 2:
                    controller.offload_window_kv(self.layer_idx)
                torch.cuda.nvtx.range_pop()

        if self.layer_idx < skip_layer or full_mode:
            torch.cuda.nvtx.range_push("flashinfer_full_attn")
            k_full, v_full = controller.get_kv(self.layer_idx)
            k_full = k_full.squeeze(1).contiguous()
            v_full = v_full.squeeze(1).contiguous()
            attn_output = self._flashinfer_attention(query_states, k_full, v_full, is_causal=True)
            torch.cuda.nvtx.range_pop()
        elif q_len > 1:
            torch.cuda.nvtx.range_push("flashinfer_prefill_attn")
            if controller.offload and self.layer_idx >= 2:
                k_full = key_states
                v_full = value_states
            else:
                k_full, v_full = controller.get_kv(self.layer_idx)
                k_full = k_full.squeeze(1).contiguous()
                v_full = v_full.squeeze(1).contiguous()
            attn_output = self._flashinfer_attention(query_states, k_full, v_full, is_causal=True)
            torch.cuda.nvtx.range_pop()
            if controller.offload and self.layer_idx >= 2:
                controller.offload_prefill_kv(self.layer_idx, key_states, value_states)
        else:
            if controller.centroids[self.layer_idx] is None or controller.kv_seqlen <= token_budget:
                torch.cuda.nvtx.range_push("flashinfer_full_decode")
                k_full, v_full = controller.get_kv(self.layer_idx)
                k_full = k_full.squeeze(1).contiguous()
                v_full = v_full.squeeze(1).contiguous()
                attn_output = self._flashinfer_attention(query_states, k_full, v_full, is_causal=False)
                torch.cuda.nvtx.range_pop()
            else:
                torch.cuda.nvtx.range_push("sel_indices")
                if not controller.build_cluster_finish[self.layer_idx]:
                    controller.build_cluster_events[self.layer_idx].wait(controller.build_cluster_stream)
                    controller.build_cluster_finish[self.layer_idx] = True
                sel = self._select_sparse_indices(query_states, controller)
                if controller.offload and self.layer_idx >= 2:
                    _kernels.recall(
                        controller.kv_cache_mid(self.layer_idx),
                        controller.kv_cache_cpu[self.layer_idx],
                        sel,
                        controller.g2c[self.layer_idx],
                        controller.c2g,
                        controller.is_in_cache,
                        controller.is_in_topk,
                        controller.swap_out_indices,
                        controller.swap_in_indices,
                        controller.swap_out_count,
                        controller.swap_in_count,
                        controller.kv_seqlen,
                    )
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("flashinfer_sparse_decode")
                k_g, v_g = self._gather_with_sink_window(controller, sel)
                attn_output = self._flashinfer_attention(
                    query_states,
                    k_g.squeeze(0).contiguous(),
                    v_g.squeeze(0).contiguous(),
                    is_causal=False,
                )
                torch.cuda.nvtx.range_pop()

        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        controller: Optional[ClusterKVController] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_flash_attn_clusterkv: bool = False,
        use_flashinfer_clusterkv: bool = False,
        skip_layer: Optional[int] = None,
        token_budget: Optional[int] = None,
        full_mode: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        assert bsz == 1, "ClusterKVAttention only supports batch size 1."
        assert hasattr(self, 'layer_idx'), "ClusterKVAttention requires layer_idx to inference."

        if self.pretraining_tp > 1:
            assert False and "should not happen"
        else:
            torch.cuda.nvtx.range_push("qkv_proj")
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            torch.cuda.nvtx.range_pop()

        # Keep BSND until RoPE, then drop the batch dimension for ClusterKV's NHD layout.
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        )
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )

        torch.cuda.nvtx.range_push("RoPE")

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = self._apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            unsqueeze_dim=2,
        )

        query_states = query_states.squeeze(0)
        key_states = key_states.squeeze(0)
        value_states = value_states.squeeze(0)

        torch.cuda.nvtx.range_pop()

        if use_flash_attn_clusterkv or use_flashinfer_clusterkv:
            if controller is None:
                raise ValueError("controller must be set for materialized ClusterKV forward.")
            if skip_layer is None:
                skip_layer = 0
            if token_budget is None:
                token_budget = controller._token_budget
            if full_mode is None:
                full_mode = controller.full
            if use_flashinfer_clusterkv:
                attn_output = self._forward_flashinfer_clusterkv(
                    query_states,
                    key_states,
                    value_states,
                    q_len,
                    controller,
                    skip_layer,
                    token_budget,
                    full_mode,
                )
            else:
                attn_output = self._forward_flash_attn_clusterkv(
                    query_states,
                    key_states,
                    value_states,
                    q_len,
                    controller,
                    skip_layer,
                    token_budget,
                    full_mode,
                )
            attn_output = attn_output.unsqueeze(0)
            if attn_output.size() != (bsz, q_len, self.num_heads, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, q_len, self.num_heads, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            self._timer_begin(q_len, "post_ffn")
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
            torch.cuda.nvtx.range_push("o_proj")
            if self.pretraining_tp > 1:
                attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
                o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
                attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
            else:
                attn_output = self.o_proj(attn_output)
            torch.cuda.nvtx.range_pop()
            attn_weights = None
            return attn_output, attn_weights, past_key_value

        if self.layer_idx >= 2 and not controller.full:
            if q_len > 1:
                # build clusters during prefill
                assert q_len > controller.sink
                if controller.overlap_build:
                    with torch.cuda.stream(controller.build_cluster_stream):
                        build_cluster(
                            controller, self.layer_idx, key_states[controller.sink:], 0,
                            controller.nlist, controller.build_cluster_stream
                        )
                        controller.build_cluster_events[self.layer_idx].record(controller.build_cluster_stream)
                else:
                    build_cluster(
                        controller, self.layer_idx, key_states[controller.sink:], 0,
                        controller.nlist, torch.cuda.default_stream()
                    )

        torch.cuda.nvtx.range_push("append_kv")
        append_kv(
            key_states,
            value_states,
            controller,
            self.layer_idx,
        )
        torch.cuda.nvtx.range_pop()

        if self.layer_idx >= 2 and not controller.full:
            if q_len == 1 and controller.generated_len % controller.window == 0:
                # appending clustering during decoding
                append_key_for_cluster = controller.get_app_k_clustering(self.layer_idx)
                build_cluster(
                    controller, self.layer_idx, append_key_for_cluster, 
                    controller.kv_seqlen - controller.sink - controller.window,
                    controller.window_nlist, torch.cuda.default_stream()
                )
                if self.layer_idx >= 2 and controller.offload:
                    controller.offload_window_kv(self.layer_idx)

        # Prefill/Decode kernels is different
        if q_len > 1:
            torch.cuda.nvtx.range_push("prefill_attn")
            if controller.offload:
                attn_output = prefill_forward(
                    query_states,
                    controller,
                    self.layer_idx,
                    key_states=key_states,
                    value_states=value_states
                )
            else:
                attn_output = prefill_forward(
                    query_states,
                    controller,
                    self.layer_idx,
                )
            torch.cuda.nvtx.range_pop()
            if self.layer_idx >= 2 and controller.offload:
                controller.offload_prefill_kv(self.layer_idx, key_states, value_states)
        else:
            # Skipping layers is controled by PAGE_BUDGET, which is set in LlamaModel.
            if not controller.need_estimate():
                torch.cuda.nvtx.range_push("full_attn")
                attn_output = decode_sparse_attn(
                    query_states,
                    controller,
                    self.layer_idx,
                    None
                )
                torch.cuda.nvtx.range_pop()
            else:
                # sel = True
                # if sel:
                torch.cuda.nvtx.range_push("indexing")
                if not controller.build_cluster_finish[self.layer_idx]:
                    controller.build_cluster_events[self.layer_idx].wait(controller.build_cluster_stream)
                    controller.build_cluster_finish[self.layer_idx] = True
                update_sel_indices(
                    query_states,
                    controller,
                    self.layer_idx,
                )
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("approx_attn")
                # print(controller.sel_token_indices)
                attn_output = decode_sparse_attn(
                    query_states,
                    controller,
                    self.layer_idx,
                    controller.sel_token_indices
                )
                torch.cuda.nvtx.range_pop()

        attn_output = attn_output.unsqueeze(0) # unsqueeze the batch dimension
        # FlashInfer output is naturally NHD
        # Note that we manully control NHD. Should be more general
        if attn_output.size() != (bsz, q_len, self.num_heads, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, q_len, self.num_heads, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        torch.cuda.nvtx.range_push("o_proj")
        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)
        torch.cuda.nvtx.range_pop()

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
