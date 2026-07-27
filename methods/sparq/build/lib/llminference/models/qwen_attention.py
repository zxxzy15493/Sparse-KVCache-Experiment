# Copyright ...
#
# Copied from `transformers.models.qwen2.modeling_qwen2`
# Modified to enable plug-in attention sparsity.
#
# Modifications are marked with "# MODIFIED"

# mypy: ignore-errors
# fmt: off

import math
import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from transformers.cache_utils import Cache
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class Qwen2Attention(modeling_qwen2.Qwen2Attention):
    # -------------------------
    # MODIFIED (added): isolate dense attention math so it can be overridden
    # -------------------------
    def _attn(
        self,
        query_states: Tensor,
        key_states: Tensor,
        value_states: Tensor,
        attention_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        # repeat k/v heads if n_kv_heads < n_heads
        # NOTE: KV repetition should be inside _attn() for GQA to behave as intended
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Prefer flash-attn when available. Keep this conservative and only use
        # it for single-item batches to avoid padding-mask correctness issues.
        if FLASH_ATTN_AVAILABLE and query_states.shape[0] == 1:
            q = query_states.transpose(1, 2)
            k = key_states.transpose(1, 2)
            v = value_states.transpose(1, 2)
            attn_output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True,
                return_attn_probs=False,
            ).transpose(1, 2)
            return attn_output, None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # MODIFIED: do NOT hard-check (bsz,1,q_len,kv_len) so per-head masking is possible if you need it
            # (same motivation as your mistral_attention.py)
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32 for numerical stability
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output, attn_weights

    # -------------------------
    # Small helper: keep compatibility with different apply_rotary_pos_emb signatures
    # (Qwen2 variants sometimes add an unsqueeze_dim argument)
    # -------------------------
    @staticmethod
    def _apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, position_ids, unsqueeze_dim: int):
        try:
            return apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=unsqueeze_dim)
        except TypeError:
            try:
                return apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim)
            except TypeError:
                return apply_rotary_pos_emb(q, k, cos, sin, position_ids)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,  # MODIFIED: keep parity with newer decoder APIs
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        # MODIFIED - added (same as your other attention files)
        #33assert attention_mask is not None, "attention_mask cannot be None when using sparse methods"

        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated; use `attention_mask` instead."
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Qwen2 is GQA-capable: q has num_heads, kv has num_key_value_heads
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # match HF behavior: allow module to carry its own past_key_value
        past_key_value = getattr(self, "past_key_value", past_key_value)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    "Qwen2Attention needs layer_idx set when using kv cache (past_key_value is not None)."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Qwen2 RoPE: typically rotary_emb(value_states, seq_len=kv_seq_len)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = self._apply_rope(query_states, key_states, cos, sin, position_ids, unsqueeze_dim=1)

        if past_key_value is not None:

            cache_kwargs = {"sin": sin, "cos": cos}
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # MODIFIED: slice mask to actual kv length (important when kv grows with cache)
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask[..., : key_states.shape[-2]]

        # MODIFIED (call)
        attn_output, attn_weights = self._attn(query_states, key_states, value_states, attention_mask)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
