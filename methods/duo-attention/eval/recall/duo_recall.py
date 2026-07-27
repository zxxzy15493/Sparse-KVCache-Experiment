"""
Recall tracking hooks for DuoAttention.

Computes recall metrics (recall@100, recall@k, selected_attn_ratio) for
DuoAttention's head-level KV compression.

For each decode step:
  - Full-attention heads see all KV tokens -> recall = 1.0 (no compression loss)
  - Streaming-attention heads see only sink + recent tokens -> compute recall
    by comparing full attention distribution vs streaming-accessible distribution.
"""

import sys
import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import List
from transformers.models.qwen2.modeling_qwen2 import repeat_kv


def _rotate_half(x):
    """Rotates half the hidden dims of the input. Standard RoPE helper."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# ============================================================
# Global registry (shared with recall.py via module reference)
# ============================================================
stats_registry: List[list] = []
current_sample_stats: list = []


def init_new_sample_registry():
    global current_sample_stats
    if current_sample_stats:
        stats_registry.append(current_sample_stats)
    current_sample_stats = []


# ============================================================
# Recall computation
# ============================================================

def _compute_recall_per_head(full_attn, selected_positions):
    """
    Compute recall@k, recall@100, and selected_attn_ratio for one head.

    Args:
        full_attn: [1, seq_len] full attention distribution (probabilities).
        selected_positions: set of positions accessible by the compressed KV.

    Returns:
        (recall_100, recall_k, selected_attn_ratio)
    """
    probs = full_attn[0]  # [seq_len]
    seq_len = probs.shape[0]

    # --- recall@k: what fraction of top-k positions are selected ---
    k = min(len(selected_positions), seq_len)
    probs_sorted = probs.sort(descending=True)
    topk_positions = probs_sorted.indices[:k].tolist()
    topk_hit = sum(1 for p in topk_positions if p in selected_positions)
    recall_k = topk_hit / max(k, 1)

    # --- recall@100: like CakeKV, top 100 positions ---
    top100 = min(100, seq_len)
    top100_positions = probs_sorted.indices[:top100].tolist()
    top100_hit = sum(1 for p in top100_positions if p in selected_positions)
    recall_100 = top100_hit / max(top100, 1)

    # --- selected_attn_ratio: total attention mass captured ---
    sel = torch.tensor(sorted(selected_positions), device=probs.device, dtype=torch.long)
    sel = sel.clamp(0, seq_len - 1)
    selected_attn_ratio = probs[sel].sum().item()

    return recall_100, recall_k, selected_attn_ratio


# ============================================================
# DuoRecall forward: wrap DuoAttention's one_way_reordered forward
# ============================================================

def _detect_duo_attrs(self):
    """Detect DuoAttention attributes from the module."""
    num_full_heads = getattr(self, 'num_full_attn_head', 0)
    num_stream_heads = getattr(self, 'num_streaming_attn_head', 0)
    num_kv_heads = getattr(self, 'num_key_value_heads',
                           getattr(self, 'num_kv_heads',
                                   getattr(self, 'num_heads', 32)))
    num_q_heads = getattr(self, 'num_heads', num_kv_heads * 
                          getattr(self, 'num_key_value_groups', 1))
    num_kv_groups = num_q_heads // num_kv_heads if num_kv_heads > 0 else 1
    head_dim = getattr(self, 'head_dim', 128)
    return num_full_heads, num_stream_heads, num_kv_heads, num_q_heads, num_kv_groups, head_dim


def build_recall_wrapped_forward(module):
    """
    Create a wrapped forward for DuoAttention that records recall information.
    """

    def _recall_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # Determine prefill vs decode
        is_prefill = q_len > 1
        is_decode = q_len == 1 and past_key_value is not None

        # ---- Prefill: reset and store KV for shadow ----
        if is_prefill:
            self._recall_step = 0
            if not getattr(self, '_recall_prefill_done', False):
                self._recall_prefill_done = True
                self._recall_prefill_k = self.k_proj(hidden_states).detach().clone()
                self._recall_prefill_pos = position_ids.detach().clone() if position_ids is not None else None

        # ---- Call original forward ----
        output = self._recall_original_forward(
            hidden_states, attention_mask, position_ids, past_key_value,
            output_attentions, use_cache, **kwargs
        )

        # ---- Post-decode: compute recall ----
        if is_decode and hasattr(self, '_recall_prefill_k'):
            self._recall_step += 1
            step = self._recall_step - 1
            num_full_heads, _, num_kv_heads, num_q_heads, num_kv_groups, head_dim = _detect_duo_attrs(self)
            sink = self._recall_sink_size
            recent = self._recall_recent_size

            prefill_k = self._recall_prefill_k  # [1, prefill_len, num_kv_heads * head_dim]
            prefill_len = prefill_k.shape[1]
            total_len = prefill_len + step + 1  # +1 for current decode token

            # Get current step Q/K (pre-RoPE)
            q_hidden = hidden_states[:, -1:, :]
            k_step = self.k_proj(q_hidden)

            # Build full K and Q (pre-RoPE)
            k_flat = torch.cat([prefill_k, k_step], dim=1)  # [1, seq_len, num_kv_heads * head_dim]
            total_len = k_flat.shape[1]  # derive actual total length from the tensor
            k_full = k_flat.view(1, total_len, num_kv_heads, head_dim).transpose(1, 2)  # [1, num_kv_heads, total_len, head_dim]

            q_step = self.q_proj(q_hidden)  # [1, 1, num_q_heads * head_dim]
            q_step = q_step.view(1, 1, num_q_heads, head_dim).transpose(1, 2)  # [1, num_q_heads, 1, head_dim]

            # Apply RoPE manually using self.rotary_emb
            device = k_full.device
            pos_ids = torch.arange(total_len, device=device).view(1, -1)
            dummy_v = torch.zeros(1, total_len, num_kv_heads, head_dim, device=device)
            cos_k, sin_k = self.rotary_emb(dummy_v, pos_ids)
            # cos_k: [1, total_len, head_dim] -> unsqueeze for head dim
            cos_k = cos_k.unsqueeze(1)  # [1, 1, total_len, head_dim]
            sin_k = sin_k.unsqueeze(1)
            k_rope = k_full * cos_k + _rotate_half(k_full) * sin_k

            # Q RoPE: only last position
            dummy_v_q = torch.zeros(1, 1, num_q_heads, head_dim, device=device)
            cos_q, sin_q = self.rotary_emb(dummy_v_q, pos_ids[:, -1:])
            cos_q = cos_q.unsqueeze(1)  # [1, 1, 1, head_dim]
            sin_q = sin_q.unsqueeze(1)
            q_rope = q_step * cos_q + _rotate_half(q_step) * sin_q

            # Full attention
            k_rope_full = repeat_kv(k_rope, num_kv_groups)
            attn_full = (q_rope.float() @ k_rope_full.float().transpose(-1, -2)) / math.sqrt(head_dim)
            attn_full = F.softmax(attn_full, dim=-1)

            # Accessible positions for streaming heads
            accessible = set()
            for i in range(min(sink, total_len)):
                accessible.add(i)
            for i in range(max(sink, total_len - recent), total_len - 1):
                accessible.add(i)
            accessible.add(total_len - 1)  # current token is always accessible

            # Ensure slot
            if step >= len(current_sample_stats):
                current_sample_stats.append({})

            # Compute recall per streaming head
            for h in range(num_q_heads):
                kv_head_idx = (h * num_kv_heads) // num_q_heads
                if kv_head_idx >= num_full_heads:
                    # Reshape to [1, total_len] (expected by _compute_recall_per_head)
                    attn_h = attn_full[:, h, :, :].squeeze().unsqueeze(0)  # [1, total_len]
                    r100, rk, sar = _compute_recall_per_head(
                        attn_h, accessible)
                    current_sample_stats[step][h] = (r100, rk, sar, total_len)

        return output

    return _recall_forward


# ============================================================
# Entry points
# ============================================================

def enable_duo_recall(model, sink_size=64, recent_size=256):
    """
    Apply recall tracking hooks to all DuoAttention layers.
    """
    print(f"[DuoRecall] Enabling recall tracking: sink={sink_size}, recent={recent_size}")
    count = 0
    for module in model.modules():
        if hasattr(module, "full_attention_heads") and hasattr(module, "num_heads"):
            module._recall_enabled = True
            module._recall_sink_size = sink_size
            module._recall_recent_size = recent_size
            module._recall_step = 0
            module._recall_prefill_done = False

            if not hasattr(module, '_recall_original_forward'):
                module._recall_original_forward = module.forward

            module.forward = build_recall_wrapped_forward(module).__get__(module, type(module))
            count += 1

    print(f"[DuoRecall] Enabled recall tracking on {count} DuoAttention layers.")
    return count


def disable_duo_recall(model):
    """Restore original forward functions."""
    count = 0
    for module in model.modules():
        if hasattr(module, '_recall_enabled') and hasattr(module, '_recall_original_forward'):
            module.forward = module._recall_original_forward
            del module._recall_enabled
            del module._recall_prefill_done
            del module._recall_step
            del module._recall_prefill_k
            if hasattr(module, '_recall_prefill_pos'):
                del module._recall_prefill_pos
            if hasattr(module, '_recall_original_forward'):
                del module._recall_original_forward
            count += 1
    print(f"[DuoRecall] Recall tracking disabled on {count} layers, original forwards restored.")