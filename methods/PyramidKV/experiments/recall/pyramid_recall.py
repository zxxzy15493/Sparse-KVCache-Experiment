"""
Recall tracking for PyramidKV.

Follows CakeKV's approach: reuses query_states and key_states already
computed inside the forward pass.  No separate q_proj/k_proj/RoPE needed.

- Each layer maintains its own shadow key and computes its own recall.
- Selected sets are per-layer (PyramidKV per-layer budget).
"""

import sys
import torch
import torch.nn.functional as F
import math
import time
import numpy as np
from typing import List
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb
from pyramidkv.pyramidkv_utils import PyramidKVCluster, init_pyramidkv

# ============================================================
# Global registry
# ============================================================
stats_registry: List[list] = []
current_sample_stats: list = []

def init_new_sample_registry():
    global current_sample_stats
    if current_sample_stats:
        stats_registry.append(current_sample_stats)
    current_sample_stats = []

def seal_and_save_stats(sample_idx=None):
    global current_sample_stats
    stats_registry.append((sample_idx, current_sample_stats))
    current_sample_stats = []

# ============================================================
# Patched update_kv — saves query_states, key_states + _selected_indices
# (All already q_proj'd, k_proj'd, RoPE'd by the original forward)
# ============================================================

def _single_pass_update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups, tm=None):
    assert key_states.shape[-2] == query_states.shape[-2]
    bsz, num_heads, q_len, head_dim = query_states.shape
    _, num_kv_heads, _, head_dim_k = key_states.shape
    window_size = self.window_size
    window_start = q_len - window_size

    key_states_orig = key_states      # [bsz, num_kv_heads, q_len, head_dim] — already RoPE'd
    value_states_orig = value_states

    # Save for recall (pre-compression, already projected + RoPE'd)
    self._recall_key = key_states_orig.clone()
    self._recall_query = query_states      # [bsz, num_heads, q_len, head_dim]
    self._recall_is_prefill = (q_len > 1)

    if num_kv_heads != num_heads:
        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)

    if q_len < self.max_capacity_prompt:
        self._selected_indices = [set(range(q_len)) for _ in range(num_kv_heads)]
        return key_states_orig, value_states_orig

    attn_weights = torch.matmul(
        query_states[..., -window_size:, :], key_states.transpose(2, 3)
    ) / math.sqrt(head_dim)

    mask = torch.full((window_size, window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    mask_cond = torch.arange(window_size, device=attn_weights.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(window_size, 1), 0)
    attn_weights[:, :, -window_size:, -window_size:] += mask[None, None, :, :]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights_sum = attn_weights[:, :, -window_size:, :-window_size].sum(dim=-2)

    if self.pooling == 'avgpool':
        attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size,
                                  padding=self.kernel_size // 2, stride=1)
    elif self.pooling == 'maxpool':
        attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size,
                                  padding=self.kernel_size // 2, stride=1)
    else:
        raise ValueError('Pooling not supported')

    min_num = (self.max_capacity_prompt - window_size) // self.beta
    max_num = (self.max_capacity_prompt - window_size) * 2 - min_num
    if max_num >= q_len - window_size:
        max_num = q_len - window_size
        min_num = (self.max_capacity_prompt - window_size) * 2 - max_num
    steps = (max_num - min_num) // self.num_hidden_layers
    max_capacity_prompt = max_num - self.layer_idx * steps

    if q_len < (self.max_capacity_prompt - window_size) * 2:
        topk = self.max_capacity_prompt - window_size
    else:
        topk = max_capacity_prompt

    indices_28 = attn_cache.topk(topk, dim=-1).indices
    indices_4 = indices_28[:, ::num_key_value_groups, :]
    indices_4_exp = indices_4.unsqueeze(-1).expand(-1, -1, -1, head_dim_k)

    k_past_compress = key_states_orig[:, :, :-window_size, :].gather(dim=2, index=indices_4_exp)
    v_past_compress = value_states_orig[:, :, :-window_size, :].gather(dim=2, index=indices_4_exp)
    k_cur = key_states_orig[:, :, -window_size:, :]
    v_cur = value_states_orig[:, :, -window_size:, :]

    selected = []
    for h in range(num_kv_heads):
        h_indices = indices_4[0, h].cpu().tolist()
        selected.append(set(h_indices + list(range(window_start, q_len))))
    self._selected_indices = selected

    return (torch.cat([k_past_compress, k_cur], dim=2),
            torch.cat([v_past_compress, v_cur], dim=2))

PyramidKVCluster.update_kv = _single_pass_update_kv


# ============================================================
# Recall computation (batched GPU ops, single sync)
# ============================================================

def _compute_recall_per_head(probs, selected_list, kv_len_comp, sel_tensor_cache=None):
    """Returns [num_heads, 3] numpy array: recall_100, recall_k, selected_attn_ratio.
    
    sel_tensor_cache: if provided, reuse cached tensors (avoids set→tensor per decode step)
    Returns updated cache.
    """
    num_heads = probs.shape[0]
    seq_len = probs.shape[1]
    device = probs.device

    actual_100 = min(100, seq_len)
    actual_k = min(kv_len_comp, seq_len)

    if actual_k == 0:
        return np.zeros((num_heads, 3), dtype=np.float32), sel_tensor_cache

    top100_idx = torch.topk(probs, actual_100, dim=-1).indices
    topk_idx = torch.topk(probs, actual_k, dim=-1).indices

    recall_100_vals = torch.zeros(num_heads, device=device)
    recall_k_vals = torch.zeros(num_heads, device=device)
    attn_ratio_vals = torch.zeros(num_heads, device=device)

    if sel_tensor_cache is None:
        sel_tensor_cache = [None] * num_heads

    for h in range(num_heads):
        s = selected_list[h]
        if not s:
            continue
        # Reuse cached tensor if lengths match (decode: +1 token each step, prefill: rebuild)
        if sel_tensor_cache[h] is not None and sel_tensor_cache[h].numel() == len(s):
            sel = sel_tensor_cache[h]
        else:
            sel = torch.tensor(sorted(s), device=device, dtype=torch.long).clamp(0, seq_len - 1)
            sel_tensor_cache[h] = sel
        recall_100_vals[h] = torch.isin(top100_idx[h], sel).sum()
        recall_k_vals[h] = torch.isin(topk_idx[h], sel).sum()
        attn_ratio_vals[h] = probs[h][sel].sum()

    results = torch.stack([recall_100_vals / actual_100,
                           recall_k_vals / actual_k,
                           attn_ratio_vals], dim=1).cpu().numpy()
    return results.astype(np.float32), sel_tensor_cache


# ============================================================
# Model-level recall hook (one per attention module)
# ============================================================

def make_recall_class_forward(original_pyramidkv_forward, num_layers, attention_modules):
    """Wrap PyramidKV's FA2 forward with recall computation.

    Uses query_states and key_states already computed inside the forward
    (no extra q_proj/k_proj/RoPE), same approach as CakeKV.
    """
    shared = {'step': 0}

    _timer = {'recall_ms': 0.0, 'model_ms': 0.0}

    def recall_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        global current_sample_stats
        bsz, q_len, _ = hidden_states.size()
        is_prefill = past_key_value is None or q_len > 1
        enabled = hasattr(self, '_recall_enabled')

        # ---- Prefill setup ----
        if enabled and is_prefill:
            init_pyramidkv(self, num_hidden_layers=num_layers)

            if self.layer_idx == 0:
                _timer['model_ms'] = 0.0
                _timer['recall_ms'] = 0.0
                shared['step'] = 0

        # ---- Run PyramidKV FA2 forward (calls patched _single_pass_update_kv) ----
        # original_pyramidkv_forward is a bound method (got via module.forward),
        # so self is already bound — do NOT pass self again.
        _tm0 = time.time()
        output = original_pyramidkv_forward(
            hidden_states, attention_mask, position_ids, past_key_value,
            output_attentions, use_cache, **kwargs
        )
        _tm1 = time.time()
        if enabled:
            _timer['model_ms'] += (_tm1 - _tm0) * 1000

        if not enabled:
            return output

        # ---- Post-prefill: step 0 recall (last layer runs it for all layers) ----
        if is_prefill:
            # _single_pass_update_kv has already saved to kv_cluster:
            #   kv_cluster._recall_key   (full pre-compression key, RoPE'd)
            #   kv_cluster._recall_query (query_states)
            #   kv_cluster._selected_indices
            #   kv_cluster._recall_is_prefill = True
            kv = self.kv_cluster

            # Set shadow key for this layer
            self._recall_shadow_key = kv._recall_key
            # Save selected_indices on attention module for decode steps (kv_cluster is recreated each step)
            self._recall_selected_indices = kv._selected_indices

            # Last layer: compute step 0 recall for ALL layers
            if self.layer_idx == num_layers - 1:
                # Build shared shadow_expanded once (same K for all layers, same dtype as model)
                shared['shadow_expanded'] = repeat_kv(
                    self._recall_shadow_key, self.num_key_value_groups
                ).float()  # float for matmul precision

                current_sample_stats.append({})
                for mod in attention_modules:
                    if not hasattr(mod, '_recall_shadow_key'):
                        continue
                    mkv = mod.kv_cluster
                    q = mkv._recall_query[:, :, -1:, :]  # last position query
                    attn = (q.float() @ shared['shadow_expanded'].transpose(-1, -2)) / math.sqrt(self.head_dim)
                    probs = F.softmax(attn, dim=-1)[0, :, 0, :]

                    expanded = [mkv._selected_indices[h // mod.num_key_value_groups]
                                for h in range(mod.num_heads)]
                    kv_len_comp = max(len(s) for s in mkv._selected_indices)
                    metrics, cache = _compute_recall_per_head(probs, expanded, kv_len_comp)
                    mod._sel_tensor_cache = cache
                    current_sample_stats[0][mod.layer_idx] = metrics
                shared['step'] = 1  # step 0 taken by prefill

        # ---- Post-decode: extend shadow key + compute recall ----
        if not is_prefill:
            # kv_cluster is freshly created by init_pyramidkv each step during decode,
            # and update_kv is never called in the decode path, so we compute the
            # current token's key/query from hidden_states ourselves.
            kv = self.kv_cluster
            # Only layer 0 advances the global step (same as CakeKV)
            if self.layer_idx == 0:
                step = shared['step']
                shared['step'] += 1
            else:
                step = shared['step'] - 1

            if step >= len(current_sample_stats):
                current_sample_stats.append({})

            # Compute current token key/query from hidden_states
            bsz, q_len, hdim = hidden_states.shape
            cur_q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            cur_k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            # Apply RoPE — use value_states for cos/sin (any RoPE'd tensor works)
            cos, sin = self.rotary_emb(cur_k, position_ids[:, -q_len:] if position_ids is not None else None)
            cur_q, cur_k = apply_rotary_pos_emb(cur_q, cur_k, cos, sin)

            # Extend shadow key
            self._recall_shadow_key = torch.cat([self._recall_shadow_key, cur_k], dim=2)

            # The new decode token is always kept in KV cache — add it to selected set
            new_pos = self._recall_shadow_key.shape[2] - 1
            for s in self._recall_selected_indices:
                s.add(new_pos)

            # Compute recall: this layer's query vs full shadow key
            sk = repeat_kv(self._recall_shadow_key, self.num_key_value_groups)
            attn = (cur_q.float() @ sk.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
            probs = F.softmax(attn, dim=-1)[0, :, 0, :]

            expanded = [self._recall_selected_indices[h // self.num_key_value_groups]
                        for h in range(self.num_heads)]
            kv_len_comp = max(len(s) for s in self._recall_selected_indices)
            metrics, cache = _compute_recall_per_head(probs, expanded, kv_len_comp,
                                                       sel_tensor_cache=self._sel_tensor_cache)
            self._sel_tensor_cache = cache
            current_sample_stats[step][self.layer_idx] = metrics

        return output

    return recall_forward


# ============================================================
# Enable function
# ============================================================

def enable_pyramid_recall(model, check_recall=False, **compress_args):
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaFlashAttention2, LlamaSdpaAttention

    model_type = model.config.model_type.lower()
    qwen_attn_classes = ()
    if "qwen" in model_type:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2FlashAttention2, Qwen2SdpaAttention
        qwen_attn_classes = (Qwen2Attention, Qwen2FlashAttention2, Qwen2SdpaAttention)

    # Collect all attention modules
    attention_modules = []
    for module in model.modules():
        if isinstance(module, (LlamaAttention, LlamaFlashAttention2, LlamaSdpaAttention) + qwen_attn_classes):
            attention_modules.append(module)

    num_layers = len(attention_modules)
    if num_layers == 0:
        raise ValueError("Could not find attention modules")

    # Monkey-patch each attention module's forward
    for i, module in enumerate(attention_modules):
        module.layer_idx = i
        module._recall_enabled = True
        module._recall_step = 0
        module.forward = make_recall_class_forward(module.forward, num_layers, attention_modules).__get__(module)

    print(f"PyramidKV Recall enabled. Patched {num_layers} attention layers.")