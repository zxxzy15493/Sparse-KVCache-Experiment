import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
import math
import numpy as np
import types
from transformers.utils import is_flash_attn_2_available
from transformers.cache_utils import Cache

from headkv.monkeypatch import replace_llama as _hkv_replace_llama
from headkv.monkeypatch import replace_qwen2 as _hkv_replace_qwen2
from headkv.snapkv_utils import DynamicCacheSplitHeadFlatten

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func

stats_registry = []
current_sample_stats = []

def init_new_sample_registry():
    global current_sample_stats
    if current_sample_stats:
        stats_registry.append(current_sample_stats)
    current_sample_stats = []


# === recall hooks ===
def _recall_prefill(self, key_states, query_states):
    self._recall_shadow_key = key_states.clone()
    self._recall_step = 0
    num_heads = query_states.shape[1]
    ws = self.kv_cluster.window_size
    q_len = query_states.shape[2]

    # Track selected indices (per-head sets of compressed token positions)
    # Use the actual selected indices stored by the compression module
    if hasattr(self.kv_cluster, '_selected_indices') and self.kv_cluster._selected_indices is not None:
        self._recall_selected = [set(idx.tolist()) for idx in self.kv_cluster._selected_indices]
    else:
        # fallback: assume all tokens are kept (no compression or unrecognized cluster type)
        self._recall_selected = [set(range(q_len)) for _ in range(num_heads)]


def _recall_decode(self, query_states, key_states):
    global current_sample_stats
    step = self._recall_step
    if self.layer_idx == 0:
        if step >= len(current_sample_stats):
            current_sample_stats.append({})

    num_heads = query_states.shape[1]
    head_dim = query_states.shape[-1]

    self._recall_shadow_key = torch.cat(
        [self._recall_shadow_key, key_states[:, :, -1:, :]], dim=2
    )

    total_len = self._recall_shadow_key.shape[2]
    # The decode token is always kept — add it to the selected set
    new_pos = total_len - 1
    for s in self._recall_selected:
        s.add(new_pos)
    kv_len = sum(len(s) for s in self._recall_selected) // num_heads

    q_current = query_states[:, :, -1:, :]
    raw_attn = torch.matmul(q_current, self._recall_shadow_key.transpose(-1, -2)) / math.sqrt(head_dim)
    full_probs = F.softmax(raw_attn, dim=-1, dtype=torch.float32)[0, :, 0, :]

    actual_k = min(kv_len, total_len)
    actual_100 = min(100, total_len)
    topk_idx = torch.topk(full_probs, actual_k, dim=-1).indices
    top100_idx = torch.topk(full_probs, actual_100, dim=-1).indices

    metrics = np.zeros((num_heads, 3), dtype=np.float32)
    for h in range(num_heads):
        topk_set = set(topk_idx[h].tolist())
        top100_set = set(top100_idx[h].tolist())
        sel_set = self._recall_selected[h]
        attn_ratio = full_probs[h, sorted(sel_set)].sum().item()
        r100 = len(sel_set & top100_set) / 100.0
        rk = len(sel_set & topk_set) / float(actual_k) if actual_k > 0 else 0.0
        metrics[h] = [r100, rk, attn_ratio]

    current_sample_stats[step][self.layer_idx] = metrics
    if self.layer_idx == 0:
        self._recall_step += 1


def _headkv_forward_ada(self, hidden_states, attention_mask=None, position_ids=None,
                        past_key_value=None, output_attentions=False, use_cache=False,
                        cache_position=None, position_embeddings=None, **kwargs):
    from headkv.snapkv_utils import init_headkv
    from headkv.adaptive_llama_hijack import _safe_decode_from_flatten_cache, _refresh_varlen_decode_metadata
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    init_headkv(self)
    if past_key_value is not None and not isinstance(past_key_value, DynamicCacheSplitHeadFlatten):
        if isinstance(past_key_value, Cache) and hasattr(past_key_value, "to_legacy_cache"):
            past_key_value = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_value.to_legacy_cache())
        else:
            past_key_value = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_value)

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if hasattr(self, "kv_seq_len") and self.kv_seq_len != 0:
            kv_seq_len += self.kv_seq_len
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = position_embeddings if position_embeddings is not None else self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    dropout_rate = self.attention_dropout if self.training else 0.0

    from transformers.models.llama.modeling_llama import _flash_attention_forward
    if past_key_value is None:
        attn_output = _flash_attention_forward(
            query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2),
            attention_mask if q_len > 1 else None, q_len,
            position_ids=position_ids, dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=getattr(self, "_flash_attn_uses_top_left_mask", False),
            is_causal=getattr(self, "is_causal", True),
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    else:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        is_prefill = q_len > 1
        if isinstance(past_key_value, DynamicCacheSplitHeadFlatten):
            has_cache = (len(past_key_value.key_cache) > self.layer_idx
                         and past_key_value.key_cache[self.layer_idx] is not None)
            is_prefill = is_prefill or (not has_cache)
        if is_prefill:
            self.kv_seq_len = kv_seq_len
            k_comp, v_comp = self.kv_cluster.update_kv(key_states, query_states, value_states)
            past_key_value.update(k_comp, v_comp, self.layer_idx, cache_kwargs)
            attn_output = _flash_attention_forward(
                query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2),
                attention_mask if q_len > 1 else None, q_len,
                position_ids=position_ids, dropout=dropout_rate,
                sliding_window=getattr(self, "sliding_window", None),
                use_top_left_mask=getattr(self, "_flash_attn_uses_top_left_mask", False),
                is_causal=getattr(self, "is_causal", True),
            )
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
            # Recall hook: save shadow key & selected indices
            _recall_prefill(self, key_states, query_states)
        else:
            self.kv_seq_len += q_len
            cache_kwargs["head_lens"] = self.kv_cluster.head_lens
            cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
            new_key = key_states  # save 4D key before update (q_len=1, the new token's key)
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            self.kv_cluster.head_lens = self.kv_cluster.head_lens + q_len
            _refresh_varlen_decode_metadata(self.kv_cluster)
            # Recall hook: pass the saved 4D key (contains the new token's key)
            _recall_decode(self, query_states, new_key)
            query_states = query_states.view(-1, 1, self.head_dim)
            key_states = key_states.view(-1, 1, self.head_dim)
            value_states = value_states.view(-1, 1, self.head_dim)
            use_fast = (is_flash_attn_2_available() and not getattr(self, "_disable_varlen_decode", False))
            if use_fast:
                try:
                    attn_output = flash_attn_varlen_func(
                        query_states, key_states, value_states,
                        self.kv_cluster.cu_qlen, self.kv_cluster.cu_klen,
                        1, int(self.kv_cluster.max_seqlen_k.item()) if torch.is_tensor(self.kv_cluster.max_seqlen_k) else self.kv_cluster.max_seqlen_k,
                        causal=True,
                    ).reshape(bsz, self.num_heads, q_len, self.head_dim)
                    attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
                except RuntimeError:
                    self._disable_varlen_decode = True
                    attn_output = _safe_decode_from_flatten_cache(
                        query_states, key_states, value_states, self.kv_cluster.cu_klen,
                        bsz, self.num_heads, q_len, self.head_dim, self.hidden_size)
            else:
                attn_output = _safe_decode_from_flatten_cache(
                    query_states, key_states, value_states, self.kv_cluster.cu_klen,
                    bsz, self.num_heads, q_len, self.head_dim, self.hidden_size)

    return self.o_proj(attn_output), None, past_key_value


def _headkv_forward_reason(self, hidden_states, attention_mask=None, position_ids=None,
                           past_key_value=None, output_attentions=False, use_cache=False,
                           cache_position=None, position_embeddings=None, **kwargs):
    from headkv.snapkv_utils import init_reason_snapkv
    from headkv.adaptive_llama_hijack import _safe_decode_from_flatten_cache, _refresh_varlen_decode_metadata
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    init_reason_snapkv(self)
    if past_key_value is not None and not isinstance(past_key_value, DynamicCacheSplitHeadFlatten):
        if isinstance(past_key_value, Cache) and hasattr(past_key_value, "to_legacy_cache"):
            past_key_value = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_value.to_legacy_cache())
        else:
            past_key_value = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_value)

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if hasattr(self, "kv_seq_len") and self.kv_seq_len != 0:
            kv_seq_len += self.kv_seq_len
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = position_embeddings if position_embeddings is not None else self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    dropout_rate = self.attention_dropout if self.training else 0.0

    from transformers.models.llama.modeling_llama import _flash_attention_forward
    if past_key_value is None:
        attn_output = _flash_attention_forward(
            query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2),
            attention_mask if q_len > 1 else None, q_len,
            position_ids=position_ids, dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=getattr(self, "_flash_attn_uses_top_left_mask", False),
            is_causal=getattr(self, "is_causal", True),
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    else:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        is_prefill = q_len > 1
        if isinstance(past_key_value, DynamicCacheSplitHeadFlatten):
            has_cache = (len(past_key_value.key_cache) > self.layer_idx
                         and past_key_value.key_cache[self.layer_idx] is not None)
            is_prefill = is_prefill or (not has_cache)
        if is_prefill:
            self.kv_seq_len = kv_seq_len
            k_comp, v_comp = self.kv_cluster.update_kv(key_states, query_states, value_states)
            past_key_value.update(k_comp, v_comp, self.layer_idx, cache_kwargs)
            attn_output = _flash_attention_forward(
                query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2),
                attention_mask if q_len > 1 else None, q_len,
                position_ids=position_ids, dropout=dropout_rate,
                sliding_window=getattr(self, "sliding_window", None),
                use_top_left_mask=getattr(self, "_flash_attn_uses_top_left_mask", False),
                is_causal=getattr(self, "is_causal", True),
            )
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
            _recall_prefill(self, key_states, query_states)
        else:
            self.kv_seq_len += q_len
            cache_kwargs["head_lens"] = self.kv_cluster.head_lens
            cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
            new_key = key_states  # save 4D key before update (q_len=1, the new token's key)
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            self.kv_cluster.head_lens = self.kv_cluster.head_lens + q_len
            _refresh_varlen_decode_metadata(self.kv_cluster)
            # Recall hook: pass the saved 4D key (contains the new token's key)
            _recall_decode(self, query_states, new_key)
            query_states = query_states.view(-1, 1, self.head_dim)
            key_states = key_states.view(-1, 1, self.head_dim)
            value_states = value_states.view(-1, 1, self.head_dim)
            use_fast = (is_flash_attn_2_available() and not getattr(self, "_disable_varlen_decode", False))
            if use_fast:
                try:
                    attn_output = flash_attn_varlen_func(
                        query_states, key_states, value_states,
                        self.kv_cluster.cu_qlen, self.kv_cluster.cu_klen,
                        1, int(self.kv_cluster.max_seqlen_k.item()) if torch.is_tensor(self.kv_cluster.max_seqlen_k) else self.kv_cluster.max_seqlen_k,
                        causal=True,
                    ).reshape(bsz, self.num_heads, q_len, self.head_dim)
                    attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
                except RuntimeError:
                    self._disable_varlen_decode = True
                    attn_output = _safe_decode_from_flatten_cache(
                        query_states, key_states, value_states, self.kv_cluster.cu_klen,
                        bsz, self.num_heads, q_len, self.head_dim, self.hidden_size)
            else:
                attn_output = _safe_decode_from_flatten_cache(
                    query_states, key_states, value_states, self.kv_cluster.cu_klen,
                    bsz, self.num_heads, q_len, self.head_dim, self.hidden_size)

    return self.o_proj(attn_output), None, past_key_value


def _set_config_attrs(model, config_params, method):
    """Set compression config on every attention module's .config object."""
    try:
        from transformers.models.llama.modeling_llama import (
            LlamaAttention, LlamaSdpaAttention, LlamaFlashAttention2)
        llama_cls = (LlamaAttention, LlamaSdpaAttention, LlamaFlashAttention2)
    except ImportError:
        llama_cls = ()
    try:
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2Attention, Qwen2SdpaAttention, Qwen2FlashAttention2)
        qwen_cls = (Qwen2Attention, Qwen2SdpaAttention, Qwen2FlashAttention2)
    except ImportError:
        qwen_cls = ()

    for _, mod in model.named_modules():
        if (llama_cls and isinstance(mod, llama_cls)) or (qwen_cls and isinstance(mod, qwen_cls)):
            mod.config.window_size = config_params.get("window_size", 32)
            mod.config.kernel_size = config_params.get("kernel_size", 7)
            mod.config.pooling = config_params.get("pooling", "maxpool")
            mod.config.base_capacity = config_params.get("base_capacity", 1024)
            if method == "adativekv":
                mod.config.floor = config_params.get("floor", 0.1)
                mod.config.skip = config_params.get("skip", 2)
                mod.config.normalize = config_params.get("normalize", False)
            elif method == "reasonkv":
                mod.config.head_choice = config_params.get("head_choice", "reason")
                mod.config.beta = config_params.get("beta", 1.5)
                mod.config.temp = config_params.get("temp", 1.0)


def enable_headkv_recall(model, method="adativekv", check_recall=False, **config):
    """Enable HeadKV with recall tracking.

    1. Set compression config on model (params from json config file)
    2. Run HeadKV's monkeypatch (model forward + prepare_inputs + attention)
    3. Override attention forward with recall-enabled version
    """
    method_name = "AdativeKV" if method == "adativekv" else "ReasonKV"

    # Step 1: set config attrs so init_headkv/init_reason_snapkv can read them
    _set_config_attrs(model, config, method)

    # Step 2: HeadKV full monkeypatch
    _hkv_replace_llama(method_name)
    _hkv_replace_qwen2(method_name)

    # Step 3: override attention with recall-tracking forward
    fwd = _headkv_forward_ada if method == "adativekv" else _headkv_forward_reason

    try:
        from transformers.models.llama.modeling_llama import (
            LlamaAttention, LlamaSdpaAttention, LlamaFlashAttention2)
        llama_cls = (LlamaAttention, LlamaSdpaAttention, LlamaFlashAttention2)
    except ImportError:
        llama_cls = ()
    try:
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2Attention, Qwen2SdpaAttention, Qwen2FlashAttention2)
        qwen_cls = (Qwen2Attention, Qwen2SdpaAttention, Qwen2FlashAttention2)
    except ImportError:
        qwen_cls = ()

    patched = 0
    for _, mod in model.named_modules():
        if (llama_cls and isinstance(mod, llama_cls)) or (qwen_cls and isinstance(mod, qwen_cls)):
            mod.forward = types.MethodType(fwd, mod)
            patched += 1

    # monkeypatch also sets global class forwards — we don't use those, instance forwards win
    print(f"HeadKV Recall [{method.upper()}] enabled. {patched} layers with recall tracking.")