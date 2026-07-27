import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
import types
import warnings
from typing import Optional, Union, List
from importlib.metadata import version
from flash_attn import flash_attn_func

stats_registry = []
current_sample_stats = []

def init_new_sample_registry():
    global current_sample_stats
    if current_sample_stats:
        stats_registry.append(current_sample_stats)
    current_sample_stats = []

class SnapKVCache:
    def __init__(self, max_capacity_prompt=1024, window_size=32, kernel_size=7, pooling='avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt 
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.cache_size = max_capacity_prompt
        
        self.select_idx = None
        self._global_step = 0
        self.absolute_indices = None
        self.total_processed_tokens = 0 
        self.shadow_key = None 
        self.hh_score = None

@torch.no_grad()
def common_diagnostic_and_evict(self, query_states, key_states, value_states, past_key_value, bsz, q_len, sin, cos):
    device = query_states.device
    num_heads = query_states.shape[1]
    head_dim = query_states.shape[-1]
    
    check_recall = getattr(self, "check_recall", False)

    if q_len > 1:
        kv_seq_len = key_states.shape[2]
        
        self.kv_cache.absolute_indices = torch.arange(kv_seq_len, device=device).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, -1)
        self.kv_cache.total_processed_tokens = kv_seq_len
        
        if check_recall:
            self.kv_cache.shadow_key = key_states.clone()

        if kv_seq_len <= self.kv_cache.cache_size:
            k_comp, v_comp = key_states, value_states
            global_indices_spliced = self.kv_cache.absolute_indices
        else:
            window_size = self.kv_cache.window_size
            attn_weights = torch.matmul(query_states[..., -window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            
            mask = torch.full((window_size, window_size), torch.finfo(attn_weights.dtype).min, device=device)
            mask_cond = torch.arange(mask.size(-1), device=device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            attn_weights[:, :, -window_size:, -window_size:] += mask[None, None, :, :]

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -window_size:, : -window_size].sum(dim=-2)
            
            if self.kv_cache.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kv_cache.kernel_size, padding=self.kv_cache.kernel_size//2, stride=1)
            elif self.kv_cache.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kv_cache.kernel_size, padding=self.kv_cache.kernel_size//2, stride=1)
            
            keep_capacity = self.kv_cache.cache_size - window_size
            topk_indices = attn_cache.topk(keep_capacity, dim=-1).indices
            topk_indices, _ = torch.sort(topk_indices, dim=-1)
            
            indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-window_size, :].gather(dim=2, index=indices)
            v_past_compress = value_states[:, :, :-window_size, :].gather(dim=2, index=indices)
            
            k_cur = key_states[:, :, -window_size:, :]
            v_cur = value_states[:, :, -window_size:, :]
            k_comp = torch.cat([k_past_compress, k_cur], dim=2)
            v_comp = torch.cat([v_past_compress, v_cur], dim=2)
            
            window_indices = torch.arange(kv_seq_len - window_size, kv_seq_len, device=device).view(1, 1, -1).expand(bsz, num_heads, window_size)
            keep_idx = torch.cat([topk_indices, window_indices], dim=-1)
            global_indices_spliced = torch.gather(self.kv_cache.absolute_indices, -1, keep_idx)
            self.kv_cache.absolute_indices = global_indices_spliced

            past_key_value.update(k_comp, v_comp, self.layer_idx, {"sin": sin, "cos": cos})

    else:
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, {"sin": sin, "cos": cos})
        
        if check_recall:
            self.kv_cache.shadow_key = torch.cat([self.kv_cache.shadow_key, key_states[:, :, -1:, :]], dim=2)
            
            new_coord = torch.tensor([[self.kv_cache.total_processed_tokens]], device=device).expand(bsz, num_heads, 1)
            self.kv_cache.absolute_indices = torch.cat([self.kv_cache.absolute_indices, new_coord], dim=-1)
            self.kv_cache.total_processed_tokens += 1
            
        global_indices_spliced = self.kv_cache.absolute_indices

    current_select_global = global_indices_spliced[0]

    if check_recall:
        global current_sample_stats
        step = self.kv_cache._global_step
        if self.layer_idx == 0:
            if step >= len(current_sample_stats):
                current_sample_stats.append({})
        
        q_current = query_states[:, :, -1:, :]
        raw_attn_global = torch.matmul(q_current, self.kv_cache.shadow_key.transpose(-1, -2)) / math.sqrt(head_dim)
        probs_global = F.softmax(raw_attn_global, dim=-1, dtype=torch.float32)
        full_probs_global = probs_global[0, :, 0, :]  
        
        total_uncompressed_len = self.kv_cache.shadow_key.shape[2]
        
        try:
            current_kv_len = k_comp.shape[2]
        except NameError:
            current_kv_len = key_states.shape[2]
        actual_k = min(current_kv_len, total_uncompressed_len)
        actual_100 = min(100, total_uncompressed_len)
        
        _, true_topk_indices = torch.topk(full_probs_global, actual_k, dim=-1)
        _, true_top100_indices = torch.topk(full_probs_global, actual_100, dim=-1)
        
        metrics_matrix = np.zeros((num_heads, 3), dtype=np.float32)
        for h in range(num_heads):
            true_topk_set = set(true_topk_indices[h].tolist())
            true_top100_set = set(true_top100_indices[h].tolist())
            filtered_set = set(current_select_global[h].tolist())
            
            selected_attn_ratio = full_probs_global[h, current_select_global[h]].sum().item()
            recall_100 = len(filtered_set.intersection(true_top100_set)) / 100.0
            recall_k = len(filtered_set.intersection(true_topk_set)) / float(current_kv_len) if current_kv_len > 0 else 0.0
            
            metrics_matrix[h] = [recall_100, recall_k, selected_attn_ratio]
            
        current_sample_stats[step][self.layer_idx] = metrics_matrix

    if self.layer_idx == 0:
        self.kv_cache._global_step += 1

    attn_out = flash_attn_func(
        query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), causal=True
    ).reshape(bsz, q_len, self.hidden_size)

    return self.o_proj(attn_out), None, past_key_value

def llama_forward_recall_patch(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, **kwargs):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    return common_diagnostic_and_evict(self, query_states, key_states, value_states, past_key_value, bsz, q_len, sin, cos)

def qwen_forward_recall_patch(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, **kwargs):
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    return common_diagnostic_and_evict(self, query_states, key_states, value_states, past_key_value, bsz, q_len, sin, cos)

def check_transformers_compatibility():
    try:
        transformers_version = version("transformers")
    except Exception:
        transformers_version = "0.0.0"
    tested_versions = ['4.45', '4.46']
    if not any(v in transformers_version for v in tested_versions):
        warnings.warn(f"Transformers version {transformers_version} compatibility warning.")

def enable_snapkv_recall(model, check_recall=False, window_sizes=None, max_capacity_prompts=None, kernel_sizes=None, pooling='avgpool'):
    check_transformers_compatibility()

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model_layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "encoder") and hasattr(model.transformer.encoder, "layers"):
        model_layers = model.transformer.encoder.layers
    else:
        raise ValueError("Could not find layers in model")
    num_layers = len(model_layers)

    if not isinstance(window_sizes, list): window_sizes = [window_sizes] * num_layers
    if not isinstance(max_capacity_prompts, list): max_capacity_prompts = [max_capacity_prompts] * num_layers
    if not isinstance(kernel_sizes, list): kernel_sizes = [kernel_sizes] * num_layers

    try:
        from transformers.models.llama.modeling_llama import LlamaAttention, LlamaSdpaAttention, LlamaFlashAttention2
        llama_classes = (LlamaAttention, LlamaSdpaAttention, LlamaFlashAttention2)
    except ImportError:
        llama_classes = ()

    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2SdpaAttention, Qwen2FlashAttention2
        qwen_classes = (Qwen2Attention, Qwen2SdpaAttention, Qwen2FlashAttention2)
    except ImportError:
        qwen_classes = ()

    patched_layers = 0
    for name, module in model.named_modules():
        if (llama_classes and isinstance(module, llama_classes)) or (qwen_classes and isinstance(module, qwen_classes)):
            l_idx = int(name.split('.')[-2])
            
            module.kv_cache = SnapKVCache(
                max_capacity_prompt=max_capacity_prompts[l_idx], 
                window_size=window_sizes[l_idx],
                kernel_size=kernel_sizes[l_idx],
                pooling=pooling
            )
            module.layer_idx = l_idx
            module.check_recall = check_recall
            
            if llama_classes and isinstance(module, llama_classes):
                module.forward = types.MethodType(llama_forward_recall_patch, module)
            else:
                module.forward = types.MethodType(qwen_forward_recall_patch, module)
            patched_layers += 1

    print(f"SnapKV Adaptive Dict-Unpacking Patch Enabled. Patched {patched_layers} attention layers.")