import torch
import torch.nn.functional as F
import math
import numpy as np
import types
from typing import Optional
from flash_attn import flash_attn_func

stats_registry = []
current_sample_stats = []

def init_new_sample_registry():
    global current_sample_stats
    if current_sample_stats:
        stats_registry.append(current_sample_stats)
    current_sample_stats = []

class StreamingLLMKVCache:
    def __init__(self, total_budget=1024, start_size=16, k_seq_dim=2, v_seq_dim=2):
        self.start_size = start_size
        self.recent_size = total_budget - start_size
        self.cache_size = total_budget
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.select_idx = None
        self._global_step = 0
        self.absolute_indices = None
        self.total_processed_tokens = 0 
        self.shadow_key = None 

@torch.no_grad()
def common_diagnostic_and_evict(self, query_states, key_states, value_states, past_key_value, bsz, q_len, sin, cos):
    device = query_states.device
    num_heads = query_states.shape[1]
    head_dim = query_states.shape[-1]
    
    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, {"sin": sin, "cos": cos})
    kv_seq_len = key_states.shape[2]
    
    if self.kv_cache.absolute_indices is None:
        self.kv_cache.absolute_indices = torch.arange(kv_seq_len, device=device).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, -1)
        self.kv_cache.total_processed_tokens = kv_seq_len
    else:
        new_len = kv_seq_len - self.kv_cache.absolute_indices.shape[-1]
        if new_len > 0:
            new_idx = torch.arange(self.kv_cache.total_processed_tokens, self.kv_cache.total_processed_tokens + new_len, device=device).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, -1)
            self.kv_cache.absolute_indices = torch.cat([self.kv_cache.absolute_indices, new_idx], dim=-1)
            self.kv_cache.total_processed_tokens += new_len

    check_recall = getattr(self, "check_recall", False)
    
    if check_recall:
        if self.kv_cache.shadow_key is None:
            self.kv_cache.shadow_key = key_states.clone()
        else:
            new_key = key_states[:, :, -q_len:, :]
            self.kv_cache.shadow_key = torch.cat([self.kv_cache.shadow_key, new_key], dim=2)

    if kv_seq_len <= self.kv_cache.cache_size:
        k_comp, v_comp = key_states, value_states
        current_select_global = self.kv_cache.absolute_indices[0]
    else:
        k_sink = key_states[:, :, :self.kv_cache.start_size, :]
        v_sink = value_states[:, :, :self.kv_cache.start_size, :]
        k_recent = key_states[:, :, -self.kv_cache.recent_size:, :]
        v_recent = value_states[:, :, -self.kv_cache.recent_size:, :]
        
        k_comp = torch.cat([k_sink, k_recent], dim=2)
        v_comp = torch.cat([v_sink, v_recent], dim=2)
        
        abs_sink = self.kv_cache.absolute_indices[:, :, :self.kv_cache.start_size]
        abs_recent = self.kv_cache.absolute_indices[:, :, -self.kv_cache.recent_size:]
        global_indices_sliced = torch.cat([abs_sink, abs_recent], dim=-1)
        current_select_global = global_indices_sliced[0]

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
        K = self.kv_cache.cache_size
        actual_k = min(K, total_uncompressed_len)
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
            recall_k = len(filtered_set.intersection(true_topk_set)) / float(K) if K > 0 else 0.0
            
            metrics_matrix[h] = [recall_100, recall_k, selected_attn_ratio]
            
        current_sample_stats[step][self.layer_idx] = metrics_matrix

    past_key_value.key_cache[self.layer_idx] = k_comp
    past_key_value.value_cache[self.layer_idx] = v_comp
    
    if kv_seq_len > self.kv_cache.cache_size:
        self.kv_cache.absolute_indices = global_indices_sliced

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

def enable_streaming_llm_recall(model, args):
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

    total_budget = args.recent_size + args.start_size
    start_size = 16

    patched_layers = 0
    for name, module in model.named_modules():
        if llama_classes and isinstance(module, llama_classes):
            module.kv_cache = StreamingLLMKVCache(total_budget=total_budget, start_size=start_size)
            module.layer_idx = int(name.split('.')[-2])
            module.check_recall = args.check_recall
            module.forward = types.MethodType(llama_forward_recall_patch, module)
            patched_layers += 1
        elif qwen_classes and isinstance(module, qwen_classes):
            module.kv_cache = StreamingLLMKVCache(total_budget=total_budget, start_size=start_size)
            module.layer_idx = int(name.split('.')[-2])
            module.check_recall = args.check_recall
            module.forward = types.MethodType(qwen_forward_recall_patch, module)
            patched_layers += 1

    print(f"StreamingLLM Global Diagnostic Patch Enabled. Patched {patched_layers} layers (Budget={total_budget}, Sink={start_size}).")