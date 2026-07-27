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

class H2OKVCache:
    def __init__(self, recent_size=32, h2o_size=992, k_seq_dim=2, v_seq_dim=2):
        self.recent_size = recent_size
        self.h2o_size = h2o_size
        self.cache_size = recent_size + h2o_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None
        self.select_idx = None  
        self._global_step = 0
        self.absolute_indices = None
        self.total_processed_tokens = 0 
        
        self.shadow_key = None 

    @torch.no_grad()
    def evict_for_space(self, query_states, key_states, value_states, check_recall=False, layer_idx=0):
        bsz, num_heads, q_len, head_dim = query_states.shape
        kv_seq_len = key_states.shape[self.k_seq_dim]
        device = query_states.device

        if self.absolute_indices is None:
            self.absolute_indices = torch.arange(kv_seq_len, device=device).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, -1)
            self.total_processed_tokens = kv_seq_len
        else:
            new_len = kv_seq_len - self.absolute_indices.shape[-1]
            if new_len > 0:
                new_idx = torch.arange(self.total_processed_tokens, self.total_processed_tokens + new_len, device=device).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, -1)
                self.absolute_indices = torch.cat([self.absolute_indices, new_idx], dim=-1)
                self.total_processed_tokens += new_len

        if check_recall:
            if self.shadow_key is None:
                self.shadow_key = key_states.clone()
            else:
                new_key = key_states[:, :, -q_len:, :]
                self.shadow_key = torch.cat([self.shadow_key, new_key], dim=2)

        current_scores = torch.zeros((bsz, num_heads, kv_seq_len), device=device, dtype=torch.bfloat16)
        chunk_size = 256
        for i in range(0, q_len, chunk_size):
            end_idx = min(i + chunk_size, q_len)
            q_chunk = query_states[:, :, i:end_idx, :]
            attn_chunk = torch.matmul(q_chunk, key_states.transpose(-1, -2)) / math.sqrt(head_dim)
            if q_len > 1:
                row_indices = torch.arange(i, end_idx, device=device).view(-1, 1)
                col_indices = torch.arange(kv_seq_len, device=device).view(1, -1)
                mask = row_indices >= (col_indices - (kv_seq_len - q_len))
                attn_chunk.masked_fill_(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            
            attn_chunk = F.softmax(attn_chunk, dim=-1, dtype=torch.bfloat16)
            current_scores += attn_chunk.sum(dim=-2)

        if self.hh_score is None:
            self.hh_score = current_scores
        else:
            if q_len == 1:
                self.hh_score = torch.cat([self.hh_score, current_scores[:, :, -1:]], dim=-1)
            else:
                self.hh_score = current_scores

        if kv_seq_len <= self.cache_size:
            self.select_idx = torch.arange(kv_seq_len, device=device).view(1, 1, -1).expand(bsz, num_heads, -1)
            return key_states, value_states

        history_len = kv_seq_len - self.recent_size
        history_scores = self.hh_score[:, :, :history_len]
        _, keep_topk_idx = torch.topk(history_scores, self.h2o_size, dim=-1)
        keep_topk_idx, _ = torch.sort(keep_topk_idx, dim=-1)
        keep_recent_idx = torch.arange(history_len, kv_seq_len, device=device).view(1, 1, -1).expand(bsz, num_heads, -1)
        
        keep_idx = torch.cat([keep_topk_idx, keep_recent_idx], dim=-1)

        if check_recall:
            global current_sample_stats
            step = self._global_step
            if layer_idx == 0:
                if step >= len(current_sample_stats):
                    current_sample_stats.append({}) 
                    
            q_current = query_states[:, :, -1:, :] 
            
            raw_attn_global = torch.matmul(q_current, self.shadow_key.transpose(-1, -2)) / math.sqrt(head_dim)
            probs_global = F.softmax(raw_attn_global, dim=-1, dtype=torch.float32)
            full_probs_global = probs_global[0, :, 0, :]  
            
            total_uncompressed_len = self.shadow_key.shape[2]
            K = self.cache_size
            actual_k = min(K, total_uncompressed_len)
            actual_100 = min(100, total_uncompressed_len)
            
            _, true_topk_indices = torch.topk(full_probs_global, actual_k, dim=-1)
            _, true_top100_indices = torch.topk(full_probs_global, actual_100, dim=-1)
            
            h2o_global_indices = torch.gather(self.absolute_indices, -1, keep_idx)
            current_select_global = h2o_global_indices[0]  
            
            metrics_matrix = np.zeros((num_heads, 3), dtype=np.float32)
            
            for h in range(num_heads):

                true_topk_set = set(true_topk_indices[h].tolist())
                true_top100_set = set(true_top100_indices[h].tolist())

                filtered_set = set(current_select_global[h].tolist())

                selected_attn_ratio = full_probs_global[h, current_select_global[h]].sum().item()
                recall_100 = len(filtered_set.intersection(true_top100_set)) / 100.0
                recall_k = len(filtered_set.intersection(true_topk_set)) / float(K) if K > 0 else 0.0
                
                metrics_matrix[h] = [recall_100, recall_k, selected_attn_ratio]
                
            current_sample_stats[step][layer_idx] = metrics_matrix

        self.select_idx = keep_idx
        self.hh_score = torch.gather(self.hh_score, -1, keep_idx)
        self.absolute_indices = torch.gather(self.absolute_indices, -1, keep_idx)
        
        gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_comp = torch.gather(key_states, self.k_seq_dim, gather_idx)
        v_comp = torch.gather(value_states, self.v_seq_dim, gather_idx)

        return k_comp.to(query_states.dtype), v_comp.to(query_states.dtype)


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

def common_diagnostic_and_evict(self, query_states, key_states, value_states, past_key_value, bsz, q_len, sin, cos):
    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, {"sin": sin, "cos": cos})
        check_recall = getattr(self, "check_recall", False)
        k_comp, v_comp = self.kv_cache.evict_for_space(query_states, key_states, value_states, check_recall, self.layer_idx)
        
        past_key_value.key_cache[self.layer_idx] = k_comp
        past_key_value.value_cache[self.layer_idx] = v_comp
        
        if self.layer_idx == 0:
            self.kv_cache._global_step += 1
    else:
        k_comp, v_comp = key_states, value_states

    attn_out = flash_attn_func(
        query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), causal=True
    ).reshape(bsz, q_len, self.hidden_size)

    return self.o_proj(attn_out), None, past_key_value

def enable_h2o_recall(model, args):
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
        if llama_classes and isinstance(module, llama_classes):
            module.kv_cache = H2OKVCache(recent_size=args.recent_size, h2o_size=args.heavy_hitter_size)
            module.layer_idx = int(name.split('.')[-2])
            module.check_recall = args.check_recall
            module.forward = types.MethodType(llama_forward_recall_patch, module)
            patched_layers += 1
        elif qwen_classes and isinstance(module, qwen_classes):
            module.kv_cache = H2OKVCache(recent_size=args.recent_size, h2o_size=args.heavy_hitter_size)
            module.layer_idx = int(name.split('.')[-2])
            module.check_recall = args.check_recall
            module.forward = types.MethodType(qwen_forward_recall_patch, module)
            patched_layers += 1

    print(f"H2O Global Diagnostic Patch Enabled. Patched {patched_layers} attention layers.")