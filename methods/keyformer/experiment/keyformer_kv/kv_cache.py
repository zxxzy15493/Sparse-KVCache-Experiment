import math
import torch
import torch.nn.functional as FS


class KeyformerKVCache:
    def __init__(self, key_size=992, recent_size=32,  
                 tau_init=1.0, tau_delta=0.01, k_seq_dim=2, v_seq_dim=2):
        self.key_size = key_size      
        self.recent_size = recent_size  
        self.cache_size = key_size + recent_size
        
        self.tau_init = tau_init
        self.tau_delta = tau_delta
        
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.key_score = None 
        
    @torch.no_grad()
    def evict_for_space(self, query_states, key_states, value_states, itr_count):
        bsz, num_heads, q_len, head_dim = query_states.shape
        kv_seq_len = key_states.shape[self.k_seq_dim]
        device = query_states.device
        dtype = query_states.dtype

        current_tau = self.tau_init + (itr_count * self.tau_delta)
        current_scores = torch.zeros((bsz, num_heads, kv_seq_len), device=device, dtype=torch.bfloat16)
        chunk_size = 128

        for i in range(0, q_len, chunk_size):
            end_idx = min(i + chunk_size, q_len)
            query_chunk = query_states[:, :, i:end_idx, :]
            
            attn_weights_chunk = torch.matmul(query_chunk, key_states.transpose(-1, -2)) / math.sqrt(head_dim)
            
            if q_len > 1:
                row_indices = torch.arange(i, end_idx, device=device).view(-1, 1)
                col_indices = torch.arange(kv_seq_len, device=device).view(1, -1)
                mask = row_indices >= (col_indices - (kv_seq_len - q_len))
                attn_weights_chunk.masked_fill_(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            gumbel_weight = torch.nn.functional.gumbel_softmax(
                attn_weights_chunk, tau=current_tau, hard=False, dim=-1
            ).to(torch.bfloat16)
            
            current_scores += gumbel_weight.sum(dim=-2)
            
            del attn_weights_chunk, gumbel_weight
        if q_len > 4096:
            torch.cuda.empty_cache()

        if self.key_score is None:
            self.key_score = current_scores
        else:
            if q_len == 1: 
                self.key_score = torch.cat([self.key_score, current_scores[:, :, -1:]], dim=-1)
            else: 
                self.key_score = current_scores

        if kv_seq_len <= self.cache_size:
            return key_states, value_states
 
        history_boundary = kv_seq_len - self.recent_size
        keep_recent_idx = torch.arange(history_boundary, kv_seq_len, device=device).view(1, 1, -1).expand(bsz, num_heads, -1)
        
        history_scores = self.key_score[:, :, : history_boundary]
        _, topk_idx = torch.topk(history_scores, self.key_size, dim=-1)
        topk_idx, _ = torch.sort(topk_idx, dim=-1)

        keep_idx = torch.cat([topk_idx, keep_recent_idx], dim=-1)
        keep_idx, _ = torch.sort(keep_idx, dim=-1) 

        gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        
        key_compressed = torch.gather(key_states, self.k_seq_dim, gather_idx)
        value_compressed = torch.gather(value_states, self.v_seq_dim, gather_idx)
        
        self.key_score = torch.gather(self.key_score, -1, keep_idx)
        self.select_idx = keep_idx

        return key_compressed.to(dtype), value_compressed.to(dtype)