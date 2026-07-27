import math
import torch
import torch.nn.functional as F

class H2OKVCache:
    def __init__(self,recent_size=32, h2o_size=992,k_seq_dim=2, v_seq_dim=2):
        self.recent_size=recent_size
        self.h2o_size=h2o_size
        self.cache_size = recent_size + h2o_size
        self.k_seq_dim=k_seq_dim
        self.v_seq_dim=v_seq_dim
        self.hh_score = None
        
    @torch.no_grad()
    def evict_for_space(self, query_states, key_states, value_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        kv_seq_len = key_states.shape[self.k_seq_dim]
        device = query_states.device
        current_scores = torch.zeros((bsz, num_heads, kv_seq_len), device=device, dtype=torch.bfloat16)
        chunk_size = 256

        for i in range(0, q_len, chunk_size):
            end_idx = min(i + chunk_size, q_len)
            query_chunk = query_states[:, :, i:end_idx, :]
            
            attn_weights_chunk = torch.matmul(query_chunk, key_states.transpose(-1, -2)) / math.sqrt(head_dim)
            
            if q_len > 1:
                row_indices = torch.arange(i, end_idx, device=device).view(-1, 1)
                col_indices = torch.arange(kv_seq_len, device=device).view(1, -1)
                mask = row_indices >= (col_indices - (kv_seq_len - q_len))
                attn_weights_chunk.masked_fill_(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            attn_weights_chunk = torch.nn.functional.softmax(attn_weights_chunk, dim=-1, dtype=torch.bfloat16)
            current_scores += attn_weights_chunk.sum(dim=-2)
            del attn_weights_chunk

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

        keep_recent_idx = torch.arange(
            history_len, kv_seq_len, device=device
        ).view(1, 1, -1).expand(bsz, num_heads, -1)

        keep_idx = torch.cat([keep_topk_idx, keep_recent_idx], dim=-1)

        gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        key_states_compress = torch.gather(key_states, self.k_seq_dim, gather_idx)
        value_states_compress = torch.gather(value_states, self.v_seq_dim, gather_idx)
        
        self.hh_score = torch.gather(self.hh_score, -1, keep_idx)
        self.select_idx = keep_idx

        return key_states_compress.to(query_states.dtype), value_states_compress.to(query_states.dtype)