import torch
import torch.nn as nn
from kmeans_gpu import KMeans
from typing import Optional, List, Tuple
import numpy as np
import os

# Controls whether high_score_cnt is saved to file
TOPP_SAVE_TOPK = os.environ.get("TOPP_SAVE_TOPK")
TOPK_SAVE_TOPP = os.environ.get("TOPK_SAVE_TOPP")

### modify ###
# === RECALL test instrumentation (mirrors pq_search.py / retrieval_based_compressor.py) ===
# Gate that enables per-(layer, head) recall CSV logging for fullKVLimitBasedCompressor.
# Same file format as PqBasedSearchCompressor: recall_list/<RECALL_NAME>/<RECALL_NAME>_<TIMESTAMP>.csv
# Header: layer,head,recall,recall@100,selected_attn
from vq_method.retrieval_based.retrieval_based_compressor import *
CHECK_RECALL = eval(os.environ.get("CHECK_RECALL", "0"))
### modify ###

def repeat(a, size, dim_idx):
    shape = a.shape
    return a.unsqueeze(dim_idx+1) \
            .expand(*shape[:dim_idx], shape[dim_idx], size, *shape[dim_idx+1:]) \
            .reshape(*shape[:dim_idx], shape[dim_idx] * size, *shape[dim_idx+1:])

def unrepeat(a, size, dim_idx):
    shape = a.shape
    return a.reshape(*shape[:dim_idx], shape[dim_idx] // size, size, *shape[dim_idx+1:]) \
            .select(dim_idx+1, 0) \
            .squeeze(dim_idx+1)

class KVCacheH2O(object):
    def __init__(self, compress_ratio, h2o_ratio, recent_ratio, drop_ratio, sink_size = 0, show_hit_rate = True):
        print("Initing KV Cache H2O compressor")
        self.compress_ratio = compress_ratio
        self.h2o_ratio = h2o_ratio
        self.recent_ratio = recent_ratio
        self.sink_size = sink_size
        self.drop_ratio = drop_ratio 
        self._current_nclus = 0
        self._current_indices = None
        self.show_hit_rate = show_hit_rate

        self.previous_scores = None
        self.attention_masks_next = None

    def apply(
        self,
        past_key_value: torch.Tensor,
        attention_score: Optional[torch.Tensor] = None,
        query_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        key_states, value_states = past_key_value
        if self.show_hit_rate:
            self.original_kv = (key_states, value_states)
        bsz, kv_heads, kv_seq_len, dim = key_states.shape

        self.cache_budget = int(self.compress_ratio * kv_seq_len)
        self.recent_budget = int(self.recent_ratio * (self.cache_budget - self.sink_size))
        self.heavy_budget = self.cache_budget - self.sink_size - self.recent_budget

        recent_index = kv_seq_len - self.recent_budget
        to_compress_token_cnt = kv_seq_len - self.recent_budget - self.sink_size
        
        key_states, value_states = key_states.view(-1, kv_seq_len, dim), value_states.view(-1, kv_seq_len, dim)
        recent_keys, recent_values = key_states[:, recent_index:, :], value_states[:, recent_index:, :]

        if self.sink_size > 0:
            sink_keys,sink_values = key_states[:, :self.sink_size, :], value_states[:, :self.sink_size, :]

        non_recent_sink_keys = key_states[:, self.sink_size:recent_index, :]
        non_recent_sink_values = value_states[:, self.sink_size:recent_index, :]

        generalized_bsz = bsz * kv_heads
        assert attention_score.shape == (bsz, kv_heads, kv_seq_len)
        attention_score = attention_score[:, :, self.sink_size:recent_index].view(generalized_bsz, to_compress_token_cnt)

        if self.heavy_budget > 0:
            top_k_indices = torch.topk(attention_score, self.heavy_budget, dim=-1, largest=True).indices
            top_k_keys = torch.gather(
                                non_recent_sink_keys, dim = 1, 
                                index = top_k_indices.unsqueeze(2).expand([generalized_bsz, self.heavy_budget, dim]))
            top_k_values = torch.gather(
                                non_recent_sink_values, dim = 1, 
                                index = top_k_indices.unsqueeze(2).expand([generalized_bsz, self.heavy_budget, dim]))

        new_key_states = torch.cat([sink_keys, top_k_keys, recent_keys], dim = 1)
        new_value_states = torch.cat([sink_values, top_k_values, recent_values], dim = 1)

        new_key_states = new_key_states.view(bsz, kv_heads, self.cache_budget, dim)
        new_value_states = new_value_states.view(bsz, kv_heads, self.cache_budget, dim)

        self.kv_cache_cnt = np.ones([bsz*kv_heads], dtype=np.int64) * self.cache_budget
        return new_key_states, new_value_states, self.kv_cache_cnt

    def restore(
        self,
        attn_weights: torch.Tensor,
        num_key_value_groups: int,
    ):
        return nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)

class KVCacheH2OOfficial:
    def __init__(self, compress_ratio, h2o_ratio, recent_ratio, sink_size = 0):
        print("Initing KV Cache H2O compressor: official code base version")
        self.compress_ratio = compress_ratio
        self.h2o_ratio = h2o_ratio
        self.recent_ratio = recent_ratio
        self.sink_size = sink_size

        self.previous_scores = None
        self.attention_masks_next = None
    
    def _reset_masks(self):
        self.attention_masks_next = None 
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None

    def apply(
        self,
        past_key_value: torch.Tensor,
        attention_score: Optional[torch.Tensor] = None,
        query_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        self._reset_masks()
        key_states, value_states = past_key_value
        bsz, kv_heads, kv_seq_len, dim = key_states.shape

        current_scores_sum = attention_score.sum(0)
        assert current_scores_sum.shape == (kv_heads, kv_seq_len), f"{current_scores_sum.shape}"

        self.heavy_budget = int(self.compress_ratio * self.h2o_ratio * (kv_seq_len - self.sink_size))
        self.recent_budget = int(self.compress_ratio * self.recent_ratio * (kv_seq_len - self.sink_size))
        self.cache_budget = self.heavy_budget + self.recent_budget + self.sink_size

        dtype_attn_weights = attention_score.dtype
        attn_weights_devices = attention_score.device
        assert attention_score.shape[0] == 1
        self.previous_scores = current_scores_sum
        attn_mask = torch.ones(current_scores_sum.shape[0], current_scores_sum.shape[1]+1).to(dtype_attn_weights).to(attn_weights_devices)

        attn_tokens_all = self.previous_scores.shape[-1]
    
        if attn_tokens_all > self.cache_budget:
            if not self.recent_budget == 0:
                attn_mask[:, self.sink_size:-self.recent_budget] = 0
                selected_set = self.previous_scores[:, self.sink_size:-self.recent_budget]
            else:
                selected_set = self.previous_scores

            if not self.heavy_budget == 0:
                _, keep_topk = selected_set.topk(k=self.heavy_budget, dim=-1, largest=True)
                attn_mask[:, self.sink_size:-self.recent_budget].scatter_(-1, keep_topk, 1)

        self.attention_masks_next = attn_mask.unsqueeze(0).unsqueeze(2)

        score_mask = attn_mask[:,:-1]
        score_mask[:, -self.recent_budget:] = 1 
        score_mask[:, :self.sink_size] = 1
        self.previous_scores = self.previous_scores * score_mask

        self.kv_cache_cnt = np.ones([bsz*kv_heads], dtype=np.int64) * self.cache_budget
        return key_states, value_states, self.kv_cache_cnt

    
    def restore(
        self,
        attn_weights: torch.Tensor,
        num_key_value_groups
    ):
        n_kv_heads = self.attention_masks_next.shape[1]
        if self.attention_masks_next is not None:
            assert torch.sum(self.attention_masks_next) == n_kv_heads * (self.cache_budget + 1), f"{torch.sum(self.attention_masks_next)}, {n_kv_heads}, {self.cache_budget}"
            repeat_attention_masks_next = repeat(self.attention_masks_next, num_key_value_groups, dim_idx=1)
            attn_weights = attn_weights * repeat_attention_masks_next + (1 - repeat_attention_masks_next) * torch.finfo(attn_weights.dtype).min

        original_dtype = attn_weights.dtype
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(original_dtype)

        current_scores_sum = attn_weights.unflatten(1, (n_kv_heads, num_key_value_groups)) \
                                        .sum(0).sum(1).sum(1)

        current_scores_sum[:, :-1] += self.previous_scores

        dtype_attn_weights = attn_weights.dtype
        attn_weights_devices = attn_weights.device
        assert attn_weights.shape[0] == 1
        self.previous_scores = current_scores_sum
        attn_mask = torch.ones(current_scores_sum.shape[0], current_scores_sum.shape[1]+1).to(dtype_attn_weights).to(attn_weights_devices)

        attn_tokens_all = self.previous_scores.shape[-1]
    
        if attn_tokens_all > self.cache_budget:
            if not self.recent_budget == 0:
                attn_mask[:, self.sink_size:-self.recent_budget] = 0
                selected_set = self.previous_scores[:, self.sink_size:-self.recent_budget]
            else:
                selected_set = self.previous_scores

            if not self.heavy_budget == 0:
                _, keep_topk = selected_set.topk(k=self.heavy_budget, dim=-1, largest=True)
                attn_mask[:, self.sink_size:-self.recent_budget].scatter_(-1, keep_topk, 1)

        self.attention_masks_next = attn_mask.unsqueeze(0).unsqueeze(2)

        score_mask = attn_mask[:,:-1]
        score_mask[:, -self.recent_budget:] = 1 
        score_mask[:, :self.sink_size] = 1
        self.previous_scores = self.previous_scores * score_mask

        return attn_weights


recall_history = []

# def save_topk_topp_sum(x):
#             if TOPK_SAVE_TOPP is not None:
#                 os.makedirs("record", exist_ok=True)
#                 with open(f"record/{TOPK_SAVE_TOPP}.txt", "a") as f:
#                     f.write(f"{[round(v, 3) for v in x.tolist()]}\n")
def save_topk_topp_sum(x):
    if TOPK_SAVE_TOPP is None:
        return

    os.makedirs("record", exist_ok=True)

    x = x.detach().float().cpu().reshape(-1)
    line = " ".join(f"{v:.3f}" for v in x.tolist())

    with open(f"record/{TOPK_SAVE_TOPP}.txt", "a") as f:
        f.write(line + "\n")
class fullKVLimitBasedCompressor(object):
    def __init__(self, compress_ratio, h2o_ratio, recent_ratio, gqa, sink_size = 0, show_hit_rate=True, fixbudget = True, budget = 0, recent_size = 32):
        self.compress_ratio = compress_ratio
        self.h2o_ratio = h2o_ratio
        self.recent_ratio = recent_ratio
        self.sink_size = sink_size
        self.fixbudget = fixbudget
        self.budget = budget
        self.recent_size = recent_size
        assert (self.h2o_ratio + self.recent_ratio) > 0.99
        self.prefill_length = 0
        self.GQA = gqa
        print(f"Initializing oracle compressor, GQA mode:{self.GQA}")
        self.fwd_cnt = 0

    def h2o_recall(self, new_indices, num_key_value_group):
        bsz, kv_head, topk = self.non_recent_sink_topk_indices.shape
        if (new_indices.shape[1] // self.non_recent_sink_topk_indices.shape[1]) == num_key_value_group:
            repeat_idx = self.non_recent_sink_topk_indices.unsqueeze(2) \
                                                    .expand(bsz, kv_head, num_key_value_group, topk) \
                                                    .reshape(-1, topk)
        else:
            repeat_idx = self.non_recent_sink_topk_indices
        repeat_idx = repeat_idx.reshape(-1, topk)
        new_indices = new_indices.reshape(-1, topk)
        all_topk_cnt = new_indices.numel()
        intersect = 0
        for i in range(new_indices.shape[0]):
            a = set(repeat_idx[0].tolist())
            b = set(new_indices[i].tolist())
            intersect += len(a.intersection(b))
        result = intersect / all_topk_cnt
        recall_history.append(result)
        np_history = np.array(recall_history)
        mean, var = np.mean(np_history), np.var(np_history)
        print(f"H2o selected token hit rate:{result}, mean:{mean}, var:{var}")
        
    def apply(
        self,
        past_key_value: torch.Tensor,
        layer_idx: int = -1,
        attention_score: Optional[torch.Tensor] = None,
        query_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        bsz, kv_heads, kv_seq_len, _ = past_key_value[0].shape
        self.prefill_length = kv_seq_len

        # Calculate budget based on mode
        if self.fixbudget:
            self.budget = min(self.budget, self.prefill_length)
            # Use fixed budget mode
            if self.budget < self.sink_size + self.recent_size:
                raise ValueError(f"fixbudget is True but budget {self.budget} is less than sink_size + recent_size ({self.sink_size} + {self.recent_size})")
            # Clamp budget to prefill_length
            # actual_budget = min(self.budget, self.prefill_length)
            self.local_budget = self.recent_size
            self.high_score_budget = self.budget - self.sink_size - self.recent_size
        else:
            # Use ratio-based calculation (original behavior)
            self.budget = int((self.prefill_length - self.sink_size) * self.compress_ratio)
            self.high_score_budget = int(self.budget * self.h2o_ratio)
            self.local_budget = self.budget - self.high_score_budget

        self.recent_index = int(kv_seq_len - self.local_budget)

        # self.non_recent_sink_topk_indices = attention_score[...,self.sink_size:self.recent_index] \
        #                                                 .topk(self.high_score_budget, dim=-1).indices
        if CHECK_RECALL and layer_idx == 0:
            write_recall_prompt_separator(prefill_len=kv_seq_len)
        return past_key_value[0], past_key_value[1], np.ones([bsz*kv_heads], dtype=np.int64) * kv_seq_len

    def restore(
        self,
        attn_weights: torch.Tensor,
        num_key_value_groups: int,
        ### modify ###
        # RECALL instrumentation hooks (CHECK_RECALL=1). The patches pass the
        # un-repeated K (per-kv-head) and Q at every call so we can compare the
        # compressor's topk against the true topk of the full K cache. Defaults
        # preserve the original 2-arg signature for production callers.
        query_states: Optional[torch.Tensor] = None,
        key_states: Optional[torch.Tensor] = None,
        layer_idx: int = -1,
        ### modify ###
    ):
        attn_weights = attn_weights.clone()
        bsz, nheads, q_len, kv_len = attn_weights.shape
        self.recent_index = int(kv_len - self.local_budget)
        non_recent_sink_cnt = self.recent_index - self.sink_size
        kv_heads = nheads//num_key_value_groups
        attn_scores = torch.softmax(attn_weights, dim = -1)

        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        raw_attn_scores = attn_scores
        # print(self.high_score_budget)
        if self.GQA:
            attn_scores = attn_scores[..., self.sink_size:self.recent_index]
            attn_scores = attn_scores.reshape([bsz, kv_heads, num_key_value_groups, q_len, non_recent_sink_cnt]) \
                                        .sum(dim=2)
            high_score_indice = attn_scores.topk(self.high_score_budget, dim=-1, largest=True).indices
            assert high_score_indice.shape == (bsz, kv_heads, q_len, self.high_score_budget)
        else:
            raise Exception("...")

        ### modify ###
        # === RECALL test: topk vs. true topk of the full K cache ===
        # topk (no_drop_lb) defaults to sink=0/recent=0, so high_score_indice
        # already indexes into the full K cache [sink_size, kv_len) and the
        # recall set is just high_score_indice shifted by sink_size. No cat,
        # no recent window, no new-token slot — kept inside calc_recall's
        # uniqueness invariant by construction.
        if CHECK_RECALL and query_states is not None and key_states is not None:
            calc_recall(
                query_states,
                key_states,
                high_score_indice + self.sink_size,
                num_key_value_groups,
                self.high_score_budget,
                layer_idx=layer_idx,
            )
        ### modify ###

        neg_inf_mask = attn_scores.new_zeros([bsz, high_score_indice.shape[1], q_len, non_recent_sink_cnt]) + float('-inf')
        neg_inf_mask.scatter_(-1, high_score_indice, 0)
        
        if self.GQA:
            repeat_neg_inf_mask = repeat(neg_inf_mask, size = num_key_value_groups, dim_idx = 1)
        else:
            raise Exception("...")
        
        if TOPK_SAVE_TOPP is not None:
            selected_mask = torch.ones_like(raw_attn_scores, dtype=torch.bool)
            selected_mask[..., self.sink_size:self.recent_index] = torch.logical_not(torch.isinf(repeat_neg_inf_mask))
            selected_score_sum = (raw_attn_scores * selected_mask.to(raw_attn_scores.dtype)).sum(dim=-1)
            # print('haaa')
            save_topk_topp_sum(selected_score_sum)
        # if np.random.randint(0,10000) % 1000 == 1:
        #     self.h2o_recall(high_score_indice, num_key_value_groups)

        attn_weights[...,self.sink_size:self.recent_index] += repeat_neg_inf_mask

        assert torch.sum(torch.logical_not(torch.isinf(attn_weights))) == nheads * (self.budget)

        new_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        return new_weights

class fullKVLimitBasedCompressor32(object):
    def __init__(self, compress_ratio, h2o_ratio, recent_ratio, gqa, sink_size = 0, show_hit_rate=True, fixbudget = True, budget = 0, recent_size = 32):
        self.compress_ratio = compress_ratio
        self.h2o_ratio = h2o_ratio
        self.recent_ratio = recent_ratio
        self.sink_size = sink_size
        self.fixbudget = fixbudget
        self.budget = budget
        self.recent_size = recent_size
        assert (self.h2o_ratio + self.recent_ratio) > 0.99
        self.prefill_length = 0
        self.GQA = gqa
        print(f"Initializing oracle compressor, GQA mode:{self.GQA}")
        self.fwd_cnt = 0

    def h2o_recall(self, new_indices, num_key_value_group):
        bsz, kv_head, topk = self.non_recent_sink_topk_indices.shape
        if (new_indices.shape[1] // self.non_recent_sink_topk_indices.shape[1]) == num_key_value_group:
            repeat_idx = self.non_recent_sink_topk_indices.unsqueeze(2) \
                                                    .expand(bsz, kv_head, num_key_value_group, topk) \
                                                    .reshape(-1, topk)
        else:
            repeat_idx = self.non_recent_sink_topk_indices
        repeat_idx = repeat_idx.reshape(-1, topk)
        new_indices = new_indices.reshape(-1, topk)
        all_topk_cnt = new_indices.numel()
        intersect = 0
        for i in range(new_indices.shape[0]):
            a = set(repeat_idx[0].tolist())
            b = set(new_indices[i].tolist())
            intersect += len(a.intersection(b))
        result = intersect / all_topk_cnt
        recall_history.append(result)
        np_history = np.array(recall_history)
        mean, var = np.mean(np_history), np.var(np_history)
        print(f"H2o selected token hit rate:{result}, mean:{mean}, var:{var}")

    def apply(
        self,
        past_key_value: torch.Tensor,
        layer_idx: int = -1,
        attention_score: Optional[torch.Tensor] = None,
        query_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        bsz, kv_heads, kv_seq_len, _ = past_key_value[0].shape
        self.prefill_length = kv_seq_len

        # Calculate budget based on mode
        if self.fixbudget:
            self.budget = min(self.budget, self.prefill_length)
            # Use fixed budget mode
            if self.budget < self.sink_size + self.recent_size:
                raise ValueError(f"fixbudget is True but budget {self.budget} is less than sink_size + recent_size ({self.sink_size} + {self.recent_size})")
            # Clamp budget to prefill_length
            # actual_budget = min(self.budget, self.prefill_length)
            self.local_budget = self.recent_size
            self.high_score_budget = self.budget - self.sink_size - self.recent_size
        else:
            # Use ratio-based calculation (original behavior)
            self.budget = int((self.prefill_length - self.sink_size) * self.compress_ratio)
            self.high_score_budget = int(self.budget * self.h2o_ratio)
            self.local_budget = self.budget - self.high_score_budget

        self.recent_index = int(kv_seq_len - self.local_budget)

        # self.non_recent_sink_topk_indices = attention_score[...,self.sink_size:self.recent_index] \
        #                                                 .topk(self.high_score_budget, dim=-1).indices

        if CHECK_RECALL and layer_idx == 0:
            write_recall_prompt_separator(prefill_len=kv_seq_len)
        return past_key_value[0], past_key_value[1], np.ones([bsz*kv_heads], dtype=np.int64) * kv_seq_len

    def restore(
        self,
        attn_weights: torch.Tensor,
        num_key_value_groups: int,
        query_states: Optional[torch.Tensor] = None,
        key_states: Optional[torch.Tensor] = None,
        layer_idx: int = -1,
    ):
        attn_weights = attn_weights.clone()
        bsz, nheads, q_len, kv_len = attn_weights.shape
        self.recent_index = int(kv_len - self.local_budget)
        non_recent_sink_cnt = self.recent_index - self.sink_size

        # compute softmax
        attn_scores = torch.softmax(attn_weights, dim=-1)
        raw_attn_scores = attn_scores
        if self.GQA:
            # take only the non_recent_sink part, no more reshape and sum
            # each qhead computes its own topk independently
            attn_scores = attn_scores[..., self.sink_size:self.recent_index]
            high_score_indice = attn_scores.topk(self.high_score_budget, dim=-1, largest=True).indices
            # high_score_indice shape: (bsz, nheads, q_len, high_score_budget)
            assert high_score_indice.shape == (bsz, nheads, q_len, self.high_score_budget)
        else:
            raise Exception("...")

        # Unlike fullKVLimitBasedCompressor, these indices are already selected
        # independently per query head. calc_recall accepts either q-head or
        # kv-head indices and only repeats the latter, so pass them through.
        if CHECK_RECALL and query_states is not None and key_states is not None:
            calc_recall(
                query_states,
                key_states,
                high_score_indice + self.sink_size,
                num_key_value_groups,
                self.high_score_budget,
                layer_idx=layer_idx,
            )
        
        # create mask: each qhead generates its own mask, no repeat needed
        neg_inf_mask = attn_scores.new_zeros([bsz, nheads, q_len, non_recent_sink_cnt]) + float('-inf')
        neg_inf_mask.scatter_(-1, high_score_indice, 0)

        # repeat operation no longer needed
        # if self.GQA:
        #     repeat_neg_inf_mask = repeat(neg_inf_mask, size = num_key_value_groups, dim_idx = 1)

        if TOPK_SAVE_TOPP is not None:
            selected_mask = torch.ones_like(raw_attn_scores, dtype=torch.bool)
            selected_mask[..., self.sink_size:self.recent_index] = torch.logical_not(torch.isinf(neg_inf_mask))
            selected_score_sum = (raw_attn_scores * selected_mask.to(raw_attn_scores.dtype)).sum(dim=-1)
            save_topk_topp_sum(selected_score_sum)
            
        attn_weights[..., self.sink_size:self.recent_index] += neg_inf_mask

        assert torch.sum(torch.logical_not(torch.isinf(attn_weights))) == nheads * (self.budget)

        new_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        return new_weights
    
class fullKVLimitBasedCompressorTOPP(object):
    """TOPP mode: sum within GQA group, threshold at threshold * num_key_value_groups, then repeat"""
    def __init__(self, fixthreshold, gqa):
        assert fixthreshold > 0 and fixthreshold <= 1
        self.threshold = fixthreshold
        self.GQA = gqa
        print(f"Initializing fullKVLimitBasedCompressorTOPP, GQA mode:{self.GQA}")
        self.fwd_cnt = 0

    def apply(
        self,
        past_key_value: torch.Tensor,
        layer_idx: int = -1,
        attention_score: Optional[torch.Tensor] = None,
        query_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        bsz, kv_heads, kv_seq_len, _ = past_key_value[0].shape
        self.prefill_length = kv_seq_len
        return past_key_value[0], past_key_value[1], np.ones([bsz*kv_heads], dtype=np.int64) * kv_seq_len

    def restore(
        self,
        attn_weights: torch.Tensor,
        num_key_value_groups: int,
        query_states: Optional[torch.Tensor] = None,
        key_states: Optional[torch.Tensor] = None,
        layer_idx: int = -1,
    ):
        attn_weights = attn_weights.clone()
        bsz, nheads, q_len, kv_len = attn_weights.shape
        kv_heads = nheads // num_key_value_groups

        # compute softmax
        attn_scores = torch.softmax(attn_weights, dim=-1)

        # GQA mode: sum qheads within the same group
        # attn_scores: [bsz, nheads, q_len, kv_len] -> [bsz, kv_heads, num_key_value_groups, q_len, kv_len] -> sum
        attn_scores = attn_scores.reshape(bsz, kv_heads, num_key_value_groups, q_len, kv_len).sum(dim=2)

        # sort and compute cumulative sum
        sorted_scores, sorted_indices = attn_scores.sort(dim=-1, descending=True)
        cumsum = sorted_scores.cumsum(dim=-1)

        # use searchsorted to find the first position >= threshold * num_key_value_groups (no loop)
        threshold_adjusted = self.threshold * num_key_value_groups
        threshold_tensor = attn_scores.new_full((bsz, kv_heads, q_len, 1), threshold_adjusted)
        high_score_cnt = torch.searchsorted(cumsum, threshold_tensor, side='left').squeeze(-1) + 1

        
        # clamp count to at most kv_len and at least 1
        high_score_cnt = high_score_cnt.clamp(min=1, max=kv_len)

        if TOPP_SAVE_TOPK is not None:
                tmp = TOPP_SAVE_TOPK 
                with open(f"record/{tmp}.txt", "a") as f:
                    f.write(f"{high_score_cnt.tolist()}\n")

        range_tensor = torch.arange(kv_len, device=attn_weights.device)
        mask_sorted = range_tensor.view(1,1,1,-1) < high_score_cnt.unsqueeze(-1)

        # scatter back to original order
        mask = torch.zeros_like(mask_sorted).scatter(-1, sorted_indices, mask_sorted)

        # repeat to all heads
        mask = mask.repeat_interleave(num_key_value_groups, dim=1)

        # apply mask
        attn_weights = attn_weights.masked_fill(~mask, float('-inf'))

        new_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        return new_weights


class fullKVLimitBasedCompressorTOPP32(object):
    """TOPP32 mode: each qhead selects independently, threshold at threshold, no repeat"""
    def __init__(self, fixthreshold, gqa):
        assert fixthreshold > 0 and fixthreshold <= 1
        self.threshold = fixthreshold
        self.GQA = gqa
        print(f"Initializing fullKVLimitBasedCompressorTOPP32, GQA mode:{self.GQA}")
        self.fwd_cnt = 0

    def apply(
        self,
        past_key_value: torch.Tensor,
        layer_idx: int = -1,
        attention_score: Optional[torch.Tensor] = None,
        query_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        bsz, kv_heads, kv_seq_len, _ = past_key_value[0].shape
        self.prefill_length = kv_seq_len
        return past_key_value[0], past_key_value[1], np.ones([bsz*kv_heads], dtype=np.int64) * kv_seq_len

    def restore(
        self,
        attn_weights: torch.Tensor,
        num_key_value_groups: int,
        query_states: Optional[torch.Tensor] = None,
        key_states: Optional[torch.Tensor] = None,
        layer_idx: int = -1,
    ):
        attn_weights = attn_weights.clone()
        bsz, nheads, q_len, kv_len = attn_weights.shape

        # compute softmax
        attn_scores = torch.softmax(attn_weights, dim=-1)

        # sort and compute cumulative sum
        sorted_scores, sorted_indices = attn_scores.sort(dim=-1, descending=True)
        cumsum = sorted_scores.cumsum(dim=-1)

        # use searchsorted to find the first position >= threshold (no loop)
        threshold_tensor = attn_scores.new_full((bsz, nheads, q_len, 1), self.threshold)
        high_score_cnt = torch.searchsorted(cumsum, threshold_tensor, side='left').squeeze(-1) + 1
        
        # clamp count to at most kv_len and at least 1
        high_score_cnt = high_score_cnt.clamp(min=1, max=kv_len)

        if TOPP_SAVE_TOPK is not None:
                tmp = TOPP_SAVE_TOPK 
                with open(f"record/{tmp}.txt", "a") as f:
                    f.write(f"{high_score_cnt.tolist()}\n")
                    
        range_tensor = torch.arange(kv_len, device=attn_weights.device)
        mask_sorted = range_tensor.view(1,1,1,-1) < high_score_cnt.unsqueeze(-1)

        # scatter back to original order
        mask = torch.zeros_like(mask_sorted).scatter(-1, sorted_indices, mask_sorted)


        # apply mask
        attn_weights = attn_weights.masked_fill(~mask, float('-inf'))

        new_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        return new_weights
