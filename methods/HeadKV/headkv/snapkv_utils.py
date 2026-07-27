import warnings
import os

import torch
import time
import numpy as np
import json
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union, Any,Dict
from transformers.cache_utils import Cache, DynamicCache


def _update_flatten_view_fallback(old_flat, new_flat, head_lens):
    if old_flat is None or old_flat.numel() == 0:
        return new_flat

    if head_lens is None:
        return torch.cat([old_flat, new_flat], dim=0)

    head_lens = head_lens.to(device=old_flat.device, dtype=torch.int64).tolist()
    if len(head_lens) == 0:
        return torch.cat([old_flat, new_flat], dim=0)

    if new_flat.shape[0] != len(head_lens):
        return torch.cat([old_flat, new_flat], dim=0)

    pieces = []
    cursor = 0
    for head_idx, head_len in enumerate(head_lens):
        next_cursor = cursor + int(head_len)
        pieces.append(old_flat[cursor:next_cursor])
        pieces.append(new_flat[head_idx:head_idx + 1])
        cursor = next_cursor

    if cursor < old_flat.shape[0]:
        pieces.append(old_flat[cursor:])

    return torch.cat(pieces, dim=0)


class DynamicCacheSplitHeadFlatten(Cache):
    def __init__(self) ->None:
        # Token wise List[]  Head wise KV List[torch.Tensor]
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0

    @property
    def seen_tokens(self):
        return self._seen_tokens

    @seen_tokens.setter
    def seen_tokens(self, value):
        self._seen_tokens = int(value)

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            key_states = self.key_cache[layer_idx]
            value_states = self.value_cache[layer_idx]
            if key_states is None:
                yield (tuple(), tuple())
            else:
                yield (key_states, value_states)

    def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            key_states = self.key_cache[layer_idx]
            value_states = self.value_cache[layer_idx]
            if key_states is None:
                return (tuple(), tuple())
            return (key_states, value_states)
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        cache_kwargs = cache_kwargs or {}
        cache_position = cache_kwargs.get("cache_position")
        # NOTE: avoid repeated GPU->CPU sync on every layer update
        if layer_idx == 0 and torch.is_tensor(cache_position) and cache_position.numel() > 0:
            self._seen_tokens = max(self._seen_tokens, int(cache_position.reshape(-1)[-1].item()) + 1)

        if len(self.key_cache) <= layer_idx:
            missing = layer_idx + 1 - len(self.key_cache)
            self.key_cache.extend([None] * missing)
            self.value_cache.extend([None] * missing)

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        elif key_states.dim() == 2:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:

            assert self.key_cache[layer_idx].dim() == 2
            bs, head, seqlen, dim = key_states.shape
            assert bs == 1 and seqlen == 1
            head_lens = cache_kwargs["head_lens"]
            cu_klen = cache_kwargs["cu_klen"]

            try:
                import nvtx
                copy_old_rng = nvtx.start_range("copy old")
            except Exception:
                copy_old_rng = None

            try:
                from tiny_api_cuda import update_flatten_view
                new_key_cache = update_flatten_view(self.key_cache[layer_idx].view(-1, dim), key_states.view(-1, dim), head_lens, cu_klen)
                new_value_cache = update_flatten_view(self.value_cache[layer_idx].view(-1, dim), value_states.view(-1, dim), head_lens, cu_klen)
            except ModuleNotFoundError:
                new_key_cache = _update_flatten_view_fallback(self.key_cache[layer_idx].view(-1, dim), key_states.view(-1, dim), head_lens)
                new_value_cache = _update_flatten_view_fallback(self.value_cache[layer_idx].view(-1, dim), value_states.view(-1, dim), head_lens)

            if copy_old_rng is not None:
                nvtx.end_range(copy_old_rng)

            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) == 0:
            return 0
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            key_states = self.key_cache[layer_idx]
            value_states = self.value_cache[layer_idx]
            if key_states is None:
                legacy_cache += (([], []),)
            else:
                legacy_cache += ((key_states, value_states),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCacheSplitHeadFlatten":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if isinstance(past_key_values, Cache):
            if hasattr(past_key_values, "to_legacy_cache"):
                past_key_values = past_key_values.to_legacy_cache()
            else:
                return cache
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                if isinstance(key_states, list) and len(key_states) == 0:
                    continue
                cache.update(key_states, value_states, layer_idx)
        return cache


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', layer_idx = None, num_hidden_layers = None, pyram_mode = False, pyram_beta = 20):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

        self.pyram_init = False
        self.pyram_mode = pyram_mode
        self.pyram_beta = pyram_beta
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers


    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # compute pyramidal capacity
        if self.pyram_mode and not self.pyram_init:
            # NOTE: (max_num + min_num) / 2 == base_capacity to restrict the total capacity
            base_capacity = self.max_capacity_prompt - self.window_size
            min_num = base_capacity // self.pyram_beta
            max_num = base_capacity * 2 - min_num
                
            # if the max_num is larger than the query length, we need to adjust the max_num
            if max_num >= q_len - self.window_size:
                max_num = q_len - self.window_size
                min_num = base_capacity * 2 - max_num
        
            # NOTE: compute interval
            steps = (max_num - min_num) // (self.num_hidden_layers - 1)

            self.max_capacity_prompt = max_num - self.layer_idx * steps + self.window_size
            self.pyram_init = True
            # print(f"Pyram mode adaptive capacity, layer: {self.layer_idx}, max_capacity_prompt: {self.max_capacity_prompt}, base_capacity: {self.max_capacity_prompt - self.window_size}")

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            cache_mask = getattr(self, "_window_causal_mask", None)
            if (
                cache_mask is None
                or cache_mask.device != attn_weights.device
                or cache_mask.dtype != attn_weights.dtype
            ):
                cache_mask = torch.full(
                    (self.window_size, self.window_size),
                    torch.finfo(attn_weights.dtype).min,
                    device=attn_weights.device,
                    dtype=attn_weights.dtype,
                )
                mask_cond = torch.arange(self.window_size, device=attn_weights.device)
                cache_mask.masked_fill_(mask_cond < (mask_cond + 1).view(self.window_size, 1), 0)
                self._window_causal_mask = cache_mask
            attention_mask = cache_mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states


class AdaptiveSnapKVCluster():
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool',base_capacity=None,floor = None,skip = None,normalize=None, layer_idx = None, num_hidden_layers = None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = base_capacity - window_size
        self.floor_ratio = floor
        self.floor_capacity = int(self.base_capacity * self.floor_ratio)
        self.adaptive_capacity = self.base_capacity - self.floor_capacity
        self.skip_layer_nums = skip

        self.normalize = normalize
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None


    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        cache_mask = getattr(self, "_window_causal_mask", None)
        if (
            cache_mask is None
            or cache_mask.device != attn_weights.device
            or cache_mask.dtype != attn_weights.dtype
        ):
            cache_mask = torch.full(
                (self.window_size, self.window_size),
                torch.finfo(attn_weights.dtype).min,
                device=attn_weights.device,
                dtype=attn_weights.dtype,
            )
            mask_cond = torch.arange(self.window_size, device=attn_weights.device)
            cache_mask.masked_fill_(mask_cond < (mask_cond + 1).view(self.window_size, 1), 0)
            self._window_causal_mask = cache_mask
        attention_mask = cache_mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)
        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return attn_weights_mean_pooling

    def update_kv(self,  key_states, query_states, value_states):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            if torch.is_tensor(k_lens):
                self.head_lens = k_lens.to(dtype=torch.int32, device=_device)
            else:
                self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.layer_qlens = torch.ones(num_heads, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.cu_offset = torch.arange(0, num_heads + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads+1, dtype=torch.int32, device=_device)

        if q_len <= self.window_size:
            init_metadata(num_heads, [q_len] * num_heads, q_len * num_heads, q_len)
            # all tokens kept — selected indices = all positions
            self._selected_indices = [torch.arange(q_len, dtype=torch.long) for _ in range(num_heads)]
            return key_states.reshape(-1, head_dim), value_states.reshape(-1, head_dim)

        attn_score = self.calcul_attn_sore(key_states,query_states)

        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * num_heads, q_len * num_heads, q_len)
            # not compress — selected indices = all positions
            self._selected_indices = [torch.arange(q_len, dtype=torch.long) for _ in range(num_heads)]
            return key_states.reshape(-1, head_dim), value_states.reshape(-1, head_dim)

        # if you need to weight the attn_score
        pass
        sorted_attn_score,sorted_attn_score_indices = attn_score.sort(dim=-1,descending=True)
        if self.layer_idx >= self.skip_layer_nums:
            adaptive_attn_score = sorted_attn_score
            length = adaptive_attn_score.size(dim=-1)
            if self.normalize:
                ratio_weight = sorted_attn_score[...,:self.base_capacity].sum(dim=-1,keepdim=True)/sorted_attn_score.sum(dim=-1,keepdim=True)
                adaptive_attn_score = adaptive_attn_score*ratio_weight
            adaptive_attn_score = adaptive_attn_score.reshape(bsz,length*num_heads)
            sorted_indices = torch.topk(adaptive_attn_score,k=num_heads*self.base_capacity,dim=-1).indices
            sorted_indices = sorted_indices//length
            # floor capacity set
            head_adaptive_capacity = torch.zeros((bsz,num_heads),device=_device,dtype = sorted_indices.dtype)
            head_adaptive_capacity.scatter_add_(-1,sorted_indices,torch.ones_like(sorted_indices,dtype=head_adaptive_capacity.dtype),)
            if os.getenv("HEADKV_STRICT_ASSERT", "0") == "1":
                if int(head_adaptive_capacity.sum().item()) != num_heads * self.base_capacity:
                    raise AssertionError("head_adaptive_capacity sum mismatch")
            head_adaptive_capacity = torch.round(head_adaptive_capacity * (1-self.floor_ratio) + self.floor_capacity).int()
        else:
            head_adaptive_capacity = torch.ones((bsz,num_heads),device=_device,dtype = sorted_attn_score_indices.dtype) * self.base_capacity

        if not hasattr(self, "_capacity_logged"):
            self._capacity_logged = True
            cap = head_adaptive_capacity[0].to(dtype=torch.int32, device=_device)
            cap_min = int(cap.min().item()) if cap.numel() else 0
            cap_max = int(cap.max().item()) if cap.numel() else 0
            cap_mean = float(cap.float().mean().item()) if cap.numel() else 0.0
            cap_preview = cap[:8].tolist() if cap.numel() else []
               
         
        keep_k_per_head = head_adaptive_capacity[0].to(dtype=torch.int32, device=_device)
        available_k = sorted_attn_score_indices.shape[-1]
        keep_k_per_head = keep_k_per_head.clamp(min=0, max=available_k)
        max_keep = int(keep_k_per_head.max().item())

        tail_k_all = key_states[:, :, -self.window_size:, :]
        tail_v_all = value_states[:, :, -self.window_size:, :]

        if max_keep > 0:
            top_indices = sorted_attn_score_indices[:, :, :max_keep]
            max_valid = key_states.shape[2] - self.window_size - 1
            if max_valid >= 0:
                top_indices = top_indices.clamp(min=0, max=max_valid)
            gather_index = top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            top_k_all = key_states.gather(dim=2, index=gather_index)
            top_v_all = value_states.gather(dim=2, index=gather_index)
        else:
            top_k_all = key_states[:, :, :0, :]
            top_v_all = value_states[:, :, :0, :]

        assert bsz == 1

        # reinit varlen metadata
        k_lens_tensor = keep_k_per_head + self.window_size
        klen_sum = int(k_lens_tensor.sum().item())
        max_seqlen_k = int(k_lens_tensor.max().item())
        k_lens = k_lens_tensor
        self.cu_klen = 0

        selected_k_all = torch.cat([top_k_all, tail_k_all], dim=2)
        selected_v_all = torch.cat([top_v_all, tail_v_all], dim=2)

        total_slots = max_keep + self.window_size
        slot_ids = torch.arange(total_slots, device=_device).unsqueeze(0)
        keep_prefix_mask = slot_ids < keep_k_per_head.unsqueeze(1)
        keep_tail_mask = slot_ids >= max_keep
        keep_mask = keep_prefix_mask | keep_tail_mask

        heads_key_states = selected_k_all[0][keep_mask]
        heads_value_states = selected_v_all[0][keep_mask]

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        # store actual selected positions for recall tracking
        window_start = q_len - self.window_size
        self._selected_indices = []
        for h in range(num_heads):
            keep_k = int(keep_k_per_head[h].item())
            if keep_k > 0:
                h_topk = sorted_attn_score_indices[0, h, :keep_k].cpu()
            else:
                h_topk = torch.tensor([], dtype=torch.long)
            h_window = torch.arange(window_start, q_len, dtype=torch.long)
            self._selected_indices.append(torch.cat([h_topk, h_window]))

        return heads_key_states,heads_value_states



class ReasonSnapKVCluster():
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool',base_capacity=None, head_choice=None, beta=None, temp=None, layer_idx = None, num_hidden_layers = None, num_attention_heads=None, model=None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = base_capacity - window_size
        self.beta = beta
        self.temp = temp

        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None
        self._head_capacity_cached = None
        self._head_capacity_device = None

        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if head_choice == 'random':
            raise ValueError
        elif head_choice == 'copy':
            if 'deepseek' in model.lower():
                path = f'{root_path}/Important_Head/head_score/DeepSeek-R1-Distill-Qwen-1.5B_retrieval.json'
            elif 'llama' in model.lower():
                path = f'{root_path}/Important_Head/head_score/Meta-Llama-3-8B-Instruct_retrieval_heads.json'
            elif 'mistral' in model.lower():
                path = f'{root_path}/Important_Head/head_score/Mistral-7B-Instruct-v0.2_retrieval_heads.json'
            elif 'qwen' in model.lower():
                path = f'{root_path}/Important_Head/head_score/Qwen2.5-7B-Instruct_retrieval.json'
            elif 'glm' in model.lower() or 'chatglm' in model.lower():
                path = f'{root_path}/Important_Head/head_score/glm-4-9b-chat-1m_retrieval.json'
            else:
                raise ValueError
        elif head_choice == 'reason':
            if 'deepseek' in model.lower():
                path = f'{root_path}/Important_Head/head_score/DeepSeek-R1-Distill-Qwen-1.5B_retrieval_reasoning_heads.json'
            elif 'llama' in model.lower():
                path = f'{root_path}/Important_Head/head_score/Meta-Llama-3-8B-Instruct_retrieval_reasoning_heads.json'
            elif 'mistral' in model.lower():
                path = f'{root_path}/Important_Head/head_score/Mistral-7B-Instruct-v0.2_retrieval_reasoning_heads.json'
            elif 'qwen' in model.lower():
                path = f'{root_path}/Important_Head/head_score/Qwen2.5-7B-Instruct_retrieval_reasoning_heads.json'
            elif 'glm' in model.lower() or 'chatglm' in model.lower():
                path = f'{root_path}/Important_Head/head_score/glm-4-9b-chat-1m_retrieval_reasoning_heads.json'
            else:
                raise ValueError
            if not os.path.exists(path):
                raise FileNotFoundError(f"ReasonKV head score file not found: {path}")
        with open(path, 'r') as file:
            head_list = json.loads(file.readline())
        head_score_list = [np.mean(l[1]) for l in head_list.items()]
        head_score_list = torch.tensor(head_score_list / sum(head_score_list))
        head_score_list = torch.pow(head_score_list, self.temp)
        head_score_list = head_score_list / torch.sum(head_score_list)
        self.total_attention = head_score_list.reshape(self.num_hidden_layers, self.num_attention_heads)

        total_pool_capacity = (self.base_capacity // self.beta) * self.num_hidden_layers * self.num_attention_heads
        min_num = (self.base_capacity - self.base_capacity // self.beta)
        self.head_capacity = torch.round(self.total_attention * total_pool_capacity + min_num).int()

        if not hasattr(self, "_capacity_logged"):
            self._capacity_logged = True
            cap = self.head_capacity[self.layer_idx]
            cap_min = int(cap.min().item()) if cap.numel() else 0
            cap_max = int(cap.max().item()) if cap.numel() else 0
            cap_mean = float(cap.float().mean().item()) if cap.numel() else 0.0
            cap_preview = cap[:8].tolist() if cap.numel() else []
            # print(f"[ReasonKV][L{self.layer_idx}] head_cap first8={cap_preview} min={cap_min} max={cap_max} mean={cap_mean:.2f}")
            
    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        cache_mask = getattr(self, "_window_causal_mask", None)
        if (
            cache_mask is None
            or cache_mask.device != attn_weights.device
            or cache_mask.dtype != attn_weights.dtype
        ):
            cache_mask = torch.full(
                (self.window_size, self.window_size),
                torch.finfo(attn_weights.dtype).min,
                device=attn_weights.device,
                dtype=attn_weights.dtype,
            )
            mask_cond = torch.arange(self.window_size, device=attn_weights.device)
            cache_mask.masked_fill_(mask_cond < (mask_cond + 1).view(self.window_size, 1), 0)
            self._window_causal_mask = cache_mask
        attention_mask = cache_mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)
        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return attn_weights_mean_pooling

    def update_kv(self,  key_states, query_states, value_states):
        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            if torch.is_tensor(k_lens):
                self.head_lens = k_lens.to(dtype=torch.int32, device=_device)
            else:
                self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.layer_qlens = torch.ones(num_heads, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.cu_offset = torch.arange(0, num_heads + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads+1, dtype=torch.int32, device=_device)

        if q_len <= self.window_size:
            init_metadata(num_heads, [q_len] * num_heads, q_len * num_heads, q_len)
            self._selected_indices = [torch.arange(q_len, dtype=torch.long) for _ in range(num_heads)]
            return key_states.reshape(-1, head_dim), value_states.reshape(-1, head_dim)

        attn_score = self.calcul_attn_sore(key_states,query_states)

        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * num_heads, q_len * num_heads, q_len)
            # not compress
            self._selected_indices = [torch.arange(q_len, dtype=torch.long) for _ in range(num_heads)]
            return key_states.reshape(-1, head_dim), value_states.reshape(-1, head_dim)

        # if you need to weight the attn_score
        available_k = attn_score.shape[-1]

        if self._head_capacity_cached is None or self._head_capacity_device != _device:
            self._head_capacity_cached = self.head_capacity.to(dtype=torch.int32, device=_device)
            self._head_capacity_device = _device
        keep_k_per_head = self._head_capacity_cached[self.layer_idx]
        keep_k_per_head = keep_k_per_head.clamp(min=0, max=available_k)
        max_keep = int(keep_k_per_head.max().item())

        tail_k_all = key_states[:, :, -self.window_size:, :]
        tail_v_all = value_states[:, :, -self.window_size:, :]

        if max_keep > 0:
            top_indices = torch.topk(attn_score, k=max_keep, dim=-1, largest=True, sorted=True).indices
            max_valid = key_states.shape[2] - self.window_size - 1
            if max_valid >= 0:
                top_indices = top_indices.clamp(min=0, max=max_valid)
            gather_index = top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            top_k_all = key_states.gather(dim=2, index=gather_index)
            top_v_all = value_states.gather(dim=2, index=gather_index)
        else:
            top_k_all = key_states[:, :, :0, :]
            top_v_all = value_states[:, :, :0, :]

        assert bsz == 1

        # reinit varlen metadata
        k_lens_tensor = keep_k_per_head + self.window_size
        klen_sum = int(k_lens_tensor.sum().item())
        max_seqlen_k = int(k_lens_tensor.max().item())
        k_lens = k_lens_tensor
        self.cu_klen = 0

        selected_k_all = torch.cat([top_k_all, tail_k_all], dim=2)
        selected_v_all = torch.cat([top_v_all, tail_v_all], dim=2)

        total_slots = max_keep + self.window_size
        slot_ids = torch.arange(total_slots, device=_device).unsqueeze(0)
        keep_prefix_mask = slot_ids < keep_k_per_head.unsqueeze(1)
        keep_tail_mask = slot_ids >= max_keep
        keep_mask = keep_prefix_mask | keep_tail_mask

        heads_key_states = selected_k_all[0][keep_mask]
        heads_value_states = selected_v_all[0][keep_mask]

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        # store actual selected positions for recall tracking
        window_start = q_len - self.window_size
        self._selected_indices = []
        for h in range(num_heads):
            keep_k = int(keep_k_per_head[h].item())
            if keep_k > 0:
                h_topk = top_indices[0, h, :keep_k].cpu()
            else:
                h_topk = torch.tensor([], dtype=torch.long)
            h_window = torch.arange(window_start, q_len, dtype=torch.long)
            self._selected_indices.append(torch.cat([h_topk, h_window]))

        return heads_key_states,heads_value_states


def init_pyramidkv(self):
    assert hasattr(self.config, 'window_size'), "window_size not set"
    assert hasattr(self.config, 'kernel_size'), "kernel_size not set"
    assert hasattr(self.config, "pooling"), "pooling not set"
    assert hasattr(self.config, "base_capacity"), "base_capacity not set"
    if not hasattr(self.config, "pyram_beta"):
        self.config.pyram_beta = 20
    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = SnapKVCluster(
            window_size = self.config.window_size, 
            max_capacity_prompt = self.config.base_capacity,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            pyram_mode = self.config.pyram_beta,
            )

def init_snapkv(self):

    assert hasattr(self.config, 'window_size'), "window_size not set"
    assert hasattr(self.config, 'kernel_size'), "kernel_size not set"
    assert hasattr(self.config, "pooling"), "pooling not set"
    assert hasattr(self.config, "base_capacity"), "base_capacity not set"
    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = SnapKVCluster(
            window_size = self.config.window_size, 
            max_capacity_prompt = self.config.base_capacity,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,

            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            )
        # print(f"Compress config(Snap): window_size={self.kv_cluster.window_size}, max_capacity_prompt={self.kv_cluster.max_capacity_prompt}, kernel_size={self.kv_cluster.kernel_size}, pooling={self.kv_cluster.pooling}")

def init_reason_snapkv(self):
    assert hasattr(self.config,'window_size'),"window_size not set"
    assert hasattr(self.config,'kernel_size'),"kernel_size not set"
    assert hasattr(self.config,"pooling"),"pooling not set"
    assert hasattr(self.config, "base_capacity"), "base_capacity not set"
    assert hasattr(self.config, 'head_choice'), "head_choice not set"
    assert hasattr(self.config, 'beta'), "beta not set"
    assert hasattr(self.config, 'temp'), 'temp not set'

    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = ReasonSnapKVCluster(
            window_size = self.config.window_size,
            base_capacity=self.config.base_capacity,
            head_choice=self.config.head_choice,
            beta=self.config.beta,
            temp=self.config.temp,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            model=self.config._name_or_path
            )


def init_headkv(self):
    assert hasattr(self.config,'window_size'),"window_size not set"
    assert hasattr(self.config,'kernel_size'),"kernel_size not set"
    assert hasattr(self.config,"pooling"),"pooling not set"
    assert hasattr(self.config, "base_capacity"), "base_capacity not set"
    assert hasattr(self.config,"floor"),"floor not set"
    assert self.config.floor is not None


    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = AdaptiveSnapKVCluster(
            window_size = self.config.window_size,
            base_capacity=self.config.base_capacity,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            floor= self.config.floor,
            skip = self.config.skip,
            layer_idx = self.layer_idx,
            normalize = self.config.normalize,
            num_hidden_layers = self.config.num_hidden_layers,
            )