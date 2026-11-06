
import os
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads to match query heads.
    Input: [batch, num_key_value_heads, seqlen, head_dim]
    Output: [batch, num_attention_heads, seqlen, head_dim]
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def global_kv_len(past_key_value, cur_len):
    """Return the largest KV length currently held by any layer.

    PyramidKV allocates a different budget per layer, so layers hold different
    KV lengths.  This returns the maximum (including the current layer) and is
    only used by the non-uniform vs uniform attention-form ablation.
    """
    max_len = cur_len
    key_cache = getattr(past_key_value, "key_cache", None)
    if key_cache is None:
        return max_len
    for tensor in key_cache:
        if tensor is not None:
            max_len = max(max_len, tensor.shape[-2])
    return max_len


def sdpa_decode_with_form(
    query_states, key_states, value_states, attention_mask, target_len, sliding_window=None
):
    """Decode-only SDPA attention used for the non-uniform vs uniform ablation.

    Layout follows FlashAttention: ``[bsz, seq, num_heads, head_dim]``.  Only the
    *shape* of the computation changes; retained tokens and their order are kept
    identical, so the result matches the FlashAttention path.
    ``target_len`` is the KV length every layer is computed against:

    * ``target_len == key_states.shape[1]`` -> exact per-layer length (no padding)
    * ``target_len == global max length``   -> uniform form: right padding plus a
      mask that hides the padded slots (wasted FLOPs only, math unchanged)

    Padding is compute-only; the KV cache storage is never modified.
    """
    bsz = query_states.shape[0]
    real_len = key_states.shape[1]
    device = key_states.device

    if attention_mask is not None and attention_mask.dim() == 2:
        keep = attention_mask[:, -real_len:].to(torch.bool)
    else:
        # Padding-free batches (the efficiency harness) pass no usable 2D mask.
        keep = torch.ones((bsz, real_len), dtype=torch.bool, device=device)

    if sliding_window is not None and real_len > sliding_window:
        keep[:, : real_len - sliding_window] = False

    pad = target_len - real_len
    if pad > 0:
        num_heads, head_dim = key_states.shape[2], key_states.shape[3]
        key_states = torch.cat(
            [key_states, key_states.new_zeros((bsz, pad, num_heads, head_dim))], dim=1
        )
        value_states = torch.cat(
            [value_states, value_states.new_zeros((bsz, pad, num_heads, head_dim))], dim=1
        )
        keep = torch.cat(
            [keep, keep.new_zeros((bsz, pad), dtype=torch.bool)], dim=1
        )

    attn_mask = keep[:, None, None, :]  # [bsz, 1, 1, target_len]
    q = query_states.transpose(1, 2)
    k = key_states.transpose(1, 2)
    v = value_states.transpose(1, 2)
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
    )
    return out.transpose(1, 2)  # [bsz, q_len, num_heads, head_dim]


class PyramidKVCluster():
    """
    PyramidKV: core class for KV cache compression during prefill.
    
    Core idea:
    - keep the most recent window_size tokens (local window, ensures generation quality)
    - for older historical tokens, select the most important ones based on attention scores
    - shallower layers keep more tokens, deeper layers keep fewer (pyramid structure)
    
    Max capacity per layer is determined by the formula:
      max_capacity_prompt[layer] = max_num - layer_idx * steps
    where max_num/min_num are based on max_capacity_prompt and beta calculation.
    """
    def __init__(self, num_hidden_layers = 32, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', beta = 20, num_layers = 80, layer_idx=None):

        self.layer_idx = layer_idx          # current layer index
        self.num_hidden_layers = num_hidden_layers  # total number of layers

        self.steps = -1
        self.beta = beta  # larger beta -> larger per-layer capacity differences

        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self._debug_printed = False

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups, tm=None):
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        bsz_k, num_kv_heads, q_len_k, head_dim_k = key_states.shape

        key_states_orig = key_states
        value_states_orig = value_states
        padding_mask = attention_mask
        if num_kv_heads != num_heads:
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)

        # ===== calculationPyramid capacity =====
        if tm is not None:
            tm.record_pref_pattern_start(self.layer_idx)
        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num

        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num

        steps = (max_num - min_num) // self.num_hidden_layers
        max_capacity_prompt = max_num - self.layer_idx * steps
        if tm is not None:
            tm.record_pref_pattern_end(self.layer_idx)

        if os.environ.get("PYRAMIDKV_DEBUG_CAPACITY", "0") == "1":
            print(f"[KV_DEBUG] layer={self.layer_idx} | "
                  f"base_capacity={self.max_capacity_prompt} | "
                  f"window={self.window_size} | "
                  f"beta={self.beta} | "
                  f"q_len={q_len} | "
                  f"min_num={min_num} max_num={max_num} steps={steps} | "
                  f"dynamic_capacity={max_capacity_prompt}")

        if tm is not None:
            tm.record_pref_idx_start(self.layer_idx)

        
        if q_len < self.max_capacity_prompt:
            if os.environ.get("PYRAMIDKV_DEBUG_CAPACITY", "0") == "1":
                print(f"[KV_DEBUG] layer={self.layer_idx} | BRANCH=1 (no compress) | "
                      f"q_len={q_len} < base_capacity={self.max_capacity_prompt}")
            if tm is not None:
                tm.record_pref_idx_end(self.layer_idx)
            return key_states_orig, value_states_orig

        elif q_len < (self.max_capacity_prompt - self.window_size) * 2:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            causal_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += causal_mask
            if padding_mask is not None and padding_mask.dim() == 2:
                valid_tokens = padding_mask[:, -q_len:].to(torch.bool)
                attn_weights.masked_fill_(~valid_tokens[:, None, None, :], torch.finfo(attn_weights.dtype).min)

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = torch.nan_to_num(attn_weights)
            if padding_mask is not None and padding_mask.dim() == 2:
                query_mask = padding_mask[:, -self.window_size:].to(device=attn_weights.device, dtype=attn_weights.dtype)
                attn_weights *= query_mask[:, None, :, None]
            
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            # indices shape: [bsz, 28, k] — 28-head version
            
            indices = indices[:, ::num_key_value_groups, :]  # [bsz, 4, k]
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim_k)  # [bsz, 4, k, 128]
            
            if os.environ.get("PYRAMIDKV_DEBUG_CAPACITY", "0") == "1":
                print(f"[KV_DEBUG] layer={self.layer_idx} | BRANCH=2 (medium) | "
                      f"q_len={q_len} | "
                      f"compress_to={self.max_capacity_prompt} | "
                      f"window={self.window_size} | "
                      f"keep_topk={self.max_capacity_prompt - self.window_size}")
            k_past_compress = key_states_orig[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states_orig[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states_orig[:, :, -self.window_size:, :]
            v_cur = value_states_orig[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            if tm is not None:
                tm.record_pref_idx_end(self.layer_idx)
            return key_states, value_states

        else:
            if os.environ.get("PYRAMIDKV_DEBUG_CAPACITY", "0") == "1":
                print(f"[KV_DEBUG] layer={self.layer_idx} | BRANCH=3 (long) | "
                      f"q_len={q_len} | "
                      f"dynamic_capacity={max_capacity_prompt} | "
                      f"window={self.window_size}")
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            causal_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += causal_mask
            if padding_mask is not None and padding_mask.dim() == 2:
                valid_tokens = padding_mask[:, -q_len:].to(torch.bool)
                attn_weights.masked_fill_(~valid_tokens[:, None, None, :], torch.finfo(attn_weights.dtype).min)

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = torch.nan_to_num(attn_weights)
            if padding_mask is not None and padding_mask.dim() == 2:
                query_mask = padding_mask[:, -self.window_size:].to(device=attn_weights.device, dtype=attn_weights.dtype)
                attn_weights *= query_mask[:, None, :, None]
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices
            indices = indices[:, ::num_key_value_groups, :]  # [bsz, 4, k]
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim_k)
            k_past_compress = key_states_orig[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states_orig[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states_orig[:, :, -self.window_size:, :]
            v_cur = value_states_orig[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            if tm is not None:
                tm.record_pref_idx_end(self.layer_idx)
            return key_states, value_states
    

def init_pyramidkv(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'pyram_beta'):
            self.config.pyram_beta = 20

        if os.environ.get("PYRAMIDKV_DEBUG_CAPACITY", "0") == "1":
            print(f"[KV_DEBUG_INIT] layer={self.layer_idx} | CREATING kv_cluster with: "
                  f"max_capacity_prompt={self.config.max_capacity_prompt}, "
                  f"window_size={self.config.window_size}, "
                  f"kernel_size={self.config.kernel_size}, "
                  f"pooling={self.config.pooling}, "
                  f"beta={self.config.pyram_beta}")

    self.kv_cluster = PyramidKVCluster(
        num_hidden_layers = num_hidden_layers,
        layer_idx = self.layer_idx,
        window_size = self.config.window_size,
        max_capacity_prompt = self.config.max_capacity_prompt,
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        beta = self.config.pyram_beta,
        )
