import torch
import torch.nn.functional as F
import numpy as np

class CompressConfig:
    def __init__(self, compress=False, cascading=False, cache_size=1024, window_size=32, hyper=None):
        self.compress = compress
        self.cascading = cascading
        self.cache_size = cache_size
        self.window_size = window_size
        self.hyper = hyper
    
    def __str__(self):
        return f"Config(cache_size={self.cache_size}, window_size={self.window_size}, hyper={self.hyper})"

def calculate_entropy(attention_scores):
    attention_scores = attention_scores.to(torch.float32)
    entropy = -torch.sum(attention_scores * torch.log(attention_scores + 1e-10))  
    entropy= entropy.to(dtype=torch.float32)
    return entropy


def global_kv_len(past_key_value, cur_len):
    """Return the largest KV length currently held by any layer.

    CakeKV assigns a different budget (and window) per layer, so layers hold
    different KV lengths.  This returns the maximum (including the current
    layer) and is only used by the non-uniform vs uniform form ablation.
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
    ``target_len`` is the KV length the layer is computed against:

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

def adjust_budgets(budget_list, total_budget, seq_len, layer_nums):

    budget_list = np.array(budget_list, dtype=int)
    # Limit the budget of all layers to not exceed seq_len
    excess = np.maximum(budget_list - seq_len, 0)
    budget_list = np.minimum(budget_list, seq_len)

    # Adjust excess budget
    total_excess = np.sum(excess)

    if total_excess > 0:

        valid_indices = budget_list < seq_len
        num_valid = np.sum(valid_indices)

        if num_valid > 0:
            
            distribute_per_layer = total_excess // num_valid
            remainder = total_excess % num_valid

            budget_list[valid_indices] += distribute_per_layer
            budget_list[np.where(valid_indices)[0][:remainder]] += 1

    # Ensure total budget equals total_budget
    current_total_budget = np.sum(budget_list)
    budget_diff = total_budget - current_total_budget

    if budget_diff != 0:
        if budget_diff > 0:
            valid_indices = budget_list < seq_len  
        else:
            valid_indices = budget_list > 1  

        num_valid = np.sum(valid_indices)

        if num_valid > 0:
            adjust_per_layer = abs(budget_diff) // num_valid
            remainder = abs(budget_diff) % num_valid

            if budget_diff > 0:
                budget_list[valid_indices] += adjust_per_layer
                budget_list[np.where(valid_indices)[0][:remainder]] += 1
            else:
                budget_list[valid_indices] -= adjust_per_layer
                budget_list[np.where(valid_indices)[0][:remainder]] -= 1

    return budget_list.tolist()