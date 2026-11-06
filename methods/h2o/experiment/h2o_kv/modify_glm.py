import math
from typing import Optional, Tuple, List
import types
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["enable_glm_pos_shift_attention"]

step_layer_recalls = {}

def glm_pos_shift_attention_forward(
    self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
):
    import sys
    modeling_chatglm = sys.modules[self.__class__.__module__]
    split_tensor_along_last_dim = modeling_chatglm.split_tensor_along_last_dim
    apply_rotary_pos_emb = modeling_chatglm.apply_rotary_pos_emb

    if not hasattr(self, "kv_cache_evictor"):
        from .kv_cache import H2OKVCache
        self.kv_cache_evictor = H2OKVCache(
            recent_size=getattr(self.config, "recent_size", 32),
            h2o_size=getattr(self.config, "h2o_size", 992),
            k_seq_dim=2,
            v_seq_dim=2
        )
        
    bsz, q_len, _ = hidden_states.size()
    
    if q_len > 1:
        self.absolute_full_k = None
        step_layer_recalls["_global_step"] = 0
        
    mixed_x_layer = self.query_key_value(hidden_states)

    if self.multi_query_attention:
        (query_layer, key_layer, value_layer) = mixed_x_layer.split(
            [
                self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
            ],
            dim=-1,
        )
        query_layer = query_layer.view(
            query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        )
        key_layer = key_layer.view(
            key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
        value_layer = value_layer.view(
            value_layer.size()[:-1]
            + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
    else:
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

    query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

    if rotary_pos_emb is not None:
        query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
        key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

    if self.multi_query_attention:
        key_layer = key_layer.unsqueeze(2)
        key_layer = key_layer.expand(
            -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
        )
        key_layer = key_layer.contiguous().view(
            key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
        )
        value_layer = value_layer.unsqueeze(2)
        value_layer = value_layer.expand(
            -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
        )
        value_layer = value_layer.contiguous().view(
            value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
        )

    if kv_cache is not None:
        cache_payload = kv_cache
        if isinstance(cache_payload, (tuple, list)) and len(cache_payload) == 1:
            cache_payload = cache_payload[0]
        cache_k = cache_v = None
        if isinstance(cache_payload, (tuple, list)) and len(cache_payload) == 2:
            cache_k, cache_v = cache_payload
        elif torch.is_tensor(cache_payload):
            if cache_payload.ndim >= 6 and cache_payload.shape[0] == 1 and cache_payload.shape[1] == 2:
                cache_k, cache_v = cache_payload[0, 0], cache_payload[0, 1]
            elif cache_payload.ndim >= 5 and cache_payload.shape[0] == 2:
                cache_k, cache_v = cache_payload[0], cache_payload[1]
        if cache_k is None or cache_v is None:
            kv_cache = None
        else:
            if cache_k.ndim == 5:
                cache_k, cache_v = cache_k[0], cache_v[0]
            key_layer = torch.cat((cache_k, key_layer), dim=2)
            value_layer = torch.cat((cache_v, value_layer), dim=2)

    full_k = None
    if getattr(self, "check_recall", False):
        if not hasattr(self, "absolute_full_k"):
            self.absolute_full_k = None
        with torch.no_grad():
            self.absolute_full_k = key_layer.detach().clone()

    key_states_compress, value_states_compress = self.kv_cache_evictor.evict_for_space(query_layer, key_layer, value_layer)

    if getattr(self, "check_recall", False) and q_len == 1:
        full_k = self.absolute_full_k
        full_len = full_k.shape[2]
        k_size = key_states_compress.shape[2]
        if full_len > k_size:
            from transformers.models.llama.modeling_llama import repeat_kv
            k_full_exp = full_k
            if full_k.shape[1] != query_layer.shape[1]:
                k_full_exp = repeat_kv(full_k, query_layer.shape[1] // full_k.shape[1])

            scores = torch.matmul(query_layer, k_full_exp.transpose(-1, -2)) / math.sqrt(query_layer.shape[-1])
            weights = torch.softmax(scores, dim=-1)
            _, real_topk_indices = torch.topk(weights, k=k_size, dim=-1)
            indices = getattr(self.kv_cache_evictor, "select_idx", None)
            if indices is not None:
                hits = torch.isin(real_topk_indices.view(-1), indices.view(-1))
            else:
                hits = torch.tensor(0.0)
            step_layer_recalls[self.layer_number] = hits.float().mean().item()

            num_layers = getattr(self.config, "num_layers", 40)
            if self.layer_number == num_layers:
                curr_step = step_layer_recalls.get("_global_step", 0)
                all_vals = [step_layer_recalls.get(i, 0.0) for i in range(1, num_layers + 1)]
                avg_recall = np.mean(all_vals)
                with open("h2o_recall_all_steps.csv", "a") as f:
                    f.write(f"{curr_step}," + ",".join([f"{v:.4f}" for v in all_vals]) + f",{avg_recall:.4f}\n")
                step_layer_recalls.clear()
                step_layer_recalls["_global_step"] = curr_step + 1

    if use_cache:
        if kv_cache is None:
            # First handling kv_cache initialization with compressed states
            kv_cache = torch.cat((key_states_compress.unsqueeze(0).unsqueeze(0), value_states_compress.unsqueeze(0).unsqueeze(0)),
                                 dim=1)
        else:
            kv_cache = (key_states_compress, value_states_compress)
    else:
        kv_cache = None

    context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
    output = self.dense(context_layer)

    return output, kv_cache


def enable_glm_pos_shift_attention(model, check_recall=False):
    if check_recall:
        step_layer_recalls.clear()
        step_layer_recalls["_global_step"] = 0
        num_layers = getattr(model.config, "num_layers", 40)
        header = "step," + ",".join([f"layer_{i}" for i in range(1, num_layers + 1)]) + ",model_avg\n"
        with open("h2o_recall_all_steps.csv", "w") as f:
            f.write(header)

    for name, module in model.named_modules():
        if module.__class__.__name__ == "SelfAttention":
            module.config = model.config
            module.forward = types.MethodType(glm_pos_shift_attention_forward, module)
            module.check_recall = check_recall
