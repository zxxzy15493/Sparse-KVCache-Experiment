import math
import time
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from transformers.modeling_outputs import BaseModelOutputWithPast

from ..cake_cache import CakeCache, CakeDecodingKVCache_LayerWise
from ..utils import calculate_entropy


def chatglm_attn_forward_cake(self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True):
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
            value_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
    else:
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
        query_layer, key_layer, value_layer = torch.split(mixed_x_layer, self.hidden_size_per_attention_head, dim=-1)

    query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

    if rotary_pos_emb is not None:
        b, np, sq, _ = query_layer.size()
        nk = key_layer.size(1)
        rot_dim = rotary_pos_emb.shape[-2] * 2

        q_pass = query_layer[..., rot_dim:]
        k_pass = key_layer[..., rot_dim:]
        rope_cache = rotary_pos_emb[:, :sq]

        q_shaped = query_layer[..., :rot_dim].reshape(b, np, sq, rot_dim // 2, 2)
        k_shaped = key_layer[..., :rot_dim].reshape(b, nk, sq, rot_dim // 2, 2)
        rope_cache = rope_cache.view(-1, 1, sq, q_shaped.size(3), 2)

        query_layer = torch.cat(
            (
                torch.stack(
                    [
                        q_shaped[..., 0] * rope_cache[..., 0] - q_shaped[..., 1] * rope_cache[..., 1],
                        q_shaped[..., 1] * rope_cache[..., 0] + q_shaped[..., 0] * rope_cache[..., 1],
                    ],
                    -1,
                ).flatten(3),
                q_pass,
            ),
            dim=-1,
        )
        key_layer = torch.cat(
            (
                torch.stack(
                    [
                        k_shaped[..., 0] * rope_cache[..., 0] - k_shaped[..., 1] * rope_cache[..., 1],
                        k_shaped[..., 1] * rope_cache[..., 0] + k_shaped[..., 0] * rope_cache[..., 1],
                    ],
                    -1,
                ).flatten(3),
                k_pass,
            ),
            dim=-1,
        )

    layer_idx = self.layer_number - 1
    is_cake_cache = isinstance(kv_cache, CakeCache)

    if is_cake_cache:
        if (
            self.config.decoding_evict[layer_idx] is None
            and len(kv_cache.layer_budget) == self.config.prefill_cake_evict[layer_idx].num_layers
        ):
            self.config.decoding_evict[layer_idx] = CakeDecodingKVCache_LayerWise(
                hh_size=kv_cache.layer_budget[layer_idx],
                window_size=self.config.window_size[layer_idx],
                k_seq_dim=2,
                v_seq_dim=2,
            )

        key_layer, value_layer = kv_cache.update(key_layer, value_layer, layer_idx)
    else:
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=2)
            value_layer = torch.cat((cache_v, value_layer), dim=2)

    if self.multi_query_attention:
        key_for_score = key_layer.unsqueeze(2)
        key_for_score = key_for_score.expand(
            -1,
            -1,
            self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition,
            -1,
            -1,
        )
        key_for_score = key_for_score.contiguous().view(
            key_for_score.size()[:1] + (self.num_attention_heads_per_partition,) + key_for_score.size()[3:]
        )

        value_for_score = value_layer.unsqueeze(2)
        value_for_score = value_for_score.expand(
            -1,
            -1,
            self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition,
            -1,
            -1,
        )
        value_for_score = value_for_score.contiguous().view(
            value_for_score.size()[:1] + (self.num_attention_heads_per_partition,) + value_for_score.size()[3:]
        )
    else:
        key_for_score, value_for_score = key_layer, value_layer

    if is_cake_cache and self.config.prefill[layer_idx]:
        window_size = self.config.window_size[layer_idx]
        q_len = query_layer.shape[2]

        tmp_attn_weights = torch.matmul(
            query_layer[..., -window_size:, :], key_for_score.transpose(2, 3)
        ) / math.sqrt(self.hidden_size_per_attention_head)

        if q_len != 1:
            mask = torch.full(
                (window_size, window_size),
                torch.finfo(tmp_attn_weights.dtype).min,
                device=tmp_attn_weights.device,
            )
            mask_cond = torch.arange(mask.size(-1), device=tmp_attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            tmp_attention_mask = mask[None, None, :, :]
            tmp_attn_weights[:, :, -window_size:, -window_size:] += tmp_attention_mask

        tmp_attn_weights = nn.functional.softmax(tmp_attn_weights, dim=-1, dtype=torch.float32).to(query_layer.dtype)

        disp = calculate_entropy(tmp_attn_weights[:, :, -window_size:, :-window_size])
        var = torch.var(tmp_attn_weights[:, :, -window_size:, :-window_size], dim=-2).sum(0).sum(0).sum(0)
        pref_score = (disp ** (1 / self.config.tau1) * var ** (1 / self.config.tau2)).cpu().numpy()

        attention_score = tmp_attn_weights[:, :, -window_size:, :]
        attn_mean = attention_score.mean(dim=-2)
        attn_var = attention_score.var(dim=-2)
        attn_cache = attn_mean + self.config.gamma * attn_var
        attn_cache = attn_cache[:, :, :-window_size]
        attn_cache = F.avg_pool1d(attn_cache, kernel_size=5, padding=2, stride=1)

        kv_heads = self.num_multi_query_groups_per_partition if self.multi_query_attention else self.num_attention_heads_per_partition
        kv_groups = self.num_attention_heads_per_partition // kv_heads
        hh_score = attn_cache.reshape(attn_cache.shape[0], kv_heads, kv_groups, -1).mean(dim=-2)

        kv_cache.update_score(pref_score, hh_score)
        kv_cache.layer_budget.append(self.config.key_size[layer_idx])
        self.config.prefill[layer_idx] = False
        kv_cache = self.config.prefill_cake_evict[layer_idx](kv_cache, key_layer.shape[2])

        #print(f"[CAKE DEBUG][Layer {layer_idx}] prefill: window_size={window_size}, q_len={q_len}, key_layer.shape={key_layer.shape}, value_layer.shape={value_layer.shape}")
        # print(f"[CAKE DEBUG][Layer {layer_idx}] kv_cache.layer_budget={getattr(kv_cache, 'layer_budget', None)}")

    if is_cake_cache and self.config.decoding_evict[layer_idx] is not None:
        window_size = self.config.window_size[layer_idx]
        tmp_attn_weights = torch.matmul(
            query_layer[..., -window_size:, :], key_for_score.transpose(2, 3)
        ) / math.sqrt(self.hidden_size_per_attention_head)
        tmp_attn_weights = nn.functional.softmax(tmp_attn_weights, dim=-1, dtype=torch.float32).to(query_layer.dtype)
        kv_cache = self.config.decoding_evict[layer_idx](kv_cache, tmp_attn_weights, layer_idx)
        key_layer, value_layer = kv_cache[layer_idx]
        if self.multi_query_attention:
            key_for_score = key_layer.unsqueeze(2)
            key_for_score = key_for_score.expand(
                -1,
                -1,
                self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition,
                -1,
                -1,
            )
            key_for_score = key_for_score.contiguous().view(
                key_for_score.size()[:1] + (self.num_attention_heads_per_partition,) + key_for_score.size()[3:]
            )

            value_for_score = value_layer.unsqueeze(2)
            value_for_score = value_for_score.expand(
                -1,
                -1,
                self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition,
                -1,
                -1,
            )
            value_for_score = value_for_score.contiguous().view(
                value_for_score.size()[:1] + (self.num_attention_heads_per_partition,) + value_for_score.size()[3:]
            )
        else:
            key_for_score, value_for_score = key_layer, value_layer

    context_layer = self.core_attention(query_layer, key_for_score, value_for_score, attention_mask)
    output = self.dense(context_layer)

    if is_cake_cache:
        return output, kv_cache

    if use_cache:
        if kv_cache is None:
            kv_cache = torch.cat(
                (key_layer.unsqueeze(0).unsqueeze(0), value_layer.unsqueeze(0).unsqueeze(0)),
                dim=1,
            )
        else:
            kv_cache = (key_layer, value_layer)
    else:
        kv_cache = None

    return output, kv_cache


def chatglm_model_forward_cake(
    self,
    input_ids,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.BoolTensor] = None,
    full_attention_mask: Optional[torch.BoolTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    call_start = time.perf_counter()
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    batch_size, seq_length = input_ids.shape

    if inputs_embeds is None:
        inputs_embeds = self.embedding(input_ids)

    seq_length = input_ids.shape[1]
    timing = getattr(self, "_cake_timing", None)
    if timing is None or seq_length > 1:
        timing = {
            "request_start": call_start,
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "decode_steps": 0,
            "ttft": None,
            "tpot": None,
            "latency": None,
        }
        self._cake_timing = timing

    if past_key_values is not None and not isinstance(past_key_values, CakeCache):
        past_key_values = CakeCache.from_legacy_cache(past_key_values)

    if full_attention_mask is None:
        if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
            full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

    rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
    if position_ids is not None:
        rotary_pos_emb = rotary_pos_emb[position_ids]
    else:
        rotary_pos_emb = rotary_pos_emb[None, :seq_length]

    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None

    kv_cache = past_key_values if (use_cache and past_key_values is not None) else (CakeCache() if use_cache else None)

    for layer in self.encoder.layers:
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states, kv_cache = layer(
            hidden_states,
            full_attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if self.encoder.post_layer_norm:
        hidden_states = self.encoder.final_layernorm(hidden_states)

    call_end = time.perf_counter()
    if seq_length > 1:
        timing["prefill_time"] = call_end - call_start
    else:
        decode_time = call_end - call_start
        timing["decode_time"] += decode_time
        timing["decode_steps"] += 1
        if timing["ttft"] is None:
            timing["ttft"] = call_end - timing["request_start"]
        timing["latency"] = call_end - timing["request_start"]
        timing["tpot"] = timing["decode_time"] / max(timing["decode_steps"], 1)

    presents = kv_cache.to_legacy_cache() if (use_cache and isinstance(kv_cache, CakeCache)) else kv_cache

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, None] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=None,
    )
