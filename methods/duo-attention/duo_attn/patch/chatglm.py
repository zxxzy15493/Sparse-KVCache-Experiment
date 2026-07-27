import math
from typing import Optional, Tuple

import torch
from torch import nn
import types

from flash_attn import flash_attn_func

from duo_attn.ulysses import UlyssesAttention


def _apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    rope_cache = rope_cache[:, :sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


def _project_qkv(self, hidden_states):
    mixed_x_layer = self.query_key_value(hidden_states)
    if self.multi_query_attention:
        query_layer, key_layer, value_layer = mixed_x_layer.split(
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
        query_layer, key_layer, value_layer = torch.chunk(mixed_x_layer, 3, dim=-1)

    query_layer, key_layer, value_layer = [x.transpose(1, 2) for x in (query_layer, key_layer, value_layer)]
    return query_layer, key_layer, value_layer


def _expand_multi_query(self, key_layer, value_layer):
    if self.multi_query_attention:
        key_layer = key_layer.unsqueeze(2)
        key_layer = key_layer.expand(
            -1,
            -1,
            self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition,
            -1,
            -1,
        )
        key_layer = key_layer.contiguous().view(
            key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
        )

        value_layer = value_layer.unsqueeze(2)
        value_layer = value_layer.expand(
            -1,
            -1,
            self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition,
            -1,
            -1,
        )
        value_layer = value_layer.contiguous().view(
            value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
        )
    return key_layer, value_layer


def _streaming_kv(key_layer, value_layer, sink_size: int, recent_size: int):
    kv_seq_len = key_layer.shape[2]
    keep = sink_size + recent_size
    if kv_seq_len <= keep:
        return key_layer, value_layer
    sink_key = key_layer[:, :, :sink_size, :]
    sink_val = value_layer[:, :, :sink_size, :]
    rec_key = key_layer[:, :, -recent_size:, :]
    rec_val = value_layer[:, :, -recent_size:, :]
    return torch.cat([sink_key, rec_key], dim=2), torch.cat([sink_val, rec_val], dim=2)


def chatglm_duo_attention_forward_one_way(
    self,
    hidden_states,
    attention_mask,
    rotary_pos_emb,
    kv_cache=None,
    use_cache=True,
):
    import sys
    modeling_chatglm = sys.modules[self.__class__.__module__]
    split_tensor_along_last_dim = modeling_chatglm.split_tensor_along_last_dim
    apply_rotary_pos_emb = modeling_chatglm.apply_rotary_pos_emb

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
            query_layer.size()[:-1]
            + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        )
        key_layer = key_layer.view(
            key_layer.size()[:-1]
            + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
        value_layer = value_layer.view(
            value_layer.size()[:-1]
            + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
    else:
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

    query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

    if rotary_pos_emb is not None:
        query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
        key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

    cache_key_layer = key_layer
    cache_value_layer = value_layer

    if kv_cache is not None:
        if isinstance(kv_cache, torch.Tensor):
            cache_k, cache_v = kv_cache[:, 0], kv_cache[:, 1]
            cache_k = cache_k.squeeze(0)
            cache_v = cache_v.squeeze(0)
        else:
            cache_k, cache_v = kv_cache
        while cache_k.dim() < key_layer.dim():
            cache_k = cache_k.unsqueeze(0)
            cache_v = cache_v.unsqueeze(0)
        key_layer = torch.cat((cache_k, key_layer), dim=2)
        value_layer = torch.cat((cache_v, value_layer), dim=2)
        cache_key_layer = key_layer
        cache_value_layer = value_layer

    if use_cache:
        if kv_cache is None:
            next_kv_cache = torch.cat(
                (
                    cache_key_layer.unsqueeze(0).unsqueeze(0),
                    cache_value_layer.unsqueeze(0).unsqueeze(0),
                ),
                dim=1,
            )
        else:
            next_kv_cache = (cache_key_layer, cache_value_layer)
    else:
        next_kv_cache = None

    key_layer, value_layer = _expand_multi_query(self, key_layer, value_layer)

    if not hasattr(self, "full_attn_head_mask") or self.full_attn_head_mask is None:
        self.full_attn_head_mask = self.full_attention_heads > 0.5
        self.num_full_attn_head = int(self.full_attn_head_mask.sum().item()) * self.num_key_value_groups
        self.num_streaming_attn_head = self.num_attention_heads_per_partition - self.num_full_attn_head
        import os
        if int(os.environ.get("DUO_ATTN_DEBUG", "0")):
            print(f"[DUO_ATTN_DEBUG] layer={getattr(self, 'layer_number', '?')} full_attn_head_mask={self.full_attn_head_mask.tolist()} "
                  f"num_full_kv={int(self.full_attn_head_mask.sum().item())} num_full_q={self.num_full_attn_head} "
                  f"num_streaming_q={self.num_streaming_attn_head} kv_groups={self.num_key_value_groups}")

    seq_len = query_layer.size(2)
    kv_seq_len = key_layer.size(2)
    
    import os
    if int(os.environ.get("DUO_ATTN_DEBUG", "0")) and seq_len != kv_seq_len:
        print(f"[DUO_ATTN_DEBUG] layer={getattr(self, 'layer_number', '?')} DECODE: q_len={seq_len} kv_len={kv_seq_len} "
              f"num_full_q={self.num_full_attn_head} num_streaming_q={self.num_streaming_attn_head}")
    
    if seq_len == kv_seq_len:
        output = self.dense(self.core_attention(query_layer, key_layer, value_layer, attention_mask))
    elif self.num_streaming_attn_head == 0:
        output = self.dense(self.core_attention(query_layer, key_layer, value_layer, attention_mask))
    elif self.num_full_attn_head == 0:
        streaming_key, streaming_value = _streaming_kv(
            key_layer.transpose(1, 2),
            value_layer.transpose(1, 2),
            self.sink_size,
            self.recent_size,
        )
        
        streaming_out = flash_attn_func(
            query_layer.transpose(1, 2),
            streaming_key,
            streaming_value,
            causal=True,
            dropout_p=0.0,
        )
        
        new_context_layer_shape = streaming_out.size()[:-2] + (self.projection_size,)
        attn_output = streaming_out.reshape(*new_context_layer_shape)
        output = self.dense(attn_output)
    else:
        full_query = query_layer[:, :self.num_full_attn_head, :, :].transpose(1, 2)
        full_key = key_layer[:, :self.num_full_attn_head, :, :].transpose(1, 2)
        full_value = value_layer[:, :self.num_full_attn_head, :, :].transpose(1, 2)
        
        full_out = flash_attn_func(
            full_query,
            full_key,
            full_value,
            causal=True,
            dropout_p=0.0,
        )
        
        del full_key, full_value
        
        streaming_query = query_layer[:, self.num_full_attn_head:, :, :].transpose(1, 2)
        streaming_key = key_layer[:, self.num_full_attn_head:, :, :].transpose(1, 2)
        streaming_value = value_layer[:, self.num_full_attn_head:, :, :].transpose(1, 2)
        
        del query_layer, key_layer, value_layer
        
        streaming_key, streaming_value = _streaming_kv(
            streaming_key,
            streaming_value,
            self.sink_size,
            self.recent_size,
        )
        
        streaming_out = flash_attn_func(
            streaming_query,
            streaming_key,
            streaming_value,
            causal=True,
            dropout_p=0.0,
        )
        
        del streaming_key, streaming_value
        
        attn_output = torch.cat([full_out, streaming_out], dim=2)
        
        del full_out, streaming_out
        
        new_context_layer_shape = attn_output.size()[:-2] + (self.projection_size,)
        attn_output = attn_output.reshape(*new_context_layer_shape)
        output = self.dense(attn_output)

    return output, next_kv_cache


def chatglm_duo_attention_forward_two_way(
    self,
    hidden_states,
    attention_mask,
    rotary_pos_emb,
    kv_cache=None,
    use_cache=True,
):
    bsz_x_2 = hidden_states.size(0)
    assert bsz_x_2 % 2 == 0
    bsz = bsz_x_2 // 2
    full_hidden_states = hidden_states[:bsz]
    streaming_hidden_states = hidden_states[bsz:]

    with torch.no_grad():
        full_q, full_k, full_v = _project_qkv(self, full_hidden_states)
    stream_q, stream_k, stream_v = _project_qkv(self, streaming_hidden_states)

    if rotary_pos_emb is not None:
        with torch.no_grad():
            full_q = _apply_rotary_pos_emb(full_q, rotary_pos_emb)
            full_k = _apply_rotary_pos_emb(full_k, rotary_pos_emb)
        stream_q = _apply_rotary_pos_emb(stream_q, rotary_pos_emb)
        stream_k = _apply_rotary_pos_emb(stream_k, rotary_pos_emb)

    full_k_exp, full_v_exp = _expand_multi_query(self, full_k, full_v)
    stream_k_exp, stream_v_exp = _expand_multi_query(self, stream_k, stream_v)

    with torch.no_grad():
        full_attn_output = self.full_attn_func(
            full_q.transpose(1, 2),
            full_k_exp.transpose(1, 2),
            full_v_exp.transpose(1, 2),
            causal=True,
            dropout_p=0.0,
        )

    streaming_attn_output = self.streaming_attn_func(
        stream_q.transpose(1, 2),
        stream_k_exp.transpose(1, 2),
        stream_v_exp.transpose(1, 2),
        causal=True,
        dropout_p=0.0,
    )

    full_query_head_mask = torch.repeat_interleave(
        self.full_attention_heads > 0.5, self.num_key_value_groups
    )
    full_query_head_mask = full_query_head_mask.view(1, 1, -1, 1).to(streaming_attn_output.dtype)
    mixed_stream = (1 - full_query_head_mask) * streaming_attn_output + full_query_head_mask * full_attn_output

    with torch.no_grad():
        full_output = self.dense(full_attn_output.reshape(full_attn_output.size(0), full_attn_output.size(1), -1).contiguous())
    streaming_output = self.dense(mixed_stream.reshape(mixed_stream.size(0), mixed_stream.size(1), -1).contiguous())

    output = torch.cat([full_output, streaming_output], dim=0)
    return output, kv_cache


def _iter_chatglm_attn_modules(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "encoder"):
        for layer in model.transformer.encoder.layers:
            yield layer.self_attention
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        for layer in model.encoder.layers:
            yield layer.self_attention
    else:
        raise ValueError("Model type not supported")


def enable_chatglm_duo_attention_training(
    model,
    sink_size,
    recent_size,
    max_length,
    initial_value=1.0,
    enable_ulysses_attention=False,
    streaming_attn_implementation="blocksparse",
):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for module in _iter_chatglm_attn_modules(model):
        if not hasattr(module, "config"):
            module.config = model.config

        module.forward = types.MethodType(chatglm_duo_attention_forward_two_way, module)
        module.sink_size = sink_size
        module.recent_size = recent_size

        kv_heads = (
            module.num_multi_query_groups_per_partition
            if module.multi_query_attention
            else module.num_attention_heads_per_partition
        )
        module.num_key_value_groups = module.num_attention_heads_per_partition // kv_heads

        module.register_parameter(
            "full_attention_heads",
            nn.Parameter(torch.ones(kv_heads, device=device, dtype=dtype, requires_grad=True) * initial_value),
        )
        if not enable_ulysses_attention:
            module.streaming_attn_func = flash_attn_func
            module.full_attn_func = flash_attn_func
        else:
            module.streaming_attn_func = UlyssesAttention(attn_func=flash_attn_func)
            module.full_attn_func = UlyssesAttention(attn_func=flash_attn_func)


def enable_chatglm_duo_attention_eval(model, full_attention_heads, sink_size, recent_size):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for idx, module in enumerate(_iter_chatglm_attn_modules(model)):
        if not hasattr(module, "config"):
            module.config = model.config

        module.forward = types.MethodType(chatglm_duo_attention_forward_one_way, module)
        module.sink_size = sink_size
        module.recent_size = recent_size

        kv_heads = (
            module.num_multi_query_groups_per_partition
            if module.multi_query_attention
            else module.num_attention_heads_per_partition
        )
        kv_groups = module.num_attention_heads_per_partition // kv_heads
        module.num_key_value_groups = kv_groups

        layer_full_attention_heads_raw = torch.tensor(full_attention_heads[idx], device=device, dtype=dtype)
        full_attn_mask = layer_full_attention_heads_raw > 0.5  # [kv_heads], bool tensor

        # Q: [num_q_heads * head_dim, hidden]  (4096 x 4096)
        # K: [kv_heads * head_dim, hidden]     (512 x 4096)
        # V: [kv_heads * head_dim, hidden]     (512 x 4096)
        head_dim = module.hidden_size_per_attention_head
        num_q_heads = module.num_attention_heads_per_partition
        qkv = module.query_key_value
        q_slice = num_q_heads * head_dim
        kv_slice = kv_heads * head_dim

        q_w = qkv.weight.data[:q_slice]
        k_w = qkv.weight.data[q_slice:q_slice + kv_slice]
        v_w = qkv.weight.data[q_slice + kv_slice:]

        q_mask = torch.repeat_interleave(full_attn_mask, kv_groups * head_dim)
        q_w_reordered = torch.cat([q_w[q_mask], q_w[~q_mask]], dim=0)

        kv_mask = torch.repeat_interleave(full_attn_mask, head_dim)
        k_w_reordered = torch.cat([k_w[kv_mask], k_w[~kv_mask]], dim=0)
        v_w_reordered = torch.cat([v_w[kv_mask], v_w[~kv_mask]], dim=0)

        if qkv.bias is not None:
            q_b = qkv.bias.data[:q_slice]
            k_b = qkv.bias.data[q_slice:q_slice + kv_slice]
            v_b = qkv.bias.data[q_slice + kv_slice:]
            q_b_reordered = torch.cat([q_b[q_mask], q_b[~q_mask]], dim=0)
            k_b_reordered = torch.cat([k_b[kv_mask], k_b[~kv_mask]], dim=0)
            v_b_reordered = torch.cat([v_b[kv_mask], v_b[~kv_mask]], dim=0)
            qkv.bias.data = torch.cat([q_b_reordered, k_b_reordered, v_b_reordered], dim=0)

        qkv.weight.data = torch.cat([q_w_reordered, k_w_reordered, v_w_reordered], dim=0)

        o_mask = torch.repeat_interleave(full_attn_mask, kv_groups * head_dim)
        dw = module.dense.weight.data
        module.dense.weight.data = torch.cat([dw[:, o_mask], dw[:, ~o_mask]], dim=1)

        num_full = int(full_attn_mask.sum().item())
        layer_full_attention_heads = torch.zeros(kv_heads, device=device, dtype=dtype)
        layer_full_attention_heads[:num_full] = 1.0

        module.register_buffer("full_attention_heads", layer_full_attention_heads)
        module.streaming_attn_func = torch.nn.functional.scaled_dot_product_attention
        module.full_attn_func = torch.nn.functional.scaled_dot_product_attention
        
        layer_number = idx + 1
        module.attention_softmax_scale = math.sqrt(module.hidden_size_per_attention_head)
        if getattr(module, 'apply_query_key_layer_scaling', False) or getattr(module.config, 'apply_query_key_layer_scaling', False):
            module.attention_softmax_scale *= layer_number


def get_chatglm_full_attention_heads(model):
    full_attention_heads = []
    for module in _iter_chatglm_attn_modules(model):
        if hasattr(module, "full_attention_heads"):
            full_attention_heads.append(module.full_attention_heads)
    return full_attention_heads


def set_chatglm_full_attention_heads(model, full_attention_heads):
    modules = list(_iter_chatglm_attn_modules(model))
    for idx, module in enumerate(modules):
        if hasattr(module, "full_attention_heads"):
            module.full_attention_heads.data = full_attention_heads[idx].to(
                module.full_attention_heads.device, module.full_attention_heads.dtype
            )
    return model


def map_chatglm_full_attention_heads(model, func):
    for module in _iter_chatglm_attn_modules(model):
        if hasattr(module, "full_attention_heads"):
            func(module.full_attention_heads)
