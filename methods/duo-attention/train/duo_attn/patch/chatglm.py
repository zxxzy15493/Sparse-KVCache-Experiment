from typing import Optional, Tuple

import torch
from torch import nn
import types

from flash_attn import flash_attn_func

from duo_attn.ulysses import UlyssesAttention


def _to_local_tensor(x: torch.Tensor) -> torch.Tensor:
    # FSDP may wrap parameters as DTensor; convert to local tensor before
    # mixing with flash-attn outputs that are plain torch.Tensor.
    if hasattr(x, "to_local"):
        return x.to_local()
    return x


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
            query_layer.size()[:-1]
            + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
        )
        key_layer = key_layer.view(
            key_layer.size()[:-1]
            + (
                self.num_multi_query_groups_per_partition,
                self.hidden_size_per_attention_head,
            )
        )
        value_layer = value_layer.view(
            value_layer.size()[:-1]
            + (
                self.num_multi_query_groups_per_partition,
                self.hidden_size_per_attention_head,
            )
        )
    else:
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
        query_layer, key_layer, value_layer = torch.split(
            mixed_x_layer, self.hidden_size_per_attention_head, dim=-1
        )

    query_layer, key_layer, value_layer = [
        x.transpose(1, 2) for x in (query_layer, key_layer, value_layer)
    ]
    return query_layer, key_layer, value_layer


def _expand_multi_query(self, key_layer, value_layer):
    if self.multi_query_attention:
        key_layer = key_layer.unsqueeze(2)
        key_layer = key_layer.expand(
            -1,
            -1,
            self.num_attention_heads_per_partition
            // self.num_multi_query_groups_per_partition,
            -1,
            -1,
        )
        key_layer = key_layer.contiguous().view(
            key_layer.size()[:1]
            + (self.num_attention_heads_per_partition,)
            + key_layer.size()[3:]
        )

        value_layer = value_layer.unsqueeze(2)
        value_layer = value_layer.expand(
            -1,
            -1,
            self.num_attention_heads_per_partition
            // self.num_multi_query_groups_per_partition,
            -1,
            -1,
        )
        value_layer = value_layer.contiguous().view(
            value_layer.size()[:1]
            + (self.num_attention_heads_per_partition,)
            + value_layer.size()[3:]
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
    query_layer, key_layer, value_layer = _project_qkv(self, hidden_states)

    if rotary_pos_emb is not None:
        query_layer = _apply_rotary_pos_emb(query_layer, rotary_pos_emb)
        key_layer = _apply_rotary_pos_emb(key_layer, rotary_pos_emb)

    if kv_cache is not None:
        cache_k, cache_v = kv_cache
        key_layer = torch.cat((cache_k, key_layer), dim=2)
        value_layer = torch.cat((cache_v, value_layer), dim=2)

    if use_cache:
        next_kv_cache = (key_layer, value_layer)
    else:
        next_kv_cache = None

    full_key_layer, full_value_layer = _expand_multi_query(self, key_layer, value_layer)
    streaming_key_layer, streaming_value_layer = _streaming_kv(
        key_layer,
        value_layer,
        self.sink_size,
        self.recent_size,
    )
    streaming_key_layer, streaming_value_layer = _expand_multi_query(
        self, streaming_key_layer, streaming_value_layer
    )

    full_attn_output = self.full_attn_func(
        query_layer.transpose(1, 2),
        full_key_layer.transpose(1, 2),
        full_value_layer.transpose(1, 2),
        causal=True,
        dropout_p=0.0,
    )

    streaming_attn_output = self.streaming_attn_func(
        query_layer.transpose(1, 2),
        streaming_key_layer.transpose(1, 2),
        streaming_value_layer.transpose(1, 2),
        causal=True,
        dropout_p=0.0,
    )

    full_attn_head_weight = _to_local_tensor(self.full_attention_heads.clamp(0, 1))
    full_query_head_weight = torch.repeat_interleave(
        full_attn_head_weight, self.num_key_value_groups
    )
    full_query_head_weight = full_query_head_weight.view(1, 1, -1, 1).to(
        full_attn_output.dtype
    )

    attn_output = (
        (1 - full_query_head_weight) * streaming_attn_output
        + full_query_head_weight * full_attn_output
    )
    attn_output = attn_output.reshape(attn_output.size(0), attn_output.size(1), -1).contiguous()
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

    # project qkv for both full and streaming paths
    full_q, full_k, full_v = _project_qkv(self, full_hidden_states)
    stream_q, stream_k, stream_v = _project_qkv(self, streaming_hidden_states)

    if rotary_pos_emb is not None:
        full_q = _apply_rotary_pos_emb(full_q, rotary_pos_emb)
        full_k = _apply_rotary_pos_emb(full_k, rotary_pos_emb)
        stream_q = _apply_rotary_pos_emb(stream_q, rotary_pos_emb)
        stream_k = _apply_rotary_pos_emb(stream_k, rotary_pos_emb)

    # expand full branch K/V
    full_k_exp, full_v_exp = _expand_multi_query(self, full_k, full_v)

    # truncate streaming branch before expansion so the window logic matches KV heads
    stream_k_trunc, stream_v_trunc = _streaming_kv(
        stream_k, stream_v, self.sink_size, self.recent_size
    )
    stream_k_exp, stream_v_exp = _expand_multi_query(self, stream_k_trunc, stream_v_trunc)
    # compute full and streaming attention outputs
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

    full_query_head_weight = torch.repeat_interleave(
        _to_local_tensor(self.full_attention_heads.clamp(0, 1)),
        self.num_key_value_groups,
    )
    full_query_head_weight = full_query_head_weight.view(1, 1, -1, 1).to(
        streaming_attn_output.dtype
    )
    mixed_stream = (
        (1 - full_query_head_weight) * streaming_attn_output
        + full_query_head_weight * full_attn_output
    )

    with torch.no_grad():
        full_output = self.dense(
            full_attn_output.reshape(
                full_attn_output.size(0), full_attn_output.size(1), -1
            ).contiguous()
        )
    streaming_output = self.dense(
        mixed_stream.reshape(mixed_stream.size(0), mixed_stream.size(1), -1).contiguous()
    )

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
            nn.Parameter(
                torch.ones(
                    kv_heads,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
                * initial_value
            ),
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
        module.num_key_value_groups = module.num_attention_heads_per_partition // kv_heads

        layer_full_attention_heads = torch.tensor(
            full_attention_heads[idx], device=device, dtype=dtype
        )
        layer_full_attention_heads = (layer_full_attention_heads > 0.5).to(dtype)
        module.register_buffer("full_attention_heads", layer_full_attention_heads)
        module.streaming_attn_func = flash_attn_func
        module.full_attn_func = flash_attn_func


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
