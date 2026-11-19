"""DuoAttention head-selection training for dense Qwen3 models.

The batch contains the full-attention teacher followed by the streaming student.
This patch is for training with unpadded sequences and no KV cache; deployment
and generation require a separate inference patch.
"""

import os
import types

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3Model,
    apply_rotary_pos_emb,
)

from duo_attn.ulysses import UlyssesAttention


def _streaming_attention_blocksparse(query, key, value, streaming_info):
    from block_sparse_attn import block_streaming_attn_func

    batch_size, seq_len, num_heads, head_dim = query.shape
    kv_heads = key.shape[2]
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, seq_len,
        device=query.device, dtype=torch.int32,
    )
    head_mask_type = torch.full(
        (num_heads,), -1, device=query.device, dtype=torch.int32,
    )
    output = block_streaming_attn_func(
        query.reshape(-1, num_heads, head_dim),
        key.reshape(-1, kv_heads, head_dim),
        value.reshape(-1, kv_heads, head_dim),
        cu_seqlens,
        cu_seqlens,
        head_mask_type,
        streaming_info,
        seq_len,
        seq_len,
        p_dropout=0.00,
        is_causal=True,
    )
    return output.reshape(batch_size, seq_len, num_heads, head_dim)


def _project_qkv(module, hidden_states, position_embeddings):
    batch_size, seq_len, _ = hidden_states.shape
    head_dim = module.head_dim
    query = module.q_norm(module.q_proj(hidden_states).view(
        batch_size, seq_len, module.config.num_attention_heads, head_dim,
    ))
    key = module.k_norm(module.k_proj(hidden_states).view(
        batch_size, seq_len, module.config.num_key_value_heads, head_dim,
    ))
    value = module.v_proj(hidden_states).view(
        batch_size, seq_len, module.config.num_key_value_heads, head_dim,
    )
    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=2)
    return query, key, value


def _mlp_chunk(module, hidden_states):
    return module.down_proj(
        module.act_fn(module.gate_proj(hidden_states)) * module.up_proj(hidden_states),
    )


def qwen3_duo_mlp_forward(self, hidden_states):
    batch_size = hidden_states.shape[0] // 2
    with torch.no_grad():
        teacher = torch.cat([
            _mlp_chunk(self, chunk)
            for chunk in hidden_states[:batch_size].split(self.duo_ffn_chunk_size, dim=1)
        ], dim=1)
    student_chunks = []
    for chunk in hidden_states[batch_size:].split(self.duo_ffn_chunk_size, dim=1):
        # A decoder checkpoint alone would recreate every chunk's FFN
        # intermediates together in backward. Checkpoint chunks separately.
        if torch.is_grad_enabled() and chunk.requires_grad:
            output = checkpoint(
                _mlp_chunk, self, chunk, use_reentrant=False, preserve_rng_state=False,
            )
        else:
            output = _mlp_chunk(self, chunk)
        student_chunks.append(output)
    return torch.cat((teacher, torch.cat(student_chunks, dim=1)), dim=0)


def qwen3_duo_attention_forward_two_way(
    self,
    hidden_states,
    position_embeddings=None,
    attention_mask=None,
    past_key_values=None,
    past_key_value=None,
    use_cache=False,
    **kwargs,
):
    if use_cache or past_key_values is not None or past_key_value is not None:
        raise ValueError("Qwen3 DuoAttention training does not support a KV cache")
    doubled_batch_size, seq_len, _ = hidden_states.shape
    if doubled_batch_size == 0 or doubled_batch_size % 2:
        raise ValueError("Expected a nonempty teacher/student batch with even size")
    if position_embeddings is None:
        raise ValueError("Qwen3 attention requires model-level position_embeddings")
    batch_size = doubled_batch_size // 2
    cos, sin = position_embeddings
    if cos.shape[0] == doubled_batch_size:
        full_rope = (cos[:batch_size], sin[:batch_size])
        streaming_rope = (cos[batch_size:], sin[batch_size:])
    else:
        full_rope = streaming_rope = position_embeddings

    with torch.no_grad():
        full_query, full_key, full_value = _project_qkv(
            self, hidden_states[:batch_size], full_rope,
        )
        full_output = self.full_attn_func(
            full_query, full_key, full_value,
            causal=True, dropout_p=0.00, softmax_scale=self.scaling,
        )
        del full_query, full_key, full_value

    query, key, value = _project_qkv(
        self, hidden_states[batch_size:], streaming_rope,
    )
    streaming_output = self.streaming_attn_func(
        query, key, value, self.streaming_mask,
    )
    # FSDP2 can expose a DTensor parameter while the kernels return local tensors.
    heads = self.full_attention_heads
    if hasattr(heads, "to_local"):
        heads = heads.to_local()
    heads = heads.clamp(0.00, 1.00).repeat_interleave(self.num_key_value_groups)
    heads = heads.view(1, 1, self.config.num_attention_heads, 1)
    streaming_output = (1.00 - heads) * streaming_output + heads * full_output

    # Qwen3-32B has 8192 attention features, but only 5120 hidden features.
    attention_size = self.config.num_attention_heads * self.head_dim
    with torch.no_grad():
        full_output = self.o_proj(full_output.reshape(batch_size, seq_len, attention_size))
    streaming_output = self.o_proj(
        streaming_output.reshape(batch_size, seq_len, attention_size),
    )
    return torch.cat((full_output, streaming_output), dim=0), None


def qwen3_duo_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
    **kwargs,
):
    """Use native Qwen3 layers/RoPE without constructing a dense causal mask."""
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Specify exactly one of input_ids or inputs_embeds")
    if use_cache or past_key_values is not None:
        raise ValueError("Qwen3 DuoAttention training requires use_cache=False")
    if output_attentions:
        raise ValueError("DuoAttention kernels do not return attention weights")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape != inputs_embeds.shape[:2]:
            raise ValueError("Expected an unpadded 2D attention_mask matching the input")
        if not bool((attention_mask == 1).all()):
            raise ValueError("Qwen3 DuoAttention training does not support padded sequences")
    seq_len = inputs_embeds.shape[1]
    if position_ids is None:
        if cache_position is None:
            cache_position = torch.arange(seq_len, device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
    elif position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0)

    output_hidden_states = (
        self.config.output_hidden_states
        if output_hidden_states is None else output_hidden_states
    )
    return_dict = self.config.use_return_dict if return_dict is None else return_dict
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    all_hidden_states = () if output_hidden_states else None
    for layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        layer_output = layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
            **kwargs,
        )
        # Transformers versions differ in whether the decoder returns a tuple.
        hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    if not return_dict:
        return (hidden_states, all_hidden_states) if output_hidden_states else (hidden_states,)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states, hidden_states=all_hidden_states,
    )


def _get_backbone(model):
    if isinstance(model, Qwen3ForCausalLM):
        return model.model
    if isinstance(model, Qwen3Model):
        return model
    raise ValueError("Expected a dense Qwen3ForCausalLM or Qwen3Model")


def enable_qwen3_duo_attention_training(
    model,
    sink_size,
    recent_size,
    max_length,
    initial_value=1.00,
    enable_ulysses_attention=False,
    streaming_attn_implementation="blocksparse",
    ffn_chunk_size=0,
):
    backbone = _get_backbone(model)
    if streaming_attn_implementation != "blocksparse":
        raise ValueError("Qwen3 DuoAttention training currently supports only 'blocksparse'")
    if sink_size < 0 or recent_size <= 0 or max_length <= 0:
        raise ValueError("sink_size must be nonnegative; recent_size and max_length must be positive")
    if ffn_chunk_size < 0:
        raise ValueError("ffn_chunk_size must be nonnegative")
    if not 0.00 <= initial_value <= 1.00:
        raise ValueError("initial_value must be between 0.00 and 1.00")
    if any(hasattr(layer.self_attn, "full_attention_heads") for layer in backbone.layers):
        raise ValueError("Qwen3 DuoAttention training is already enabled")
    if getattr(model.config, "use_sliding_window", False):
        raise ValueError("This training patch expects Qwen3 full-attention layers")

    # Fail before patching the model when an extension is missing or incompatible.
    from flash_attn import flash_attn_func
    from block_sparse_attn import block_streaming_attn_func  # noqa: F401

    world_size = 1
    if enable_ulysses_attention:
        world_size = (
            dist.get_world_size() if dist.is_initialized()
            else int(os.environ.get("WORLD_SIZE", "1"))
        )
    config = model.config
    if world_size < 1 or any(
        heads % world_size for heads in (config.num_attention_heads, config.num_key_value_heads)
    ):
        raise ValueError("Ulysses WORLD_SIZE must divide both query and KV head counts")
    num_sink_blocks = (sink_size + 127) // 128
    num_recent_blocks = (recent_size + 127) // 128
    num_heads_per_device = config.num_attention_heads // world_size
    print(
        f"Using Qwen3 blocksparse implementation with {num_sink_blocks} sink blocks, "
        f"{num_recent_blocks} recent blocks, and {num_heads_per_device} heads per device"
    )
    for layer in backbone.layers:
        if ffn_chunk_size:
            layer.mlp.duo_ffn_chunk_size = ffn_chunk_size
            layer.mlp.forward = types.MethodType(qwen3_duo_mlp_forward, layer.mlp)
        module = layer.self_attn
        module.sink_size = sink_size
        module.recent_size = recent_size
        module.register_parameter("full_attention_heads", nn.Parameter(torch.full(
            (config.num_key_value_heads,), initial_value,
            device=module.q_proj.weight.device, dtype=module.q_proj.weight.dtype,
        )))
        module.register_buffer("streaming_mask", torch.tensor(
            [num_sink_blocks, num_recent_blocks] * num_heads_per_device,
            device=module.q_proj.weight.device, dtype=torch.int32,
        ))
        # With one rank, avoiding all-to-all also allows direct non-torchrun use.
        if enable_ulysses_attention and world_size > 1:
            module.streaming_attn_func = UlyssesAttention(_streaming_attention_blocksparse)
            module.full_attn_func = UlyssesAttention(flash_attn_func)
        else:
            module.streaming_attn_func = _streaming_attention_blocksparse
            module.full_attn_func = flash_attn_func
        module.forward = types.MethodType(qwen3_duo_attention_forward_two_way, module)
    backbone.forward = types.MethodType(qwen3_duo_model_forward, backbone)
    model.config.use_cache = False


def get_qwen3_full_attention_heads(model):
    return [layer.self_attn.full_attention_heads for layer in _get_backbone(model).layers]


@torch.no_grad()
def set_qwen3_full_attention_heads(model, full_attention_heads):
    parameters = get_qwen3_full_attention_heads(model)
    if len(parameters) != len(full_attention_heads):
        raise ValueError("Expected one attention-head vector per Qwen3 layer")
    values = [
        torch.as_tensor(value, device=param.device, dtype=param.dtype)
        for param, value in zip(parameters, full_attention_heads)
    ]
    if any(value.shape != param.shape for param, value in zip(parameters, values)):
        raise ValueError("Attention-head vector shape does not match the model")
    for parameter, value in zip(parameters, values):
        if hasattr(parameter, "device_mesh"):
            from torch.distributed.tensor import distribute_tensor

            value = distribute_tensor(value, parameter.device_mesh, parameter.placements)
        parameter.copy_(value)
    return model


def map_qwen3_full_attention_heads(model, func):
    for heads in get_qwen3_full_attention_heads(model):
        func(heads)
