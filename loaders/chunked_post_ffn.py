"""Chunked post-FFN runtime patch for methods that only patch attention."""

import types

import torch

CHUNK_SIZE = 1024


def chunked_post_ffn(hidden_states, layernorm, mlp, chunk_size=CHUNK_SIZE, seq_dim=-2):
    seq_len = hidden_states.shape[seq_dim]
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = hidden_states[start:end] if seq_dim == 0 else hidden_states[..., start:end, :]

        normalized = layernorm(chunk)
        mlp_output = mlp(normalized)

        chunk.add_(mlp_output)
    return hidden_states


def _wrap_layer(layer, chunk_size):
    # ChatGLM's GLMBlock keeps [seq, batch, hidden]; other decoder layers are batch-first.
    seq_dim = 0 if type(layer).__name__ == "GLMBlock" else -2
    if getattr(layer, "apply_residual_connection_post_layernorm", False):
        return
    layernorm = layer.post_attention_layernorm
    mlp = layer.mlp

    def forward(self, hidden_states, *args, **kwargs):
        if torch.is_grad_enabled():
            return original_forward(hidden_states, *args, **kwargs)

        # Run the original layer with a no-op post-attention norm/MLP so the
        # attention, KV cache and residual add keep the upstream behavior.
        # The no-op MLP returns zeros, so the layer output is the pre-FFN
        # residual; each chunk is then normalized, projected and added back.
        norm_forward = layernorm.forward
        mlp_forward = mlp.forward
        layernorm.forward = lambda x: x
        mlp.forward = lambda x: torch.zeros((), dtype=x.dtype, device=x.device)
        try:
            outputs = original_forward(hidden_states, *args, **kwargs)
        finally:
            layernorm.forward = norm_forward
            mlp.forward = mlp_forward

        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        chunked_post_ffn(hidden_states, layernorm, mlp, chunk_size, seq_dim)
        return outputs

    wrapped = types.MethodType(forward, layer)
    if hasattr(layer, "_old_forward"):
        # Accelerate hooks call _old_forward inside pre/post_forward, so the
        # chunked FFN runs while the weights are still on the device.
        original_forward = layer._old_forward
        layer._old_forward = wrapped
    else:
        original_forward = layer.forward
        layer.forward = wrapped


def enable_chunked_post_ffn(model, chunk_size=CHUNK_SIZE):
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        layers = model.transformer.encoder.layers

    for layer in layers:
        _wrap_layer(layer, chunk_size)
    return model
