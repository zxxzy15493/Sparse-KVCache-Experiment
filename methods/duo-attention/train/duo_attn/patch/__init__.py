# tensor_parallel internally uses pkg_resources (removed in setuptools>=70),
# shim it with importlib.metadata for forward compatibility.
import sys as _sys
import importlib.metadata as _metadata
class _PkgResourcesShim:
    @staticmethod
    def get_distribution(name):
        ver = _metadata.version(name)
        return type("Distribution", (), {"version": ver})()
_sys.modules.setdefault("pkg_resources", _PkgResourcesShim)

from importlib import import_module
import numpy as np
import os
import torch


def _get_model_patch(model):
    model_type = model.config.model_type
    if model_type == "qwen3":
        family = "qwen3"
    elif "llama" in model_type:
        family = "llama"
    elif "qwen2" in model_type:
        family = "qwen2"
    elif "mistral" in model_type or "mixtral" in model_type:
        family = "mistral"
    elif "chatglm" in model_type or "glm" in model_type:
        family = "chatglm"
    else:
        raise ValueError(f"Model type {model_type} not supported")
    # Older patches depend on removed Transformers exports. Import only the
    # selected model's patch, so they do not prevent Qwen3 training from starting.
    return import_module(f".{family}", __name__), family


def __getattr__(name):
    # Preserve access to the previously re-exported model-specific helpers.
    for family in ("llama", "qwen2", "qwen3", "mistral", "chatglm"):
        if name in {
            f"enable_{family}_duo_attention_training",
            f"enable_{family}_duo_attention_eval",
            f"get_{family}_full_attention_heads",
            f"set_{family}_full_attention_heads",
            f"map_{family}_full_attention_heads",
        }:
            return getattr(import_module(f".{family}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def enable_duo_attention_training(
    model,
    sink_size,
    recent_size,
    max_length,
    initial_value=1.0,
    enable_ulysses_attention=False,
    streaming_attn_implementation="blocksparse",
    ffn_chunk_size=0,
):
    print(
        f"Enabling DuoAttention training using {streaming_attn_implementation} imlementation"
    )
    patch, family = _get_model_patch(model)
    extra_kwargs = {}
    if family == "qwen3":
        extra_kwargs["ffn_chunk_size"] = ffn_chunk_size
    elif ffn_chunk_size:
        raise ValueError("FFN chunking is currently supported only for Qwen3")
    return getattr(patch, f"enable_{family}_duo_attention_training")(
        model, sink_size, recent_size, max_length,
        initial_value=initial_value,
        enable_ulysses_attention=enable_ulysses_attention,
        streaming_attn_implementation=streaming_attn_implementation,
        **extra_kwargs,
    )


def enable_duo_attention_eval(
    model,
    full_attention_heads,
    sink_size,
    recent_size,
):
    print(
        f"Enabling DuoAttention evaluation using sink size {sink_size} and recent size {recent_size}"
    )
    patch, family = _get_model_patch(model)
    enable_eval = getattr(patch, f"enable_{family}_duo_attention_eval", None)
    if enable_eval is None:
        raise NotImplementedError(f"The {family} patch currently supports training only")
    return enable_eval(model, full_attention_heads, sink_size, recent_size)


def get_full_attention_heads(model):
    patch, family = _get_model_patch(model)
    return getattr(patch, f"get_{family}_full_attention_heads")(model)


def set_full_attention_heads(model, full_attention_heads):
    patch, family = _get_model_patch(model)
    getattr(patch, f"set_{family}_full_attention_heads")(model, full_attention_heads)
    return model


def map_full_attention_heads(model, func):
    patch, family = _get_model_patch(model)
    return getattr(patch, f"map_{family}_full_attention_heads")(model, func)


def load_full_attention_heads(load_dir, filename="full_attention_heads.tsv"):
    full_attention_heads = np.loadtxt(
        os.path.join(load_dir, filename),
        dtype=float,
        delimiter="\t",
    )
    full_attention_heads = np.clip(full_attention_heads, 0, 1)
    full_attention_heads = torch.tensor(full_attention_heads, dtype=torch.float32)
    return full_attention_heads
