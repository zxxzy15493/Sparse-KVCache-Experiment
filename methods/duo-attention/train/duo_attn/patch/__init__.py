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

from .llama import (
    enable_llama_duo_attention_training,
    enable_llama_duo_attention_eval,
    get_llama_full_attention_heads,
    set_llama_full_attention_heads,
    map_llama_full_attention_heads,
)

from .qwen2 import (
    enable_qwen2_duo_attention_training,
    enable_qwen2_duo_attention_eval,
    get_qwen2_full_attention_heads,
    set_qwen2_full_attention_heads,
    map_qwen2_full_attention_heads,
)

from .mistral import (
    enable_mistral_duo_attention_training,
    enable_mistral_duo_attention_eval,
    get_mistral_full_attention_heads,
    set_mistral_full_attention_heads,
    map_mistral_full_attention_heads,
)

from .chatglm import (
    enable_chatglm_duo_attention_training,
    enable_chatglm_duo_attention_eval,
    get_chatglm_full_attention_heads,
    set_chatglm_full_attention_heads,
    map_chatglm_full_attention_heads,
)

import numpy as np
import os
import torch


def enable_duo_attention_training(
    model,
    sink_size,
    recent_size,
    max_length,
    initial_value=1.0,
    enable_ulysses_attention=False,
    streaming_attn_implementation="blocksparse",
):
    print(
        f"Enabling DuoAttention training using {streaming_attn_implementation} imlementation"
    )
    if "llama" in model.config.model_type:
        enable_llama_duo_attention_training(
            model,
            sink_size,
            recent_size,
            max_length,
            initial_value=initial_value,
            enable_ulysses_attention=enable_ulysses_attention,
            streaming_attn_implementation=streaming_attn_implementation,
        )
    elif "qwen2" in model.config.model_type:
        enable_qwen2_duo_attention_training(
            model,
            sink_size,
            recent_size,
            max_length,
            initial_value=initial_value,
            enable_ulysses_attention=enable_ulysses_attention,
            streaming_attn_implementation=streaming_attn_implementation,
        )
    elif "mistral" in model.config.model_type or "mixtral" in model.config.model_type:
        enable_mistral_duo_attention_training(
            model,
            sink_size,
            recent_size,
            max_length,
            initial_value=initial_value,
            enable_ulysses_attention=enable_ulysses_attention,
            streaming_attn_implementation=streaming_attn_implementation,
        )
    elif "chatglm" in model.config.model_type or "glm" in model.config.model_type:
        enable_chatglm_duo_attention_training(
            model,
            sink_size,
            recent_size,
            max_length,
            initial_value=initial_value,
            enable_ulysses_attention=enable_ulysses_attention,
            streaming_attn_implementation=streaming_attn_implementation,
        )
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def enable_duo_attention_eval(
    model,
    full_attention_heads,
    sink_size,
    recent_size,
):
    print(
        f"Enabling DuoAttention evaluation using sink size {sink_size} and recent size {recent_size}"
    )
    if "llama" in model.config.model_type:
        enable_llama_duo_attention_eval(
            model,
            full_attention_heads,
            sink_size,
            recent_size,
        )
    elif "qwen2" in model.config.model_type:
        enable_qwen2_duo_attention_eval(
            model,
            full_attention_heads,
            sink_size,
            recent_size,
        )
    elif "mistral" in model.config.model_type or "mixtral" in model.config.model_type:
        enable_mistral_duo_attention_eval(
            model,
            full_attention_heads,
            sink_size,
            recent_size,
        )
    elif "chatglm" in model.config.model_type or "glm" in model.config.model_type:
        enable_chatglm_duo_attention_eval(
            model,
            full_attention_heads,
            sink_size,
            recent_size,
        )
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def get_full_attention_heads(model):
    if "llama" in model.config.model_type:
        return get_llama_full_attention_heads(model)
    elif "qwen2" in model.config.model_type:
        return get_qwen2_full_attention_heads(model)
    elif "mistral" in model.config.model_type or "mixtral" in model.config.model_type:
        return get_mistral_full_attention_heads(model)
    elif "chatglm" in model.config.model_type or "glm" in model.config.model_type:
        return get_chatglm_full_attention_heads(model)
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def set_full_attention_heads(model, full_attention_heads):
    if "llama" in model.config.model_type:
        model = set_llama_full_attention_heads(model, full_attention_heads)
    elif "qwen2" in model.config.model_type:
        model = set_qwen2_full_attention_heads(model, full_attention_heads)
    elif "mistral" in model.config.model_type or "mixtral" in model.config.model_type:
        model = set_mistral_full_attention_heads(model, full_attention_heads)
    elif "chatglm" in model.config.model_type or "glm" in model.config.model_type:
        model = set_chatglm_full_attention_heads(model, full_attention_heads)
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")
    return model


def map_full_attention_heads(model, func):
    if "llama" in model.config.model_type:
        return map_llama_full_attention_heads(model, func)
    elif "qwen2" in model.config.model_type:
        return map_qwen2_full_attention_heads(model, func)
    elif "mistral" in model.config.model_type or "mixtral" in model.config.model_type:
        return map_mistral_full_attention_heads(model, func)
    elif "chatglm" in model.config.model_type or "glm" in model.config.model_type:
        return map_chatglm_full_attention_heads(model, func)
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def load_full_attention_heads(load_dir, filename="full_attention_heads.tsv"):
    full_attention_heads = np.loadtxt(
        os.path.join(load_dir, filename),
        dtype=float,
        delimiter="\t",
    )
    full_attention_heads = np.clip(full_attention_heads, 0, 1)
    full_attention_heads = torch.tensor(full_attention_heads, dtype=torch.float32)
    return full_attention_heads
