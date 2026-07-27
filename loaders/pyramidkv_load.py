from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch

from benchmarks.common import method_defaults, parse_set


def load_model_and_tokenizer(args):
    """Load a model and attach PyramidKV pyramidal KV-cache compression."""
    pyramidkv_root = Path(__file__).resolve().parents[1] / "methods" / "PyramidKV"
    if str(pyramidkv_root) not in sys.path:
        sys.path.insert(0, str(pyramidkv_root))

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from pyramidkv.monkeypatch import replace_llama, replace_qwen2, replace_chatglm

    method_args = Namespace(
        window_size=64, kernel_size=5, pooling="avgpool", pyram_beta=20,
    )
    for key, value in method_defaults("pyramidkv").items():
        if key != "budget":
            setattr(method_args, key, value)
    for key, value in parse_set(args.set, args.method).items():
        setattr(method_args, key, value)

    model_name = args.model
    path = args.model_path
    device = torch.device(args.device)

    model_name_lower = model_name.lower()
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    is_chatglm = ("chatglm" in getattr(config, "model_type", "").lower()) or ("glm" in model_name_lower)

    if "llama" in model_name_lower:
        replace_llama("pyramidkv")
    elif "qwen" in model_name_lower:
        replace_qwen2("pyramidkv")

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=is_chatglm)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=is_chatglm,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    if is_chatglm:
        replace_chatglm("pyramidkv", model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    model = model.to(device).eval()

    layers = _get_model_layers(model)
    n_layers = len(layers)
    window_sizes = method_args.window_size
    kernel_sizes = method_args.kernel_size
    base_capacity = args.budget

    if not isinstance(window_sizes, list):
        window_sizes = [window_sizes] * n_layers
    if not isinstance(kernel_sizes, list):
        kernel_sizes = [kernel_sizes] * n_layers

    for i in range(n_layers):
        attn = _get_layer_attn(layers[i])
        attn.config.window_size = window_sizes[i]
        attn.config.kernel_size = kernel_sizes[i]
        attn.config.pooling = method_args.pooling
        attn.config.base_capacity = base_capacity
        attn.config.pyram_beta = method_args.pyram_beta

    args._method_args = method_args
    args._cleanup = None
    return model, tokenizer


def _get_model_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "encoder") and hasattr(model.transformer.encoder, "layers"):
        return model.transformer.encoder.layers
    raise ValueError("Could not find layers in model")


def _get_layer_attn(layer):
    return getattr(layer, "self_attn", getattr(layer, "self_attention", None))