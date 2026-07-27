from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch

from benchmarks.common import method_defaults, parse_set


def load_model_and_tokenizer(args):
    """Load a model and attach SnapKV KV-cache eviction."""
    snapkv_root = Path(__file__).resolve().parents[1] / "methods" / "SnapKV" / "experiment"
    if str(snapkv_root) not in sys.path:
        sys.path.insert(0, str(snapkv_root))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from snapkv.monkeypatch.monkeypatch import replace_llama, replace_qwen, replace_glm

    method_args = Namespace(
        max_capacity_prompt=1024,
        window_size=32,
        kernel_size=7,
        pooling="maxpool",
    )
    for key, value in method_defaults("snapkv").items():
        if key != "budget":
            setattr(method_args, key, value)
    for key, value in parse_set(args.set, args.method).items():
        setattr(method_args, key, value)
    if args.budget is not None:
        method_args.max_capacity_prompt = args.budget

    model_name = args.model
    path = args.model_path
    device = torch.device(args.device)

    replace_llama()
    replace_qwen()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    if "glm" in model_name.lower():
        replace_glm(model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    model = model.to(device).eval()

    layers = _get_model_layers(model)
    n_layers = len(layers)
    window_sizes = method_args.window_size
    max_caps = method_args.max_capacity_prompt
    kernel_sizes = method_args.kernel_size
    pooling = method_args.pooling

    if not isinstance(window_sizes, list):
        window_sizes = [window_sizes] * n_layers
    if not isinstance(max_caps, list):
        max_caps = [max_caps] * n_layers
    if not isinstance(kernel_sizes, list):
        kernel_sizes = [kernel_sizes] * n_layers

    for i in range(n_layers):
        attn = _get_layer_attn(layers[i])
        attn.config.window_size = window_sizes[i]
        attn.config.max_capacity_prompt = max_caps[i]
        attn.config.kernel_size = kernel_sizes[i]
        attn.config.pooling = pooling

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
