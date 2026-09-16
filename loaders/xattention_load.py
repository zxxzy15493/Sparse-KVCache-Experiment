from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch
from transformers import AutoTokenizer

from benchmarks.common import method_defaults, parse_set


def _build_method_args(args) -> Namespace:
    method_args = Namespace(
        model_name=getattr(args, "model_name", args.model),
        stride=16,
        threshold=None,
        print_detail=False,
        metric="xattn",
        p=0.9,
    )
    for key, value in method_defaults("xattention").items():
        if key == "budget":
            continue
        setattr(method_args, key, value)
    for key, value in parse_set(args.set, args.method).items():
        setattr(method_args, key, value)

    # Map the unified threshold override to XAttention's internal p parameter.
    fixthreshold = getattr(method_args, "fixthreshold", None)
    if fixthreshold is None:
        fixthreshold = getattr(args, "fixthreshold", None)
    if fixthreshold is not None:
        method_args.p = float(fixthreshold)

    method_args.stride = int(method_args.stride)
    if method_args.stride not in {4, 8, 16}:
        raise ValueError(f"XAttention stride must be one of 4, 8, 16, got {method_args.stride}")
    if method_args.threshold is not None:
        method_args.threshold = float(method_args.threshold)
    method_args.p = float(method_args.p)
    if method_args.p not in {0.8, 0.85, 0.9, 0.95}:
        raise ValueError(f"XAttention p must be one of 0.8, 0.85, 0.9, 0.95, got {method_args.p}")
    if method_args.metric != "xattn":
        raise ValueError(f"xattention loader only supports metric='xattn', got {method_args.metric!r}")
    return method_args


def load_model_and_tokenizer(args):
    """Load a model through the XAttention FastPrefill loaders."""
    xattn_root = Path(__file__).resolve().parents[1] / "methods" / "x-attention"
    if str(xattn_root) not in sys.path:
        sys.path.insert(0, str(xattn_root))

    method_args = _build_method_args(args)
    path = args.model_path

    if method_args.model_name.lower().startswith("qwen3"):
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        from xattn.src.load_qwen import FastPrefillConfig, forward_eval

        model = Qwen3ForCausalLM.from_pretrained(
            path,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        model.eval()
        # Qwen3 has 36 layers and a distinct Q/K normalization path. The
        # method's published Qwen2 table is 28x28, so use the documented 0.9
        # threshold as a scalar on global-attention layers. Preserve native
        # attention on Qwen3's sliding-window layers.
        threshold = method_args.threshold
        if threshold is None:
            threshold = method_args.p
        qwen3_config = FastPrefillConfig(
            threshold=threshold,
            print_detail=method_args.print_detail,
            stride=method_args.stride,
            metric=method_args.metric,
            p=method_args.p,
        )
        for layer in model.model.layers:
            attention = layer.self_attn
            if attention.sliding_window is not None:
                continue
            attention.num_heads = model.config.num_attention_heads
            attention.num_key_value_heads = model.config.num_key_value_heads
            attention.num_key_value_groups = model.config.num_attention_heads // model.config.num_key_value_heads
            attention.hidden_size = model.config.hidden_size
            attention.fastprefillconfig = qwen3_config
            attention.forward = forward_eval.__get__(attention, type(attention))
        tokenizer = AutoTokenizer.from_pretrained(path)
        model_type = "qwen3"
    elif "llama" in method_args.model_name.lower():
        from xattn.src.load_llama import FastPrefillConfig, load_model
        print("llama model")
        model_type = "llama"

        config = FastPrefillConfig(
            threshold=method_args.threshold,
            print_detail=method_args.print_detail,
            stride=method_args.stride,
            metric=method_args.metric,
            p=method_args.p,
        )
        model, tokenizer = load_model(config, name_or_path=path)
    elif "qwen" in method_args.model_name.lower():
        from xattn.src.load_qwen import FastPrefillConfig, load_model
        print("qwen model")
        model_type = "qwen"

        config = FastPrefillConfig(
            threshold=method_args.threshold,
            print_detail=method_args.print_detail,
            stride=method_args.stride,
            metric=method_args.metric,
            p=method_args.p,
        )
        model, tokenizer = load_model(config, name_or_path=path)
    elif "glm" in method_args.model_name.lower():
        from xattn.src.load_glm import FastPrefillConfig, load_model
        print("glm model")
        model_type = "glm"

        config = FastPrefillConfig(
            threshold=method_args.threshold,
            print_detail=method_args.print_detail,
            stride=method_args.stride,
            metric=method_args.metric,
            p=method_args.p,
        )
        model, tokenizer = load_model(config, name_or_path=path)
    else:
        raise ValueError(f"XAttention does not support model_name '{method_args.model_name}'")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    args._method_args = method_args
    args._xattention_model_type = model_type
    args._cleanup = None
    return model.eval(), tokenizer
