from __future__ import annotations

import json
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
        threshold_path=None,
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


def _load_threshold_table(path: str | Path, *, num_layers: int, num_heads: int) -> torch.Tensor:
    threshold_path = Path(path).expanduser()
    if not threshold_path.is_file():
        raise FileNotFoundError(f"XAttention threshold table does not exist: {threshold_path}")
    if threshold_path.suffix.lower() != ".json":
        raise ValueError(
            f"XAttention threshold_path must point to a JSON file, got: {threshold_path}"
        )

    with threshold_path.open("r", encoding="utf-8") as file:
        values = json.load(file)
    threshold = torch.tensor(values, dtype=torch.float32)
    expected_shape = (num_layers, num_heads)
    if tuple(threshold.shape) != expected_shape:
        raise ValueError(
            f"XAttention threshold table must have shape {expected_shape}, "
            f"got {tuple(threshold.shape)} from {threshold_path}"
        )
    if not torch.isfinite(threshold).all():
        raise ValueError(f"XAttention threshold table contains non-finite values: {threshold_path}")
    if torch.any((threshold < 0) | (threshold > 1)):
        raise ValueError(f"XAttention threshold values must be in [0, 1]: {threshold_path}")
    return threshold


def load_model_and_tokenizer(args):
    """Load a model through the XAttention FastPrefill loaders."""
    xattn_root = Path(__file__).resolve().parents[1] / "methods" / "x-attention"
    if str(xattn_root) not in sys.path:
        sys.path.insert(0, str(xattn_root))

    method_args = _build_method_args(args)
    path = args.model_path

    if "qwen3" in method_args.model_name.lower():
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        from xattn.src.load_qwen import FastPrefillConfig, patch_model

        model = Qwen3ForCausalLM.from_pretrained(
            path,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        model.eval()
        threshold_path = method_args.threshold_path
        if threshold_path is not None and method_args.threshold is not None:
            raise ValueError("Set only one of xattention.threshold and xattention.threshold_path")
        threshold = method_args.threshold
        if threshold_path is not None:
            threshold = _load_threshold_table(
                threshold_path,
                num_layers=model.config.num_hidden_layers,
                num_heads=model.config.num_attention_heads,
            )
        qwen3_config = FastPrefillConfig(
            threshold=threshold,
            print_detail=method_args.print_detail,
            stride=method_args.stride,
            metric=method_args.metric,
            p=method_args.p,
            model_type=model.config.model_type,
        )
        model = patch_model(model, qwen3_config, skip_sliding_window=True)
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
