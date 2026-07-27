from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch

from benchmarks.common import method_defaults, parse_set


def load_model_and_tokenizer(args):
    """Load a model and attach H2O (Heavy-Hitter Oracle) KV-cache eviction."""
    h2o_root = Path(__file__).resolve().parents[1] / "methods" / "h2o" / "experiment"
    if str(h2o_root) not in sys.path:
        sys.path.insert(0, str(h2o_root))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from h2o_kv.enable_h2o import enable_h2o

    method_args = Namespace(
        heavy_hitter_size=992,
        recent_size=32,
    )
    for key, value in method_defaults("h2o").items():
        if key != "budget":
            setattr(method_args, key, value)
    for key, value in parse_set(args.set, args.method).items():
        setattr(method_args, key, value)
    if args.budget is not None:
        method_args.heavy_hitter_size = args.budget - method_args.recent_size

    path = args.model_path
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    model = model.to(device).eval()
    enable_h2o(model, method_args)

    args._method_args = method_args
    args._cleanup = None
    return model, tokenizer
