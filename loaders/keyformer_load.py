from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch

from benchmarks.common import method_defaults, parse_set


def load_model_and_tokenizer(args):
    """Load a model and attach KeyFormer KV-cache eviction."""
    kf_root = Path(__file__).resolve().parents[1] / "methods" / "keyformer" / "experiment"
    if str(kf_root) not in sys.path:
        sys.path.insert(0, str(kf_root))

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from keyformer_kv.enable_keyformer import enable_keyformer

    method_args = Namespace(
        recent_size=32,
        key_size=992,
        tau_init=1.0,
        tau_delta=0.01,
    )
    for key, value in method_defaults("keyformer").items():
        if key != "budget":
            setattr(method_args, key, value)
    for key, value in parse_set(args.set, args.method).items():
        setattr(method_args, key, value)
    if args.budget is not None:
        method_args.key_size = args.budget - method_args.recent_size

    path = args.model_path
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    if "glm" in path.lower():
        config_cls = get_class_from_dynamic_module(
            "configuration_chatglm.ChatGLMConfig", path
        )
        model_cls = get_class_from_dynamic_module(
            "modeling_chatglm.ChatGLMForConditionalGeneration", path
        )
        config = config_cls.from_pretrained(path, trust_remote_code=True)
        model = model_cls.from_pretrained(
            path,
            config=config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    else:
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
    enable_keyformer(model, method_args)

    args._method_args = method_args
    args._cleanup = None
    return model, tokenizer
