from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path

import torch

from benchmarks.common import method_defaults, parse_set


def load_model_and_tokenizer(args):
    """Load PQCache's original (uncompressed) attention model directly."""
    pqcache_root = Path(__file__).resolve().parents[1] / "methods" / "pqcache"
    for path in (pqcache_root, pqcache_root / "vq_method" / "retrieval_based" / "lfu" / "build"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from transformers import AutoConfig, AutoTokenizer
    from vq_method.llama31_patch import VQLlama31ForCausalLM
    from vq_method.qwen25_patch import VQQwen2ForCausalLM

    # Keep the original PQCache argument names, while receiving their values from
    # the unified CLI and the method YAML rather than a method-specific parser.
    method_args = Namespace(
        compress_ratio=1.0, fixbudget=True, budget=args.budget, important_ratio=0.0,
        recent_ratio=1.0, enable_vq_cache=True, enable_h2o_cache=False, sink_size=16,
        recent_size=32, keyformer_mode=0, drop_ratio=0.0, preserve_layer=0,
        score_func="sum", compressor="original", threshold=1.0,
        n_subvec_per_head=2, n_subbits=6, topr=32, gqa="True",
        sparq_mean_v_trick="False", max_iter=0, fixthreshold=-1.0, pp_size=1,
    )
    for key, value in method_defaults("full_attention").items():
        if key != "budget":
            setattr(method_args, key, value)
    for key, value in parse_set(args.set, args.method).items():
        setattr(method_args, key, value)

    path = args.model_path
    device = torch.device(args.device)
    if args.model.startswith("glm-"):
        from vq_method.glm_patch import VQGlmForCausalLM

        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        config.attention_bias = config.add_qkv_bias
        config.attention_dropout = config.attention_dropout
        config.head_dim = config.kv_channels
        config.intermediate_size = config.ffn_hidden_size
        config.num_key_value_heads = config.multi_query_group_num
        config.num_hidden_layers = config.num_hidden_layers
        config.max_position_embeddings = config.seq_length
        config.rope_theta = float(config.rope_ratio)
        config.rms_norm_eps = config.layernorm_epsilon
        config.vocab_size = config.padded_vocab_size
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_class = VQGlmForCausalLM
        model_kwargs = {"trust_remote_code": True}
    elif args.model == "llama-3.1-8b":
        config = AutoConfig.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        model_class = VQLlama31ForCausalLM
        model_kwargs = {}
    elif args.model in {"qwen-2.5-7b", "qwen-2.5-7b-1m", "ds-qwen-1.5b"}:
        config = AutoConfig.from_pretrained(path, trust_remote_code=args.model == "ds-qwen-1.5b")
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=args.model == "ds-qwen-1.5b")
        model_class = VQQwen2ForCausalLM
        model_kwargs = {}
    else:
        raise ValueError(f"PQCache original attention does not support model '{args.model}'")

    config.compress_ratio = method_args.compress_ratio
    config.fixbudget = method_args.fixbudget
    config.budget = method_args.budget
    config.important_ratio = method_args.important_ratio
    config.pp_size = method_args.pp_size
    config.sink_size = method_args.sink_size
    config.recent_size = method_args.recent_size
    config.keyformer_mode = method_args.keyformer_mode == 1
    config.drop_ratio = method_args.drop_ratio
    config.preserve_layer = method_args.preserve_layer
    config.score_func = method_args.score_func
    config.compressor = method_args.compressor
    config.threshold = method_args.threshold
    config.n_subvec_per_head = method_args.n_subvec_per_head
    config.n_subbits = method_args.n_subbits
    config.fixthreshold = method_args.fixthreshold
    config.topr = method_args.topr
    config.gqa = method_args.gqa == "True"
    config.max_iter = method_args.max_iter
    config.device = device
    config.mean_v_trick = method_args.sparq_mean_v_trick == "True"
    config.recent_ratio = method_args.recent_ratio

    model = model_class.from_pretrained(path, torch_dtype=torch.bfloat16, config=config, **model_kwargs)
    model.patch(config)
    model = model.to(device).eval()
    args._method_args = method_args
    args._cleanup = None
    return model, tokenizer
