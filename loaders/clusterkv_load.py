from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch

from benchmarks.common import method_defaults, parse_set


def load_model_and_tokenizer(args):
    """Load a model and attach ClusterKV attention without using pred.py's loader."""
    cluster_root = Path(__file__).resolve().parents[1] / "methods" / "ClusterKV"
    if str(cluster_root) not in sys.path:
        sys.path.insert(0, str(cluster_root))

    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
    import accuracy.cluster_attention as cluster_attention
    import accuracy.patch as cluster_patch

    cluster_args = Namespace(
        model=args.model, token_budget=args.budget, chunk_size=16,
        quest=False, sink=16, cluster=True, head_sel="truc", balance=False,
        nlist=400, fit_iter=20, gqa_policy=None, dist_t="cosine", cache_steps=0,
        topk_stat=False,
    )
    for key, value in method_defaults("clusterkv").items():
        if key != "budget":
            setattr(cluster_args, key, value)
    for key, value in parse_set(args.set, args.method).items():
        setattr(cluster_args, key, value)

    model_name = cluster_args.model
    path = args.model_path
    device = torch.device(args.device)
    if "qwen" in model_name or "glm4" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            use_cache=True,
        ).to(device)
    elif "llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            use_cache=True,
        )
    else:
        raise ValueError(f"ClusterKV does not support model '{args.model}'")

    model = model.eval()
    # The original patch uses a module-global counter when assigning layer ids.
    # Reset it for each unified model load so multiple requested models are valid.
    cluster_patch.layer_id = getattr(model.config, "num_hidden_layers", 32)
    cluster_patch.enable_attention_eval(model_name, model, cluster_args)

    args._method_args = cluster_args
    args._base_nlist = cluster_args.nlist
    args._cluster_attention = cluster_attention
    args._cleanup = None
    return model, tokenizer
