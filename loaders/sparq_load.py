from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch

from benchmarks.common import method_defaults, parse_set


def _positive_int(value: object, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"SparQ {name} must be positive, got {parsed}")
    return parsed


def _build_method_args(args) -> Namespace:
    defaults = method_defaults("sparq")
    budget = args.budget if getattr(args, "budget", None) is not None else defaults.get("budget", 1024)
    method_args = Namespace(
        budget=int(budget),
        k=None,
        local_k=32,
        rank=16,
        score="sparse_q",
        reallocate_to_mean_value=True,
        dtype="bfloat16",
        model_type="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=True,
    )
    for key, value in defaults.items():
        if key in {"budget", "k"}:
            continue
        setattr(method_args, key, value)
    for key in ("local_k", "rank"):
        value = getattr(args, key, None)
        if value is not None:
            setattr(method_args, key, value)
    for key, value in parse_set(args.set, args.method).items():
        setattr(method_args, key, value)

    method_args.budget = _positive_int(method_args.budget, "budget")
    method_args.k = method_args.budget
    method_args.local_k = _positive_int(method_args.local_k, "local_k")
    method_args.rank = _positive_int(method_args.rank, "rank")
    if method_args.score not in {"sparse_q", "low_rank"}:
        raise ValueError(f"SparQ score must be 'sparse_q' or 'low_rank', got {method_args.score!r}")
    if method_args.model_type not in {"auto", "llama", "qwen", "glm"}:
        raise ValueError(f"SparQ model_type must be auto/llama/qwen/glm, got {method_args.model_type!r}")
    return method_args


def load_model_and_tokenizer(args):
    """Load a model and attach SparQ ANN sparse attention."""
    sparq_root = Path(__file__).resolve().parents[1] / "methods" / "sparq"
    if str(sparq_root) not in sys.path:
        sys.path.insert(0, str(sparq_root))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llminference.myexperiments import Sparsity, SparsityMethods

    method_args = _build_method_args(args)
    path = args.model_path
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    sparq = Sparsity(
        name="ann",
        k=method_args.k,
        local_k=method_args.local_k,
        rank=method_args.rank,
        score=method_args.score,
        reallocate_to_mean_value=method_args.reallocate_to_mean_value,
    )
    model = SparsityMethods.apply(sparq, model).to(device).eval()

    args._method_args = method_args
    args._cleanup = None
    return model, tokenizer
