from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch

from benchmarks.common import method_defaults, parse_set
from loaders.chunked_post_ffn import enable_chunked_post_ffn


def _build_method_args(args) -> Namespace:
    defaults = method_defaults("quest")
    budget = args.budget if getattr(args, "budget", None) is not None else defaults.get("budget", 1024)
    method_args = Namespace(
        model=args.model,
        model_name=getattr(args, "model_name", args.model),
        budget=int(budget),
        token_budget=None,
        chunk_size=16,
        page_size=None,
        dtype="bfloat16",
        model_type="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        use_cache=True,
    )
    for key, value in defaults.items():
        if key in {"budget", "token_budget", "page_size"}:
            continue
        setattr(method_args, key, value)
    for key, value in parse_set(args.set, args.method).items():
        setattr(method_args, key, value)

    method_args.budget = int(method_args.budget)
    if method_args.budget <= 0:
        raise ValueError(f"Quest budget must be positive, got {method_args.budget}")
    method_args.token_budget = method_args.budget
    method_args.chunk_size = int(method_args.chunk_size)
    if method_args.chunk_size <= 0:
        raise ValueError(f"Quest chunk_size must be positive, got {method_args.chunk_size}")
    method_args.page_size = method_args.chunk_size
    for key in ("trust_remote_code", "low_cpu_mem_usage", "use_cache"):
        value = getattr(method_args, key)
        if isinstance(value, str):
            normalized = value.lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                value = True
            elif normalized in {"0", "false", "no", "n", "off"}:
                value = False
            else:
                raise ValueError(f"Unsupported boolean value: {value}")
        setattr(method_args, key, value)
    return method_args


def _load_efficiency_model(args, method_args, path):
    """Load Quest's paged-KV model used by its own efficiency benchmark."""
    if "llama" in method_args.model_name.lower():
        from quest.models.myllama import LlamaForCausalLM as QuestForCausalLM
    elif "qwen" in method_args.model_name.lower():
        from quest.models.qwen2 import Qwen2ForCausalLM as QuestForCausalLM
    else:
        raise ValueError(f"Quest efficiency backend does not support model '{args.model}'")
    device = torch.device(args.device)
    model = QuestForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device).eval()
    model.quest_init(
        page_size=method_args.page_size,
        max_seq_len=int(args.max_seq_length) + int(args.fixed_output_length) + 512,
        token_budget=method_args.token_budget,
        dtype=torch.float16,
        device=device,
    )
    return model


def load_model_and_tokenizer(args):
    """Load a Hugging Face model and attach the evaluation Quest attention patch."""
    quest_root = Path(__file__).resolve().parents[1] / "methods" / "quest"
    if str(quest_root) not in sys.path:
        sys.path.insert(0, str(quest_root))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    method_args = _build_method_args(args)
    path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    if getattr(args, "efficiency_backend", False):
        model = _load_efficiency_model(args, method_args, path)
        args._method_args = method_args
        args._quest_model_type = "efficiency"
        args._cleanup = None
        return model, tokenizer

    model_name = str(getattr(method_args, "model_name", args.model))
    model_name_lower = model_name.lower()
    attn_implementation = "eager" if "glm" in model_name_lower else method_args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    ).eval()
    
    if "llama" in method_args.model_name.lower():
        from evaluation.quest_attention import enable_quest_attention_eval
        print("llama model")
        model_type = "llama"
    elif "qwen" in method_args.model_name.lower():
        from evaluation.quest_qwen_attention import enable_quest_attention_eval
        print("qwen model")
        model_type = "qwen"
    elif "glm" in method_args.model_name.lower():
        from evaluation.quest_glm_attention import enable_quest_attention_eval
        print("glm model")
        model_type = "glm"
    else:
        raise ValueError(f"Quest does not support model_name '{method_args.model_name}'")

    enable_quest_attention_eval(model, method_args)
    enable_chunked_post_ffn(model)

    args._method_args = method_args
    args._quest_model_type = model_type
    args._cleanup = None
    return model, tokenizer
