from __future__ import annotations

import sys
from importlib import import_module
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
        kernel_backend="auto",
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

    if method_args.kernel_backend not in ("auto", "single", "batch"):
        raise ValueError("Quest kernel_backend must be auto, single or batch; "
                         f"got {method_args.kernel_backend!r}")

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


def _resolve_kernel_backend(kernel_backend, batch_size):
    """Resolve an explicit backend without silently falling back to another."""
    if kernel_backend not in ("auto", "single", "batch"):
        raise ValueError("Quest kernel_backend must be auto, single or batch; "
                         f"got {kernel_backend!r}")
    if batch_size < 1:
        raise ValueError(f"Quest batch_size must be positive, got {batch_size}")
    if kernel_backend == "auto":
        return "single" if batch_size == 1 else "batch"
    if kernel_backend == "single" and batch_size != 1:
        raise ValueError("Quest kernel_backend=single requires batch_size=1; "
                         f"got {batch_size}. Select batch or auto for multiple requests.")
    return kernel_backend


def _load_efficiency_model(args, method_args, path):
    """Load the selected original single-request or native batch Quest backend."""
    from transformers import AutoConfig

    batch_size = int(getattr(args, "batch_size", 1))
    backend = _resolve_kernel_backend(method_args.kernel_backend, batch_size)
    config = AutoConfig.from_pretrained(path)
    if config.model_type != "llama":
        raise ValueError("Quest kernel efficiency backends currently support Llama only")
    if backend == "single":
        try:
            import_module("quest._kernels")
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "Quest selected kernel_backend=single, which requires a loadable quest._kernels "
                "extension. Compile/install it for the active environment, or explicitly set "
                "kernel_backend=batch."
            ) from error
        # The original RAFT TopK wrapper instantiates exactly 32 rows.
        if config.num_attention_heads != 32:
            raise ValueError("Original Quest TopK requires 32 query heads; "
                             "select kernel_backend=batch for other head counts")
        from quest.models.myllama import LlamaForCausalLM as QuestForCausalLM
    else:
        from quest.native_batch import NativeBatchLlamaForCausalLM as QuestForCausalLM

    device = torch.device(args.device)
    model = QuestForCausalLM.from_pretrained(
        path, config=config, dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation="eager", device_map=device,
    ).eval()
    init_kwargs = dict(
        page_size=method_args.page_size,
        max_seq_len=int(args.max_seq_length) + int(args.fixed_output_length),
        token_budget=method_args.token_budget,
    )
    if backend == "single":
        init_kwargs.update(dtype=torch.float16, device=device)
    else:
        init_kwargs["batch_size"] = batch_size
    model.quest_init(**init_kwargs)
    method_args.resolved_kernel_backend = backend
    return model


def load_model_and_tokenizer(args):
    """Load evaluation Quest, or a selected kernel backend with --efficiency-backend."""
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
        method_args.dtype = "float16"
        args._method_args = method_args
        args._quest_kernel_backend = method_args.resolved_kernel_backend
        args._quest_model_type = ("original_kernel" if args._quest_kernel_backend == "single"
                                  else "native_batch")
        args._cleanup = model.quest_clear
        print(f"Quest efficiency backend: requested={method_args.kernel_backend}, "
              f"selected={args._quest_kernel_backend}, batch={getattr(args, 'batch_size', 1)}, "
              f"budget={method_args.token_budget}, page_size={method_args.page_size}", flush=True)
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
