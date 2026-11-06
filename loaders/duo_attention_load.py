from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch

from benchmarks.common import method_defaults, parse_set


def load_model_and_tokenizer(args):
    """Load a model and attach DuoAttention streaming + full-attention head selection."""
    duo_root = Path(__file__).resolve().parents[1] / "methods" / "duo-attention"
    if str(duo_root) not in sys.path:
        sys.path.insert(0, str(duo_root))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from duo_attn.patch import enable_duo_attention_eval, load_full_attention_heads
    from duo_attn.utils import sparsify_attention_heads

    method_args = Namespace(
        sink_size=64, recent_size=256,
    )
    for key, value in method_defaults("duo-attention").items():
        if key != "budget":
            setattr(method_args, key, value)
    for key, value in parse_set(args.set, args.method).items():
        setattr(method_args, key, value)

    # Map fixthreshold (framework naming) → sparsity (duo_attn naming)
    if hasattr(method_args, "fixthreshold"):
        method_args.sparsity = method_args.fixthreshold

    model_name = args.model
    # A head-selection table is architecture-specific.  Check it before
    # allocating a 32B model so unsupported Qwen3 variants fail clearly rather
    # than consuming GPU memory with an incompatible Qwen2.5 table.
    attn_patterns_dir = duo_root / "attn_patterns"
    mapping = {
        "llama-3.1-8b": "Meta-Llama-3.1-8B-Instruct",
        "qwen-2.5-7b": "Qwen2.5-7B-Instruct",
        "qwen-2.5-7b-1m": "Qwen2.5-7B-Instruct",
        "glm-4-9b-1m": "glm-4-9b-chat-1m",
        "ds-qwen-1.5b": "deepseek-r1-distill-qwen-1.5b",
        "qwen3-32b": "Qwen3-32B",
    }
    subdir = mapping.get(model_name)
    if subdir is None:
        raise ValueError(f"DuoAttention: no full_attention_heads mapping for model '{model_name}'")
    head_dir = attn_patterns_dir / subdir
    candidates = list(head_dir.glob("full_attention_heads*.tsv"))
    if not candidates:
        raise FileNotFoundError(f"DuoAttention: no full_attention_heads.tsv under {head_dir}")
    head_file = candidates[0]

    print(f"[DuoAttention] sparsity={getattr(method_args, 'sparsity', 'N/A')}  sink={method_args.sink_size}  recent={method_args.recent_size}")

    path = args.model_path
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # In Transformers 4.57 rotary embeddings moved from attention modules to
    # the parent model. DuoAttention's legacy forward still accesses this field.
    base_model = getattr(model, "model", None)
    model_rotary_emb = getattr(base_model, "rotary_emb", None)
    # Transformers 4.57 keeps attention head counts on config rather than the
    # attention module; DuoAttention's kernels still read the module aliases.
    for module in model.modules():
        if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
            if not hasattr(module, "rotary_emb") and model_rotary_emb is not None:
                module.rotary_emb = model_rotary_emb
            module.num_heads = getattr(module, "num_heads", model.config.num_attention_heads)
            module.num_key_value_heads = getattr(module, "num_key_value_heads", getattr(model.config, "num_key_value_heads", model.config.num_attention_heads))
            module.num_key_value_groups = getattr(module, "num_key_value_groups", module.num_heads // module.num_key_value_heads)
            module.hidden_size = getattr(module, "hidden_size", model.config.hidden_size)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    full_attention_heads = load_full_attention_heads(str(head_dir), head_file.name)

    # DuoAttention's own sparsity semantics: threshold heads at the sparsity
    # quantile (small values -> streaming, large -> full). Every official eval
    # entrypoint applies this before enabling the patch, so mirror it here
    # rather than changing the patch internals.
    sparsity = getattr(method_args, "sparsity", None)
    if sparsity is not None:
        heads_np = full_attention_heads.cpu().numpy()
        heads_np, true_sparsity = sparsify_attention_heads(
            heads_np, None, sparsity=sparsity
        )
        full_attention_heads = torch.tensor(heads_np, dtype=torch.float32)
        print(f"[DuoAttention] true sparsity: {true_sparsity}")

    enable_duo_attention_eval(
        model,
        full_attention_heads=full_attention_heads,
        sink_size=method_args.sink_size,
        recent_size=method_args.recent_size,
    )

    model = model.eval()

    args._method_args = method_args
    args._cleanup = None
    return model, tokenizer
