from __future__ import annotations
import sys
from argparse import Namespace
from pathlib import Path
import torch
from benchmarks.common import method_defaults, parse_set

def _build_method_args(args) -> Namespace:
    method_args = Namespace(
        budget=getattr(args, "budget", None),
        block_size=128,
        gamma=0.9,
        tau=0.1,
        min_budget=512,
        max_budget=None,
        dtype="bfloat16",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=True,
    )
    for key, value in method_defaults("flexprefill").items():
        if key == "budget":
            continue
        setattr(method_args, key, value)
    for key, value in parse_set(args.set, args.method).items():
        setattr(method_args, key, value)

    # Map the unified threshold override to FlexPrefill's internal gamma.
    fixthreshold = getattr(method_args, "fixthreshold", None)
    if fixthreshold is None:
        fixthreshold = getattr(args, "fixthreshold", None)
    if fixthreshold is not None:
        method_args.gamma = float(fixthreshold)

    for key in ("block_size", "min_budget"):
        parsed = int(getattr(method_args, key))
        if parsed <= 0:
            raise ValueError(f"FlexPrefill {key} must be positive, got {parsed}")
        setattr(method_args, key, parsed)
    method_args.gamma = float(method_args.gamma)
    method_args.tau = float(method_args.tau)
    if method_args.max_budget is not None:
        method_args.max_budget = int(method_args.max_budget)
        if method_args.max_budget <= 0:
            raise ValueError(f"FlexPrefill max_budget must be positive, got {method_args.max_budget}")
        if method_args.max_budget < method_args.min_budget:
            raise ValueError(
                f"FlexPrefill max_budget must be >= min_budget, got {method_args.max_budget} < {method_args.min_budget}"
            )
    return method_args

def load_model_and_tokenizer(args):
    """Load a model and attach FlexPrefill prefill attention."""
    flex_root = Path(__file__).resolve().parents[1] / "methods" / "flexPrefill"
    if str(flex_root) not in sys.path:
        sys.path.insert(0, str(flex_root))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from flex_prefill import disable_hf_flash_attention_check, patch_model

    method_args = _build_method_args(args)
    path = args.model_path
    device = torch.device(args.device)

    disable_hf_flash_attention_check()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    ).eval()

    flex_config = {
        "block_size": method_args.block_size,
        "flex_prefill_gamma": method_args.gamma,
        "flex_prefill_tau": method_args.tau,
        "flex_prefill_min_budget": method_args.min_budget,
        "flex_prefill_max_budget": method_args.max_budget,
    }
    patch_model(model, "flex_prefill", flex_config)

    args._method_args = method_args
    args._cleanup = None
    return model, tokenizer
