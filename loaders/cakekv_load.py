from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch

from benchmarks.common import method_defaults, parse_set


def load_model_and_tokenizer(args):
    """Load a model and attach CakeKV KV-cache compression."""
    cakekv_root = Path(__file__).resolve().parents[1] / "methods" / "cakekv"
    # monkeypatch.py uses `from cakekv.cake.model.modify_llama import …`, so
    # the `methods/` directory must be on sys.path for `cakekv` to resolve as a
    # namespace package.
    methods_dir = cakekv_root.parent
    for p in (str(methods_dir), str(cakekv_root)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cakekv.cake.monkeypatch import (
        replace_flashllama_attn_with_cakeattn,
        replace_flashmistral_attn_with_cakeattn,
        replace_flashqwen2_attn_with_cakeattn,
        replace_chatglm_attn_with_cakeattn,
    )
    from cakekv.cake.utils import CompressConfig
    from cakekv.cake.cake_cache import CakeCache, CakeprefillKVCache

    method_args = Namespace(
        window_size=32,
    )
    for key, value in method_defaults("cakekv").items():
        if key != "budget":
            setattr(method_args, key, value)
    for key, value in parse_set(args.set, args.method).items():
        setattr(method_args, key, value)

    model_name = args.model
    path = args.model_path
    device = torch.device(args.device)
    is_chatglm = "glm" in model_name.lower() or "chatglm" in model_name.lower()

    if "llama" in model_name:
        replace_flashllama_attn_with_cakeattn()
    elif "mistral" in model_name:
        replace_flashmistral_attn_with_cakeattn()
    elif "qwen" in model_name:
        replace_flashqwen2_attn_with_cakeattn()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=is_chatglm)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=is_chatglm,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    if is_chatglm:
        replace_chatglm_attn_with_cakeattn(model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    window_size = method_args.window_size
    cache_size = args.budget
    key_size = cache_size - window_size

    layers = _get_model_layers(model)
    n_layers = len(layers)

    if not isinstance(window_size, list):
        window_size = [window_size] * n_layers
    if not isinstance(key_size, list):
        key_size = [key_size] * n_layers

    config = CompressConfig(compress=True, cascading=False, cache_size=cache_size, window_size=window_size[0], hyper=[1.0, 1.0, 200.0])
    model.config.compress_config = config
    model.config.window_size = window_size
    model.config.key_size = key_size
    model.config.prefill = [True] * n_layers
    model.config.decoding_evict = [None] * n_layers
    model.config.tau1 = config.hyper[0]
    model.config.tau2 = config.hyper[1]
    model.config.gamma = config.hyper[2]
    model.config.cake_cache = CakeCache

    first_attn = layers[0].self_attn
    model.config.prefill_cake_evict = [CakeprefillKVCache(
        cache_size=cache_size,
        window_size=window_size[0],
        k_seq_dim=2,
        v_seq_dim=2,
        num_heads=first_attn.num_heads,
        num_layers=n_layers,
        use_cascading=False,
    )] * n_layers

    model = model.to(device).eval()

    args._method_args = method_args
    args._cleanup = None
    return model, tokenizer


def _get_model_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "encoder") and hasattr(model.transformer.encoder, "layers"):
        return model.transformer.encoder.layers
    raise ValueError("Could not find layers in model")