from __future__ import annotations

import torch

from loaders.chunked_post_ffn import enable_chunked_post_ffn


def load_model_and_tokenizer(args):
    """通过 Transformers 原生模型加载完整注意力。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    model = model.to(torch.device(args.device)).eval()
    enable_chunked_post_ffn(model)
    args._cleanup = None
    return model, tokenizer
