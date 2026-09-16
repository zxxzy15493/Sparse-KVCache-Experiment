"""Executable PQCache smoke test for Transformers 4.57 Qwen3.

Run from the repository root with the PQCache package and lfucache build on
PYTHONPATH.  It intentionally uses a short prompt and small cache allocation.
"""

import os

import torch
from transformers import AutoConfig, AutoTokenizer

from vq_method.qwen3_patch import VQQwen3ForCausalLM
from vq_method.retrieval_based.pq_search import del_objects, initialize_objects


def main():
    model_path = os.environ.get("QWEN3_MODEL_PATH", "Qwen/Qwen3-8B")
    # PQCache's multi-process KMeans backend reads these settings from the
    # environment.  Keep them aligned with this test config and with the
    # checked-in LongBench clustering coefficient (64_64_4).
    os.environ.setdefault("SUBVEC", "2")
    os.environ.setdefault("SUBBITS", "6")
    os.environ.setdefault("MAX_CPU_IN_USE", "64")
    config = AutoConfig.from_pretrained(model_path)
    config.compressor = "pq_search"
    config.compress_ratio = 0.1
    config.fixbudget = True
    config.budget = 16
    config.important_ratio = 0.5
    config.recent_ratio = 0.5
    config.recent_size = 4
    config.sink_size = 2
    config.n_subvec_per_head = 2
    config.n_subbits = 6
    config.topr = 8
    config.gqa = True
    config.max_iter = 0
    config.pp_size = 1
    # The existing PQ worker samples initialization indices above a tiny
    # 512-token buffer even for a short prompt; 2048 remains a small smoke
    # allocation while satisfying that worker invariant.
    config.max_seq_len = 2048
    config.cache_block_size = 128
    config.global_cache_size = 64
    config.cache_topk = 8

    initialize_objects(config, model="qwen3-8b-smoke")
    try:
        model = VQQwen3ForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=torch.bfloat16
        ).to("cuda").eval()
        model.patch(config)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        inputs = tokenizer("PQCache test. " * 32, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=2, do_sample=False, use_cache=True)
        assert output.shape[1] > inputs["input_ids"].shape[1]
        assert model.model.layers[0].self_attn.fwd_cnt >= 2
        print("Qwen3 PQCache smoke test passed")
    finally:
        del_objects()


if __name__ == "__main__":
    main()
