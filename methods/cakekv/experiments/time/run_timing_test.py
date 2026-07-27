"""
CakeKV timing test - CUDA Event measurement.

Measures:
  Prefill: lat(s) / pattern(s) / attn(s) / ffn(s) / index(s) / write_cache(s)
  Decode: lat(ms) / attn(ms) / ffn(ms) / retrieve(ms) / write_cache(ms)  (32-step average)

Protocol: 4 warmup rounds + 1 measure round, each round written to report.
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from cake.cake_cache import CakeprefillKVCache
from cake.utils import CompressConfig

from TimeManager import time_manager
from models_patch import patch_model_with_timing

MODEL2PATH = {
    "llama3.1-8b-128k": "meta-llama/Llama-3.1-8B-Instruct",
}

DECODE_STEPS = 32
GENERATION_NEW_TOKENS = DECODE_STEPS + 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3.1-8b-128k", choices=list(MODEL2PATH.keys()))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cache_size", type=int, default=1024)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--input_max_token", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--measure", type=int, default=1)
    parser.add_argument("--text", type=str, default=None)
    return parser.parse_args()


def _reset_cakekv_state(model, num_layers):
    attn_cfg = model.model.layers[0].self_attn.config
    attn_cfg.prefill = [True] * num_layers
    attn_cfg.decoding_evict = [None] * num_layers
    if hasattr(model.model, "_cake_timing"):
        model.model._cake_timing = None


def _run_one_round(model, gen_kwargs, inner_model):
    time_manager.prefill_latency = 0.0
    time_manager.decode_latency = 0.0
    time_manager.decode_steps = 0
    time_manager.decode_step = 0

    with torch.inference_mode():
        model.generate(**gen_kwargs)
    torch.cuda.synchronize()

    cake_timing = getattr(inner_model, "_cake_timing", None)
    if cake_timing is not None:
        time_manager.prefill_latency = cake_timing.get("prefill_time", 0.0)

    L = time_manager.num_layers
    S = time_manager.decode_step
    total_event_us = time_manager._attn_start[0].elapsed_time(
        time_manager._attn_end[(S - 1) * L + (L - 1)]
    ) if S > 0 else 0.0
    total_event = total_event_us / 1000.0
    time_manager.decode_latency = max(total_event - time_manager.prefill_latency, 0.0)

    time_manager.finish_round()


def _format_per_round(result: dict, idx: int) -> list:
    steps = result.get("decode_steps", 0)
    return [
        f"  --- Round {idx} ---",
        f"  Prefill total time (wall):  {result['prefill_latency']:.6f} s",
        f"    - Attention:           {result['prefill_attn']:.6f} s",
        f"    - FFN:                 {result['prefill_ffn']:.6f} s",
        f"    - mode assignment:             {result['prefill_pattern']:.6f} s",
		f"    - index build:             {result['prefill_idx']:.6f} s",
        f"    - Write Cache:         {result['prefill_write_cache']:.6f} s",
        f"  Decode average per step ({steps} steps, ms):",
		f"    - total time:              {result['decode_latency']:.3f} ms",
        f"    - Attention:           {result['decode_attn']:.3f} ms",
        f"    - FFN:                 {result['decode_ffn']:.3f} ms",
        f"    - retrieve:                {result['decode_retrieve']:.3f} ms",
        f"    - Write Cache:         {result['decode_write_cache']:.3f} ms",
    ]


def _build_report(rounds: list, warmup: int) -> str:
    sep = "=" * 60
    lines = [sep, "  CakeKV timing report", sep]
    for idx, result in enumerate(rounds):
        tag = "(warmup)" if idx < warmup else "(measure)"
        lines.append(f"  === Round {idx + 1} {tag} ===")
        lines.extend(_format_per_round(result, idx + 1))
        lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}")

    compress_config = CompressConfig(True, False)
    compress_config.cache_size = args.cache_size
    compress_config.window_size = args.window_size
    compress_config.hyper = [1.0, 1.0, 200.0]

    model_path = MODEL2PATH[args.model]
    print(f"Loading model: {model_path}")

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from cake.monkeypatch import replace_flashllama_attn_with_cakeattn
    replace_flashllama_attn_with_cakeattn()

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device).eval()
    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True

    num_layers = config.num_hidden_layers
    for i in range(num_layers):
        attn = model.model.layers[i].self_attn
        attn.config.key_size = [compress_config.cache_size - compress_config.window_size] * num_layers
        attn.config.window_size = [compress_config.window_size] * num_layers
        attn.config.prefill = [True] * num_layers
        attn.config.decoding_evict = [None] * num_layers
        attn.config.tau1 = compress_config.hyper[0]
        attn.config.tau2 = compress_config.hyper[1]
        attn.config.gamma = compress_config.hyper[2]
        attn.config.prefill_cake_evict = [CakeprefillKVCache(
            cache_size=compress_config.cache_size,
            window_size=compress_config.window_size,
            k_seq_dim=2,
            v_seq_dim=2,
            num_heads=attn.num_heads,
            num_layers=num_layers,
            use_cascading=compress_config.cascading,
        )] * num_layers

    model = patch_model_with_timing(model)
    print(f"Timing patch applied to {num_layers} layers.")

    if args.text is None:
        paragraph = (
            "The capital of France is Paris, a city known for its art, culture, and history. "
            "It is one of the most visited cities in the world. "
            "The Eiffel Tower, located in Paris, is a global cultural icon of France. "
        )
        repeat = (args.input_max_token * 2) // len(paragraph.split()) + 1
        args.text = paragraph * repeat

    inputs = tokenizer([args.text], return_tensors="pt", padding=True, truncation=True, max_length=args.input_max_token)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    print(f"Input tokens: {input_ids.shape[1]}")

    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": GENERATION_NEW_TOKENS,
        "num_beams": 1,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    inner_model = getattr(model, "model", None) or model

    time_manager.clear_rounds()
    all_rounds = []

    print(f"Warmup ({args.warmup} rounds)...")
    for w in range(args.warmup):
        _reset_cakekv_state(model, num_layers)
        _run_one_round(model, gen_kwargs, inner_model)
        result = time_manager.get_last_round()
        all_rounds.append(result)
        print(
            f"  Warmup {w + 1}: prefill={result['prefill_latency']:.4f}s  "
            f"prefill_write_cache={result['prefill_write_cache']:.4f}s  "
            f"decode_total={result['decode_latency']:.3f}ms  "
            f"retrieval={result['decode_retrieve']:.3f}ms  "
            f"write_cache={result['decode_write_cache']:.3f}ms"
        )
        torch.cuda.empty_cache()

    print(f"Measure ({args.measure} rounds)...")
    for m in range(args.measure):
        _reset_cakekv_state(model, num_layers)
        _run_one_round(model, gen_kwargs, inner_model)
        result = time_manager.get_last_round()
        all_rounds.append(result)
        print(
            f"  Measure {m + 1}: prefill={result['prefill_latency']:.4f}s  "
            f"prefill_write_cache={result['prefill_write_cache']:.4f}s  "
            f"decode_total={result['decode_latency']:.3f}ms  "
            f"retrieval={result['decode_retrieve']:.3f}ms  "
            f"write_cache={result['decode_write_cache']:.3f}ms"
        )
        torch.cuda.empty_cache()

    report_str = _build_report(all_rounds, args.warmup)
    print(report_str)

    report_path = f"timing_{args.model}_{args.input_max_token}_{DECODE_STEPS}_report.txt"
    with open(report_path, "w") as file:
        file.write(report_str + "\n")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
