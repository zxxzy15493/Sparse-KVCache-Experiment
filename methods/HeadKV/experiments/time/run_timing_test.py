"""
HeadKV timing test — 12-component CUDA Event measurement.

Protocol:  warmup + measure rounds, all written to report.
Usage:
    cd methods/HeadKV/experiments/time
    python run_timing_test.py --model llama3.1-8b-128k --method ReasonKV --max_capacity_prompts 1024
"""

import os
import sys
import time
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from TimeManager import time_manager
from models_patch import patch_model_with_timing

MODELS = {"llama3.1-8b-128k": "meta-llama/Llama-3.1-8B-Instruct"}
METHODS = ["AdativeKV", "ReasonKV"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3.1-8b-128k", choices=list(MODELS.keys()))
    parser.add_argument("--method", type=str, default="ReasonKV", choices=METHODS,
                        help="HeadKV method: AdativeKV / ReasonKV")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--input_max_token", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_capacity_prompts", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--measure", type=int, default=1)
    parser.add_argument("--text", type=str, default=None)
    return parser.parse_args()


def _format_per_round(r: dict) -> list:
    steps = r.get("decode_steps", 0)
    return [
        f"  Prefill total time (wall):  {r['prefill_latency']:.6f} s",
        f"    - Attention:           {r['prefill_attn']:.6f} s",
        f"    - FFN:                 {r['prefill_ffn']:.6f} s",
        f"    - mode part:             {r['prefill_pattern']:.6f} s",
        f"    - index build:             {r['prefill_idx']:.6f} s",
        f"    - Write Cache:         {r['prefill_write_cache']:.6f} s",
        f"  Decode average per step ({steps} steps, ms):",
        f"    - total time:              {r['decode_latency']:.3f} ms",
        f"    - Attention:           {r['decode_attn']:.3f} ms",
        f"    - FFN:                 {r['decode_ffn']:.3f} ms",
        f"    - retrieve:                {r['decode_retrieve']:.3f} ms",
        f"    - Write Cache:         {r['decode_write_cache']:.3f} ms",
    ]


def _build_report(rounds: list, warmup: int, method: str) -> str:
    sep = "=" * 60
    lines = [sep, f"  HeadKV ({method}) timing report", sep]
    for i, r in enumerate(rounds):
        tag = "(warmup)" if i < warmup else "(measure)"
        lines.append(f"  === Round {i + 1} {tag} ===")
        lines.extend(_format_per_round(r))
        lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}")
    model_path = MODELS[args.model]

    import transformers
    from headkv.adaptive_llama_hijack import (
        adaptive_llama_flash_attn2_forward,
        reason_llama_flash_attn2_forward,
        adaptive_LlamaModel_forward,
        prepare_inputs_for_generation_llama,
    )
    llama = transformers.models.llama.modeling_llama
    llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama
    if args.method == "ReasonKV":
        llama.LlamaFlashAttention2.forward = reason_llama_flash_attn2_forward
        llama.LlamaAttention.forward = reason_llama_flash_attn2_forward
        llama.LlamaSdpaAttention.forward = reason_llama_flash_attn2_forward
    else:
        llama.LlamaFlashAttention2.forward = adaptive_llama_flash_attn2_forward
        llama.LlamaAttention.forward = adaptive_llama_flash_attn2_forward
        llama.LlamaSdpaAttention.forward = adaptive_llama_flash_attn2_forward
    llama.LlamaModel.forward = adaptive_LlamaModel_forward
    print(f"Monkey-patched Llama with {args.method}.")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", trust_remote_code=True,
    ).to(device).eval()
    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True

    cfg = model.config
    cfg.window_size = 8
    cfg.kernel_size = 5
    cfg.pooling = "avgpool"
    cfg.base_capacity = args.max_capacity_prompts
    if args.method == "AdativeKV":
        cfg.floor = 0.25
        cfg.skip = False
        cfg.normalize = True
    else:
        cfg.head_choice = "reason"
        cfg.beta = 1.5
        cfg.temp = 1.0

    num_layers = len(model.model.layers)
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
        "input_ids": input_ids, "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens, "num_beams": 1, "do_sample": False,
        "use_cache": True, "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    time_manager.clear_rounds()
    all_rounds = []

    for w in range(args.warmup):
        time_manager.prefill_latency = 0.0
        time_manager.decode_latency = 0.0
        time_manager.decode_step = 0
        with torch.inference_mode():
            out = model.generate(**gen_kwargs)
        torch.cuda.synchronize()
        cake_timing = getattr(model.model, "_cake_timing", None)
        if cake_timing is not None:
            time_manager.prefill_latency = cake_timing.get("prefill_time", 0.0)
        L = time_manager.num_layers
        S = time_manager.decode_step
        total_event_s = time_manager._attn_start[0].elapsed_time(
            time_manager._attn_end[(S - 1) * L + (L - 1)]
        ) / 1000.0 if S > 0 else 0.0
        time_manager.decode_latency = max(total_event_s - time_manager.prefill_latency, 0.0)
        time_manager.finish_round()
        r = time_manager.get_last_round()
        all_rounds.append(r)
        print(f"  Warmup {w + 1}: prefill={r['prefill_latency']:.4f}s  "
              f"prefill_write_cache={r['prefill_write_cache']:.4f}s  "
              f"decode_avg={r['decode_latency']:.3f}ms  "
              f"retrieval={r['decode_retrieve']:.3f}ms  "
              f"decode_write_cache={r['decode_write_cache']:.3f}ms")
        torch.cuda.empty_cache()

    for m in range(args.measure):
        time_manager.prefill_latency = 0.0
        time_manager.decode_latency = 0.0
        time_manager.decode_step = 0
        with torch.inference_mode():
            out = model.generate(**gen_kwargs)
        torch.cuda.synchronize()
        cake_timing = getattr(model.model, "_cake_timing", None)
        if cake_timing is not None:
            time_manager.prefill_latency = cake_timing.get("prefill_time", 0.0)
        L = time_manager.num_layers
        S = time_manager.decode_step
        total_event_s = time_manager._attn_start[0].elapsed_time(
            time_manager._attn_end[(S - 1) * L + (L - 1)]
        ) / 1000.0 if S > 0 else 0.0
        time_manager.decode_latency = max(total_event_s - time_manager.prefill_latency, 0.0)
        time_manager.finish_round()
        r = time_manager.get_last_round()
        all_rounds.append(r)
        print(f"  Measure {m + 1}: prefill={r['prefill_latency']:.4f}s  "
              f"prefill_write_cache={r['prefill_write_cache']:.4f}s  "
              f"decode_avg={r['decode_latency']:.3f}ms  "
              f"retrieval={r['decode_retrieve']:.3f}ms  "
              f"decode_write_cache={r['decode_write_cache']:.3f}ms")
        torch.cuda.empty_cache()

    report_str = _build_report(all_rounds, args.warmup, args.method)
    print(report_str)
    report_path = f"timing_{args.model}_{args.method}_{args.input_max_token}_{args.max_new_tokens}_report.txt"
    with open(report_path, "w") as f:
        f.write(report_str + "\n")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()