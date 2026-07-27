import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["SYNC_TEST_TIME"] = "1"
os.environ["PQCACHE_LAYER_ATTENTION_TIMING"] = "1"

import argparse
import csv
import time
from datetime import datetime

import numpy as np
import torch
from loguru import logger
from transformers import AutoConfig, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from vq_method.retrieval_based.pq_search import initialize_objects, del_objects, wait
from vq_method.retrieval_based.global_timer import (
    init_timer, init_all_stages, set_recording_state, reset_timer,
    get_all_stages_time, print_all_stages_time,
)
from vq_method.llama31_patch import VQLlama31ForCausalLM
from vq_method.qwen25_patch import VQQwen2ForCausalLM


PQCACHE_REQUIRED_STAGES = (
    "prefill_index_build",
    "decode_retrieve",
    "decode_load",
)


def parse_int_list(values):
    """Accept either: --input-lens 4096 8192 or --input-lens 4096,8192."""
    out = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Breakdown latency test for PQCache")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="./myinput.txt")
    parser.add_argument("--input-lens", type=str, nargs="+", required=True,
                        help="Input lengths, e.g. 4096 8192 or 4096,8192")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--budget", type=int, default=1024)

    parser.add_argument("--csv", type=str, default="./breakdown_test/log/breakdown_results.csv")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--measure-rounds", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=1.0)

    args = parser.parse_args()
    args.input_lens = parse_int_list(args.input_lens)
    return args


def setup_config(args):
    config = AutoConfig.from_pretrained(args.model)

    config.compress_ratio = 0.2
    config.recent_ratio = 0.5
    config.important_ratio = 0.5
    config.fixbudget = True
    config.budget = args.budget
    config.recent_size = 32
    config.sink_size = 16

    config.compressor = "pq_search"

    config.n_subvec_per_head = 2
    config.n_subbits = 6
    os.environ["SUBVEC"] = str(config.n_subvec_per_head)
    os.environ["SUBBITS"] = str(config.n_subbits)
    os.environ["MODE"] = "off"

    config.topr = 1
    config.gqa = True
    config.mean_v_trick = False
    config.score_func = "sum"
    config.max_iter = 0
    config.pp_size = 1
    config.device = torch.device(args.device)

    config.max_seq_len = 131072
    config.cache_block_size = 128
    config.global_cache_size = 4096
    config.cache_topk = config.global_cache_size // config.cache_block_size

    return config


def load_model(args, config):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        config=config,
        trust_remote_code=True,
    )

    model_name = args.model.lower()

    if "qwen" in model_name:
        model = VQQwen2ForCausalLM.from_pretrained(
            args.model,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    elif "llama" in model_name:
        model = VQLlama31ForCausalLM.from_pretrained(
            args.model,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(
            f"Unknown model type from model name: {args.model}. "
            "Expected model name containing 'qwen' or 'llama'."
        )

    model.patch(config)
    model = model.eval().to(args.device)
    return tokenizer, model


def build_input(tokenizer, input_file, max_input_len, device):
    with open(input_file, "r", encoding="utf-8") as f:
        input_string = f.read()

    encoded = tokenizer(input_string, truncation=False, return_tensors="pt")
    input_ids = encoded.input_ids

    if input_ids.shape[1] < max_input_len:
        repeat_times = (max_input_len + input_ids.shape[1] - 1) // input_ids.shape[1]
        input_ids = input_ids.repeat(1, repeat_times)

    input_ids = input_ids[:, :max_input_len].to(device)
    return input_ids


class OuterTokenTimingCriteria(StoppingCriteria):
    """Record first-token timing inside the same generate() call as component timing."""

    def __init__(self, prompt_len):
        self.prompt_len = int(prompt_len)
        self.first_token_time = None
        self.generated_tokens = 0

    def __call__(self, input_ids, scores, **kwargs):
        self.generated_tokens = max(0, input_ids.shape[1] - self.prompt_len)
        if self.generated_tokens >= 1 and self.first_token_time is None:
            torch.cuda.synchronize()
            self.first_token_time = time.perf_counter()
        return False


def append_csv(csv_path, row):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "time", "model", "input_len", "max_new_tokens", "budget",
        "stage", "time_ms", "time_s",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def timed_generate(model, tokenizer, input_ids, max_new_tokens, config):
    torch.cuda.synchronize()
    begin = time.perf_counter()
    prompt_len = input_ids.shape[1]
    token_timer = OuterTokenTimingCriteria(prompt_len)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=None,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            stopping_criteria=StoppingCriteriaList([token_timer]),
        )[0]

    if config.compressor == "pq_search":
        wait()

    torch.cuda.synchronize()
    end = time.perf_counter()
    total = end - begin
    generated_count = int(output_ids[prompt_len:].numel())
    ttft = None
    tpot = None
    if token_timer.first_token_time is not None:
        ttft = token_timer.first_token_time - begin
        if generated_count > 1:
            tpot = (total - ttft) / (generated_count - 1)
    return total, ttft, tpot


def main():
    args = parse_args()
    config = setup_config(args)

    print("=" * 80, flush=True)
    print(f"Model          : {args.model}", flush=True)
    print(f"Input lens     : {args.input_lens}", flush=True)
    print(f"Output len     : {args.max_new_tokens}", flush=True)
    print(f"Budget         : {args.budget}", flush=True)
    print(f"Warmup/Measure : {args.warmup_rounds}/{args.measure_rounds}", flush=True)
    print(f"SYNC_TEST_TIME : {os.environ.get('SYNC_TEST_TIME', '0')}", flush=True)
    print("=" * 80, flush=True)

    if config.compressor == "pq_search":
        initialize_objects(config, model=args.model)

    tokenizer, model = load_model(args, config)
    input_ids_all = build_input(tokenizer, args.input_file, max(args.input_lens), args.device)
    print(f"Actual prepared input_ids shape: {tuple(input_ids_all.shape)}", flush=True)

    # Initialize global timer stages
    init_timer(config.num_hidden_layers)
    init_all_stages()
    set_recording_state(True)

    total_rounds = args.warmup_rounds + args.measure_rounds
    all_results = []

    for round_idx in range(total_rounds):
        is_warmup = round_idx < args.warmup_rounds
        round_name = "warmup" if is_warmup else "measure"
        print(f"\n===== Round {round_idx + 1}/{total_rounds} ({round_name}) =====", flush=True)

        for seqlen in args.input_lens:
            input_ids = input_ids_all[:, :seqlen]

            reset_timer()

            total, ttft, tpot = timed_generate(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                config=config,
            )

            times = get_all_stages_time()
            if ttft is not None:
                times["ttft"] = ttft * 1000.0
            if tpot is not None:
                times["tpot"] = tpot * 1000.0
            for stage_name in PQCACHE_REQUIRED_STAGES:
                times.setdefault(stage_name, 0.0)

            print(f"input_len={seqlen}, total={total:.4f}s", flush=True)
            for stage_name, t in sorted(times.items()):
                print(f"  {stage_name:25s}: {t:8.2f} ms", flush=True)

            if not is_warmup:
                all_results.append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model": args.model,
                    "input_len": seqlen,
                    "max_new_tokens": args.max_new_tokens,
                    "budget": args.budget,
                    "stage": "total",
                    "time_ms": f"{total * 1000.0:.2f}",
                    "time_s": f"{total:.6f}",
                })
                for stage_name, t in sorted(times.items()):
                    all_results.append({
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model": args.model,
                        "input_len": seqlen,
                        "max_new_tokens": args.max_new_tokens,
                        "budget": args.budget,
                        "stage": stage_name,
                        "time_ms": f"{t:.2f}",
                        "time_s": f"{t / 1000.0:.6f}",
                    })

        torch.cuda.empty_cache()

    # Print summary
    print("\n" + "=" * 80, flush=True)
    print("Averaged Stage Times Per Input Length (averaged over measurement rounds)", flush=True)
    print("=" * 80, flush=True)

    # Average per (input_len, stage) across measurement rounds.
    # This preserves the per-input-length distinction instead of mixing them.
    group_totals = {}
    group_counts = {}
    for r in all_results:
        key = (r["input_len"], r["stage"])
        group_totals[key] = group_totals.get(key, 0.0) + float(r["time_ms"])
        group_counts[key] = group_counts.get(key, 0) + 1

    by_input_len = {}
    for (input_len, stage_name), total_t in group_totals.items():
        by_input_len.setdefault(input_len, []).append(
            (stage_name, total_t / group_counts[(input_len, stage_name)])
        )

    for input_len in sorted(by_input_len.keys()):
        print(f"\n--- input_len={input_len} ---", flush=True)
        for stage_name, avg in sorted(by_input_len[input_len]):
            print(f"  {stage_name:25s}: {avg:8.2f} ms", flush=True)
            row = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": args.model,
                "input_len": input_len,
                "max_new_tokens": args.max_new_tokens,
                "budget": args.budget,
                "stage": stage_name,
                "time_ms": f"{avg:.2f}",
                "time_s": f"{avg / 1000.0:.6f}",
            }
            append_csv(args.csv, row)

    print("=" * 80, flush=True)

    del model
    if config.compressor == "pq_search":
        del_objects()
    torch.cuda.empty_cache()
    logger.info("Breakdown test done.")


if __name__ == "__main__":
    main()
