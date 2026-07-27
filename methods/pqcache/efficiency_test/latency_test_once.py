import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import csv
import time
from datetime import datetime

import numpy as np
import torch
from loguru import logger
from transformers import AutoConfig, AutoTokenizer

from vq_method.retrieval_based.pq_search import initialize_objects, del_objects, wait
from vq_method.llama31_patch import VQLlama31ForCausalLM
from vq_method.qwen25_patch import VQQwen2ForCausalLM


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
    parser = argparse.ArgumentParser(description="Latency test for PQCache")

    parser.add_argument("--model", type=str, required=True)
    # parser.add_argument("--model-family", type=str, choices=["llama", "qwen"], required=True)

    parser.add_argument("--input-file", type=str, default="./myinput.txt")
    parser.add_argument("--input-lens", type=str, nargs="+", required=True,
                        help="Input lengths, e.g. 4096 8192 16384 or 4096,8192,16384")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--budget", type=int, default=1024)

    parser.add_argument("--csv", type=str, default="./latency_results.csv")
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

    config.max_seq_len = 270000
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


def append_csv(csv_path, row):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "time",
        "model",
        "input_len",
        "max_new_tokens",
        "budget",
        "avg_ttft_s",
        "avg_decode_per_token_s",
        "avg_total_s",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def timed_generate(model, tokenizer, input_ids, max_new_tokens, config):
    torch.cuda.synchronize()
    begin = time.perf_counter()

    with torch.no_grad():
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=None,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )[0]

    if config.compressor == "pq_search":
        wait()

    torch.cuda.synchronize()
    end = time.perf_counter()
    return end - begin


def main():
    args = parse_args()
    config = setup_config(args)

    print("=" * 80, flush=True)
    print(f"Model          : {args.model}", flush=True)
    print(f"Input lens     : {args.input_lens}", flush=True)
    print(f"Output len     : {args.max_new_tokens}", flush=True)
    print(f"Budget         : {args.budget}", flush=True)
    print(f"Warmup/Measure : {args.warmup_rounds}/{args.measure_rounds}", flush=True)
    print(f"Cache          : block={config.cache_block_size}, global={config.global_cache_size}, topk={config.cache_topk}", flush=True)
    print("=" * 80, flush=True)

    if config.compressor == "pq_search":
        initialize_objects(config, model=args.model)

    tokenizer, model = load_model(args, config)
    input_ids_all = build_input(tokenizer, args.input_file, max(args.input_lens), args.device)
    print(f"Actual prepared input_ids shape: {tuple(input_ids_all.shape)}", flush=True)

    total_rounds = args.warmup_rounds + args.measure_rounds
    results = {
        seqlen: {"ttft": [], "total": [], "decode_elapsed": [], "decode_per_token": []}
        for seqlen in args.input_lens
    }

    for round_idx in range(total_rounds):
        is_warmup = round_idx < args.warmup_rounds
        round_name = "warmup" if is_warmup else "measure"
        print(f"\n===== Round {round_idx + 1}/{total_rounds} ({round_name}) =====", flush=True)

        for seqlen in args.input_lens:
            input_ids = input_ids_all[:, :seqlen]

            ttft = timed_generate(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_new_tokens=1,
                config=config,
            )
            torch.cuda.empty_cache()
            if args.sleep > 0:
                time.sleep(args.sleep)

            total = timed_generate(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                config=config,
            )
            torch.cuda.empty_cache()
            decode_elapsed = total - ttft
            
            if args.max_new_tokens > 1:
                decode_per_token = decode_elapsed / (args.max_new_tokens - 1)
            else:
                decode_per_token = 0.0

            print(
                f"input_len={seqlen}, output_len={args.max_new_tokens}, "
                f"ttft={ttft:.6f}s, total={total:.6f}s, "
                f"decode_elapsed={decode_elapsed:.6f}s, "
                f"decode_per_token={decode_per_token:.6f}s",
                flush=True,
            )

            if not is_warmup:
                results[seqlen]["ttft"].append(ttft)
                results[seqlen]["total"].append(total)
                results[seqlen]["decode_elapsed"].append(decode_elapsed)
                results[seqlen]["decode_per_token"].append(decode_per_token)

        torch.cuda.empty_cache()

    print("\n===== Average of measured rounds =====", flush=True)
    for seqlen in args.input_lens:
        ttft_arr = np.array(results[seqlen]["ttft"], dtype=np.float64)
        total_arr = np.array(results[seqlen]["total"], dtype=np.float64)
        decode_elapsed_arr = np.array(results[seqlen]["decode_elapsed"], dtype=np.float64)
        decode_per_token_arr = np.array(results[seqlen]["decode_per_token"], dtype=np.float64)

        row = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.model,
            "input_len": seqlen,
            "max_new_tokens": args.max_new_tokens,
            "budget": args.budget,
            "avg_ttft_s": f"{float(np.mean(ttft_arr)):.6f}",
            "avg_decode_per_token_s": f"{float(np.mean(decode_per_token_arr)):.6f}",
            "avg_total_s": f"{float(np.mean(total_arr)):.6f}",
        }
        append_csv(args.csv, row)

        print(
            f"input_len={seqlen}, "
            f"avg_ttft={row['avg_ttft_s']}s, "
            f"avg_total={row['avg_total_s']}s, "
            f"avg_decode_per_token={row['avg_decode_per_token_s']}s",
            flush=True,
        )

    del model
    if config.compressor == "pq_search":
        del_objects()
    torch.cuda.empty_cache()
    logger.info("Latency test done.")


if __name__ == "__main__":
    main()
