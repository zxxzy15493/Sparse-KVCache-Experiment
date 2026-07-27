import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from transformers import AutoConfig, AutoTokenizer
from loguru import logger

from vq_method.retrieval_based.pq_search import initialize_objects, del_objects, wait
from vq_method.llama31_patch import VQLlama31ForCausalLM
from vq_method.qwen25_patch import VQQwen2ForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Peak memory test for PQCache")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-family", type=str, choices=["llama", "qwen"], required=True)

    parser.add_argument("--input-file", type=str, default="./myinput.txt")
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)

    parser.add_argument("--csv", type=str, default="./mem_results.csv")
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()


def setup_config(args):
    config = AutoConfig.from_pretrained(args.model)

    # ===== PQCache default setting: follow efficiency test =====
    config.compress_ratio = 0.2
    config.recent_ratio = 0.5
    config.important_ratio = 0.5
    config.fixbudget = True
    config.budget = args.budget

    # Consistent with the original efficiency test
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

    # 1k input experiment: sink and recent both set to 4
    if args.input_len == 1024:
        config.recent_size = 4
        config.sink_size = 4
        config.cache_block_size = 8
        config.global_cache_size = 128
        config.cache_topk = config.global_cache_size // config.cache_block_size
    else:
        config.recent_size = 32
        config.sink_size = 16

    config.compressor = "pq_search"
    # config.compressor = "original"

    

    return config


def load_model(args, config):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        config=config,
        trust_remote_code=True,
    )

    if args.model_family == "qwen":
        model = VQQwen2ForCausalLM.from_pretrained(
            args.model,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    elif args.model_family == "llama":
        model = VQLlama31ForCausalLM.from_pretrained(
            args.model,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unknown model family: {args.model_family}")

    model.patch(config)
    model = model.eval().to(args.device)

    return tokenizer, model


def build_input(tokenizer, input_file, input_len, device):
    with open(input_file, "r", encoding="utf-8") as f:
        input_string = f.read()

    encoded = tokenizer(
        input_string,
        truncation=False,
        return_tensors="pt",
    )

    input_ids = encoded.input_ids

    # If myinput.txt is too short, repeat tokens to ensure 64k experiment has enough length
    if input_ids.shape[1] < input_len:
        repeat_times = (input_len + input_ids.shape[1] - 1) // input_ids.shape[1]
        input_ids = input_ids.repeat(1, repeat_times)

    input_ids = input_ids[:, :input_len].to(device)

    return input_ids


def append_csv(csv_path, row):
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "time",
        "model_family",
        "model",
        "input_len",
        "max_new_tokens",
        "budget",
        "sink_size",
        "recent_size",
        "peak_alloc_mib",
        "peak_alloc_bytes",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()

    print("=" * 80, flush=True)
    print(f"Model        : {args.model}", flush=True)
    print(f"Model family : {args.model_family}", flush=True)
    print(f"Input len    : {args.input_len}", flush=True)
    print(f"Output len   : {args.max_new_tokens}", flush=True)
    print(f"Budget       : {args.budget}", flush=True)
    print("=" * 80, flush=True)

    config = setup_config(args)

    if config.compressor == "pq_search":
        initialize_objects(config, model=args.model)

    tokenizer, model = load_model(args, config)
    input_ids = build_input(tokenizer, args.input_file, args.input_len, args.device)

    print(f"Actual input_ids shape: {tuple(input_ids.shape)}", flush=True)
    print(f"sink_size={config.sink_size}, recent_size={config.recent_size}", flush=True)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Enable memory history as requested
    try:
        torch.cuda.memory._record_memory_history()
    except TypeError:
        torch.cuda.memory._record_memory_history(enabled=True)

    torch.cuda.synchronize()
    begin = time.perf_counter()

    with torch.no_grad():
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=None,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.max_new_tokens,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )[0]

    if config.compressor == "pq_search":
        wait()

    torch.cuda.synchronize()
    end = time.perf_counter()

    peak_alloc = torch.cuda.memory.max_memory_allocated()

    print(f"elapsed       = {end - begin:.4f} s", flush=True)
    print(f"peak_alloc    = {peak_alloc / 1024**2:.2f} MiB", flush=True)

    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_family": args.model_family,
        "model": args.model,
        "input_len": args.input_len,
        "max_new_tokens": args.max_new_tokens,
        "budget": args.budget,
        "sink_size": config.sink_size,
        "recent_size": config.recent_size,
        "peak_alloc_mib": f"{peak_alloc / 1024**2:.2f}",
        "peak_alloc_bytes": peak_alloc,
    }
    append_csv(args.csv, row)

    try:
        torch.cuda.memory._record_memory_history(enabled=None)
    except Exception:
        pass

    del model
    if config.compressor == "pq_search":
        del_objects()

    torch.cuda.empty_cache()
    logger.info("Memory test done.")


if __name__ == "__main__":
    main()
