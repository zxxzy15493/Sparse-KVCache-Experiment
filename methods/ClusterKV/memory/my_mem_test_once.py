import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from loguru import logger


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Peak memory test for my ClusterKV models")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-family", type=str, choices=["llama", "qwen"], required=True)
    parser.add_argument("--impl", type=str, choices=["clusterkv", "myllama", "myllama2"], default="myllama")

    parser.add_argument("--input-file", type=str, default="./myinput.txt")
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)

    parser.add_argument("--csv", type=str, default="./my_mem_results.csv")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--method", type=str, default="clusterkv", choices=["clusterkv", "full"])

    parser.add_argument("--nlist", type=int, default=200)
    parser.add_argument("--niter", type=int, default=20)
    parser.add_argument("--sink", type=int, default=None)
    parser.add_argument("--window", type=int, default=320)
    parser.add_argument("--window-nlist", type=int, default=8)
    parser.add_argument("--offload", action="store_true")

    args = parser.parse_args()
    if args.input_len < 1:
        raise ValueError("--input-len must be >= 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")
    if args.offload and args.method != "clusterkv":
        raise ValueError("--offload is only meaningful for --method clusterkv")
    return args


def resolve_sparse_params(args):
    if args.sink is not None:
        return args.sink
    return 4 if args.input_len == 1024 else 16


def import_model_class(args):
    if args.model_family == "llama":
        if args.impl == "myllama":
            from clusterkv.clusterkv_models.myllama import LlamaForCausalLM
        elif args.impl == "myllama2":
            from clusterkv.clusterkv_models.myllama2 import LlamaForCausalLM
        else:
            from clusterkv.clusterkv_models.llama import LlamaForCausalLM
        return LlamaForCausalLM

    if args.model_family == "qwen":
        if args.impl == "myllama":
            from clusterkv.clusterkv_models.myqwen2 import Qwen2ForCausalLM
        else:
            from clusterkv.clusterkv_models.qwen2 import Qwen2ForCausalLM
        return Qwen2ForCausalLM

    raise ValueError(f"Unknown model family: {args.model_family}")


def load_model(args):
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_cls = import_model_class(args)
    try:
        model = model_cls.from_pretrained(
            args.model,
            device_map=args.device,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    except TypeError:
        model = model_cls.from_pretrained(
            args.model,
            device_map=args.device,
            torch_dtype=dtype,
        )

    model.eval()
    return model, tokenizer, dtype, device


def init_clusterkv(model, args, dtype, device):
    sink = resolve_sparse_params(args)
    max_seq_len = args.input_len + args.max_new_tokens + 512
    token_budget = 102400 if args.method == "full" else args.budget

    model.clusterkv_init(
        nlist=args.nlist,
        niter=args.niter,
        max_seq_len=max_seq_len,
        token_budget=token_budget,
        dtype=dtype,
        device=device,
        full=(args.method == "full"),
        sink=sink,
        window=args.window,
        window_nlist=args.window_nlist,
        offload=True if args.offload else False,
    )

    return {
        "sink": sink,
        "window": args.window,
        "token_budget": token_budget,
        "max_seq_len": max_seq_len,
    }


def build_input(tokenizer, input_file, input_len, device):
    with open(input_file, "r", encoding="utf-8") as f:
        input_string = f.read()

    encoded = tokenizer(input_string, truncation=False, return_tensors="pt")
    input_ids = encoded.input_ids

    if input_ids.shape[1] < input_len:
        repeat_times = (input_len + input_ids.shape[1] - 1) // input_ids.shape[1]
        input_ids = input_ids.repeat(1, repeat_times)

    return input_ids[:, :input_len].to(device)


def append_csv(csv_path, row):
    file_exists = os.path.exists(csv_path)
    fieldnames = [
        "time",
        "model_family",
        "impl",
        "model",
        "method",
        "input_len",
        "max_new_tokens",
        "budget",
        "token_budget",
        "max_seq_len",
        "sink",
        "window",
        "window_nlist",
        "nlist",
        "niter",
        "offload",
        "dtype",
        "elapsed_s",
        "peak_alloc_mib",
        "peak_reserved_mib",
        "peak_alloc_bytes",
        "peak_reserved_bytes",
    ]

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()
    seed_everything(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for memory measurement.")

    print("=" * 80, flush=True)
    print(f"Model        : {args.model}", flush=True)
    print(f"Model family : {args.model_family}", flush=True)
    print(f"Impl         : {args.impl}", flush=True)
    print(f"Method       : {args.method}", flush=True)
    print(f"Input len    : {args.input_len}", flush=True)
    print(f"Output len   : {args.max_new_tokens}", flush=True)
    print(f"Budget       : {args.budget}", flush=True)
    print(f"Device       : {args.device}", flush=True)
    print(f"Dtype        : {args.dtype}", flush=True)
    print(f"Offload      : {args.offload}", flush=True)
    print("=" * 80, flush=True)

    model, tokenizer, dtype, device = load_model(args)
    runtime_cfg = init_clusterkv(model, args, dtype, device)
    input_ids = build_input(tokenizer, args.input_file, args.input_len, device)

    print(f"Actual input_ids shape: {tuple(input_ids.shape)}", flush=True)
    print(f"sink={runtime_cfg['sink']}, window={runtime_cfg['window']}", flush=True)
    print(f"max_seq_len={runtime_cfg['max_seq_len']}", flush=True)
    print(f"token_budget={runtime_cfg['token_budget']}", flush=True)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    begin = time.perf_counter()

    with torch.inference_mode():
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

    torch.cuda.synchronize()
    end = time.perf_counter()

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    elapsed = end - begin

    print(f"elapsed       = {elapsed:.4f} s", flush=True)
    print(f"peak_alloc    = {peak_alloc / 1024**2:.2f} MiB", flush=True)
    print(f"peak_reserved = {peak_reserved / 1024**2:.2f} MiB", flush=True)

    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_family": args.model_family,
        "impl": args.impl,
        "model": args.model,
        "method": args.method,
        "input_len": args.input_len,
        "max_new_tokens": args.max_new_tokens,
        "budget": args.budget,
        "token_budget": runtime_cfg["token_budget"],
        "max_seq_len": runtime_cfg["max_seq_len"],
        "sink": runtime_cfg["sink"],
        "window": runtime_cfg["window"],
        "window_nlist": args.window_nlist,
        "nlist": args.nlist,
        "niter": args.niter,
        "offload": int(args.offload),
        "dtype": args.dtype,
        "elapsed_s": f"{elapsed:.6f}",
        "peak_alloc_mib": f"{peak_alloc / 1024**2:.2f}",
        "peak_reserved_mib": f"{peak_reserved / 1024**2:.2f}",
        "peak_alloc_bytes": peak_alloc,
        "peak_reserved_bytes": peak_reserved,
    }
    append_csv(args.csv, row)

    try:
        model.clusterkv_clear()
    except Exception:
        pass

    del model
    torch.cuda.empty_cache()
    logger.info("my ClusterKV memory test done.")


if __name__ == "__main__":
    main()
