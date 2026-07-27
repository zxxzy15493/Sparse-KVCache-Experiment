import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="ClusterKV latency test: single input length and single budget")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="./myinput.txt")
    parser.add_argument("--input-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--budget", type=int, default=1024)

    parser.add_argument("--csv", type=str, default="log_latency/clusterkv_latency_single.csv")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16"], default="float16")

    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--measure-rounds", type=int, default=3)

    parser.add_argument("--method", type=str, choices=["clusterkv", "full"], default="clusterkv")
    parser.add_argument(
        "--impl",
        type=str,
        choices=["clusterkv", "myllama", "myllama2"],
        default="myllama",
        help="'clusterkv' uses the original implementation; 'myllama' uses flash-attn; 'myllama2' uses FlashInfer.",
    )
    parser.add_argument("--nlist", type=int, default=200)
    parser.add_argument("--niter", type=int, default=20)
    parser.add_argument("--sink", type=int, default=16)
    parser.add_argument("--window", type=int, default=320)
    parser.add_argument("--window-nlist", type=int, default=8)
    parser.add_argument("--offload", action="store_true")

    args = parser.parse_args()

    if args.input_len < 1:
        raise ValueError("--input-len must be >= 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")
    if args.warmup_rounds < 0:
        raise ValueError("--warmup-rounds must be >= 0")
    if args.measure_rounds < 1:
        raise ValueError("--measure-rounds must be >= 1")
    if args.offload and args.method != "clusterkv":
        raise ValueError("--offload is only supported for --method clusterkv")
    return args


def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def maybe_set_torch_cuda_arch_list():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")


def import_clusterkv_model_class(model_name, impl):
    name = model_name.lower()

    if "llama" in name:
        if impl == "myllama":
            from clusterkv.clusterkv_models.myllama import LlamaForCausalLM
        elif impl == "myllama2":
            from clusterkv.clusterkv_models.myllama2 import LlamaForCausalLM
        else:
            from clusterkv.clusterkv_models.llama import LlamaForCausalLM
        return LlamaForCausalLM

    if "qwen" in name:
        if impl == "myllama":
            from clusterkv.clusterkv_models.myqwen2 import Qwen2ForCausalLM
        else:
            from clusterkv.clusterkv_models.qwen2 import Qwen2ForCausalLM
        return Qwen2ForCausalLM

    raise ValueError(
        f"Unknown model type from model name: {model_name}. "
        "Expected model name containing 'llama' or 'qwen'."
    )


def load_model_and_tokenizer(args):
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_cls = import_clusterkv_model_class(args.model, args.impl)

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

    model = model.eval()

    if not hasattr(model, "hf_device_map"):
        model = model.to(device)

    return tokenizer, model, dtype, device


def build_input(tokenizer, input_file, input_len, device):
    with open(input_file, "r", encoding="utf-8") as f:
        input_string = f.read()

    encoded = tokenizer(input_string, truncation=False, return_tensors="pt")
    input_ids = encoded.input_ids

    if input_ids.shape[1] < input_len:
        repeat_times = (input_len + input_ids.shape[1] - 1) // input_ids.shape[1]
        input_ids = input_ids.repeat(1, repeat_times)

    input_ids = input_ids[:, :input_len].to(device)
    return input_ids


def get_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Unsupported model output type: {type(output)}")


def cuda_synchronize(device):
    device = torch.device(device)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_clusterkv(model):
    if hasattr(model, "clusterkv_clear"):
        model.clusterkv_clear()


def init_clusterkv_once(model, args, dtype, device):
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
        sink=args.sink,
        window=args.window,
        window_nlist=args.window_nlist,
        offload=True if args.offload else False,
    )

    return max_seq_len, token_budget


@torch.inference_mode()
def run_one_round(model, input_ids, max_new_tokens, device):
    clear_clusterkv(model)

    cuda_synchronize(device)
    prefill_begin = time.perf_counter()

    output = model(input_ids=input_ids)

    cuda_synchronize(device)
    prefill_end = time.perf_counter()

    logits = get_logits(output)
    pred_token_idx = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    decode_latencies = []

    for _ in range(max_new_tokens):
        cuda_synchronize(device)
        decode_begin = time.perf_counter()

        output = model(input_ids=pred_token_idx)

        cuda_synchronize(device)
        decode_end = time.perf_counter()

        decode_latencies.append(decode_end - decode_begin)

        logits = get_logits(output)
        pred_token_idx = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    clear_clusterkv(model)

    prefill_latency = prefill_end - prefill_begin
    decode_total = float(np.sum(decode_latencies))
    decode_per_token = decode_total / max_new_tokens
    total_latency = prefill_latency + decode_total

    return prefill_latency, decode_total, decode_per_token, total_latency


def append_csv(csv_path, row):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "time",
        "model",
        "method",
        "impl",
        "input_len",
        "max_new_tokens",
        "budget",
        "actual_token_budget",
        "nlist",
        "niter",
        "offload",
        "avg_prefill_s",
        "avg_decode_total_s",
        "avg_decode_per_token_s",
        "avg_total_s",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()

    seed_everything(42)
    maybe_set_torch_cuda_arch_list()

    print("=" * 80, flush=True)
    print(f"Model      : {args.model}", flush=True)
    print(f"Method     : {args.method}", flush=True)
    print(f"Impl       : {args.impl}", flush=True)
    print(f"Input file : {args.input_file}", flush=True)
    print(f"Input len  : {args.input_len}", flush=True)
    print(f"Output len : {args.max_new_tokens}", flush=True)
    print(f"Budget     : {args.budget}", flush=True)
    print(f"CSV        : {args.csv}", flush=True)
    print(f"ClusterKV  : nlist={args.nlist}, niter={args.niter}, offload={args.offload}", flush=True)
    print("=" * 80, flush=True)

    tokenizer, model, dtype, device = load_model_and_tokenizer(args)

    max_seq_len, actual_token_budget = init_clusterkv_once(
        model=model,
        args=args,
        dtype=dtype,
        device=device,
    )

    print(
        f"Initialized ClusterKV once: max_seq_len={max_seq_len}, "
        f"token_budget={actual_token_budget}",
        flush=True,
    )

    input_ids = build_input(
        tokenizer=tokenizer,
        input_file=args.input_file,
        input_len=args.input_len,
        device=device,
    )

    print(f"Prepared input_ids shape: {tuple(input_ids.shape)}", flush=True)

    total_rounds = args.warmup_rounds + args.measure_rounds

    prefill_list = []
    decode_total_list = []
    decode_per_token_list = []
    total_list = []

    for round_idx in range(total_rounds):
        is_warmup = round_idx < args.warmup_rounds
        round_name = "warmup" if is_warmup else "measure"

        prefill_s, decode_total_s, decode_per_token_s, total_s = run_one_round(
            model=model,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )

        print(
            f"[{round_name} {round_idx + 1}/{total_rounds}] "
            f"input_len={args.input_len}, budget={args.budget}, "
            f"prefill={prefill_s:.6f}s, "
            f"decode_total={decode_total_s:.6f}s, "
            f"decode_per_token={decode_per_token_s:.6f}s, "
            f"total={total_s:.6f}s",
            flush=True,
        )

        if not is_warmup:
            prefill_list.append(prefill_s)
            decode_total_list.append(decode_total_s)
            decode_per_token_list.append(decode_per_token_s)
            total_list.append(total_s)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "method": args.method,
        "impl": args.impl,
        "input_len": args.input_len,
        "max_new_tokens": args.max_new_tokens,
        "budget": args.budget,
        "actual_token_budget": actual_token_budget,
        "nlist": args.nlist,
        "niter": args.niter,
        "offload": args.offload,
        "avg_prefill_s": f"{float(np.mean(prefill_list)):.6f}",
        "avg_decode_total_s": f"{float(np.mean(decode_total_list)):.6f}",
        "avg_decode_per_token_s": f"{float(np.mean(decode_per_token_list)):.6f}",
        "avg_total_s": f"{float(np.mean(total_list)):.6f}",
    }

    append_csv(args.csv, row)

    print("\n===== Average of measured rounds =====", flush=True)
    print(
        f"input_len={args.input_len}, "
        f"budget={args.budget}, "
        f"avg_prefill={row['avg_prefill_s']}s, "
        f"avg_decode_total={row['avg_decode_total_s']}s, "
        f"avg_decode_per_token={row['avg_decode_per_token_s']}s, "
        f"avg_total={row['avg_total_s']}s",
        flush=True,
    )

    clear_clusterkv(model)
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"ClusterKV latency test done. Results saved to {args.csv}", flush=True)


if __name__ == "__main__":
    main()
