import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import csv
import importlib
import time
from datetime import datetime

import numpy as np
import torch
from transformers import AutoTokenizer


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
    parser = argparse.ArgumentParser(description="Latency test for ClusterKV")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="./myinput.txt")
    parser.add_argument(
        "--input-lens",
        type=str,
        nargs="+",
        required=True,
        help="Input lengths, e.g. 4096 8192 16384 or 4096,8192,16384",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--csv", type=str, default="./latency_results.csv")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--measure-rounds", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=1.0)

    # ClusterKV-specific parameters. Defaults follow the native ClusterKV benchmark.
    parser.add_argument("--method", type=str, choices=["clusterkv", "full"], default="clusterkv")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--nlist", type=int, default=200)
    parser.add_argument("--niter", type=int, default=20)
    parser.add_argument("--sink", type=int, default=16)
    parser.add_argument("--window", type=int, default=320)
    parser.add_argument("--window-nlist", type=int, default=8)
    parser.add_argument("--offload", action="store_true")

    args = parser.parse_args()
    args.input_lens = parse_int_list(args.input_lens)
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")
    if args.warmup_rounds < 0 or args.measure_rounds < 1:
        raise ValueError("Require warmup_rounds >= 0 and measure_rounds >= 1")
    if args.offload and args.method != "clusterkv":
        raise ValueError("--offload is only meaningful for --method clusterkv")
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


def import_clusterkv_model_class(model_name):
    """
    Import the ClusterKV model class by model name.

    LLaMA path is the one used by the official ClusterKV benchmark.
    Qwen support depends on your local ClusterKV repository. This function tries
    several common module names and gives a clear error if none exists.
    """
    name = model_name.lower()

    if "llama" in name:
        from clusterkv.clusterkv_models.llama import LlamaForCausalLM
        return LlamaForCausalLM

    if "qwen" in name:
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

    model_cls = import_clusterkv_model_class(args.model)

    try:
        model = model_cls.from_pretrained(
            args.model,
            device_map=args.device,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    except TypeError:
        # Some custom ClusterKV classes do not accept trust_remote_code.
        model = model_cls.from_pretrained(
            args.model,
            device_map=args.device,
            torch_dtype=dtype,
        )

    model = model.eval()
    if not hasattr(model, "hf_device_map"):
        model = model.to(device)
    return tokenizer, model, dtype, device


def init_clusterkv(model, args, dtype, device):
    max_seq_len = max(args.input_lens) + args.max_new_tokens + 512
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


def get_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Unsupported model output type: {type(output)}")


def clear_clusterkv(model):
    if hasattr(model, "clusterkv_clear"):
        model.clusterkv_clear()


def cuda_synchronize(device):
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


@torch.inference_mode()
def timed_generate_clusterkv(model, input_ids, max_new_tokens, device):
    """
    Measure one generation run with the same high-level accounting as HF generate:
    - max_new_tokens=1 measures prefill/TTFT.
    - max_new_tokens=N measures prefill plus N-1 single-token decode steps.
    ClusterKV's native benchmark uses direct forward calls, so we do the same.
    """
    clear_clusterkv(model)
    cuda_synchronize(device)
    begin = time.perf_counter()

    output = model(input_ids=input_ids)
    logits = get_logits(output)
    pred_token_idx = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    for _ in range(max_new_tokens - 1):
        output = model(input_ids=pred_token_idx)
        logits = get_logits(output)
        pred_token_idx = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    cuda_synchronize(device)
    end = time.perf_counter()
    clear_clusterkv(model)
    return end - begin


def main():
    args = parse_args()
    seed_everything(42)
    maybe_set_torch_cuda_arch_list()

    print("=" * 80, flush=True)
    print(f"Model          : {args.model}", flush=True)
    print(f"Method         : {args.method}", flush=True)
    print(f"Input lens     : {args.input_lens}", flush=True)
    print(f"Output len     : {args.max_new_tokens}", flush=True)
    print(f"Budget         : {args.budget}", flush=True)
    print(f"Warmup/Measure : {args.warmup_rounds}/{args.measure_rounds}", flush=True)
    print(
        f"ClusterKV      : nlist={args.nlist}, niter={args.niter}, "
        f"sink={args.sink}, window={args.window}, window_nlist={args.window_nlist}, "
        f"offload={args.offload}",
        flush=True,
    )
    print("=" * 80, flush=True)

    tokenizer, model, dtype, device = load_model_and_tokenizer(args)
    max_seq_len, actual_token_budget = init_clusterkv(model, args, dtype, device)
    print(f"Initialized ClusterKV: max_seq_len={max_seq_len}, token_budget={actual_token_budget}", flush=True)

    input_ids_all = build_input(tokenizer, args.input_file, max(args.input_lens), device)
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

            ttft = timed_generate_clusterkv(
                model=model,
                input_ids=input_ids,
                max_new_tokens=1,
                device=device,
            )

            if args.sleep > 0:
                time.sleep(args.sleep)

            total = timed_generate_clusterkv(
                model=model,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )

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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n===== Average of measured rounds =====", flush=True)
    for seqlen in args.input_lens:
        ttft_arr = np.array(results[seqlen]["ttft"], dtype=np.float64)
        total_arr = np.array(results[seqlen]["total"], dtype=np.float64)
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

    clear_clusterkv(model)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("ClusterKV latency test done.", flush=True)


if __name__ == "__main__":
    main()
