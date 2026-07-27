import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["SYNC_TEST_TIME"] = "1"

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clusterkv.clusterkv_utils.global_timer import (  # noqa: E402
    get_all_stages_time,
    init_all_stages,
    init_timer,
    reset_timer,
    set_recording_state,
)


def parse_int_list(values):
    out = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Breakdown latency test for myllama ClusterKV Llama")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="./myinput.txt")
    parser.add_argument("--input-lens", type=str, nargs="+", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--csv", type=str, default="./breakdown_test/log/myllama_breakdown_results.csv")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--method", type=str, default="clusterkv", choices=["clusterkv", "full"])
    parser.add_argument("--nlist", type=int, default=200)
    parser.add_argument("--niter", type=int, default=20)
    parser.add_argument("--sink", type=int, default=16)
    parser.add_argument("--window", type=int, default=320)
    parser.add_argument("--window-nlist", type=int, default=8)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--measure-rounds", type=int, default=3)
    args = parser.parse_args()
    args.input_lens = parse_int_list(args.input_lens)
    if args.method == "full" and args.offload:
        raise ValueError("--offload is only valid with --method clusterkv")
    return args


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_sparse_params(args, input_len):
    if args.sink is not None:
        return args.sink
    return 4 if input_len == 1024 else 16


def load_model(args):
    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    from clusterkv.clusterkv_models.myllama import LlamaForCausalLM

    try:
        model = LlamaForCausalLM.from_pretrained(
            args.model,
            device_map=args.device,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    except TypeError:
        model = LlamaForCausalLM.from_pretrained(
            args.model,
            device_map=args.device,
            torch_dtype=dtype,
        )
    model.eval()
    return model, tokenizer, dtype, torch.device(args.device)


def build_input(tokenizer, input_file, max_input_len, device):
    with open(input_file, "r", encoding="utf-8") as f:
        input_string = f.read()
    encoded = tokenizer(input_string, truncation=False, return_tensors="pt")
    input_ids = encoded.input_ids
    if input_ids.shape[1] < max_input_len:
        repeat_times = (max_input_len + input_ids.shape[1] - 1) // input_ids.shape[1]
        input_ids = input_ids.repeat(1, repeat_times)
    return input_ids[:, :max_input_len].to(device)


def append_csv(csv_path, row):
    fieldnames = [
        "time",
        "model",
        "input_len",
        "max_new_tokens",
        "budget",
        "token_budget",
        "sink",
        "window",
        "window_nlist",
        "nlist",
        "niter",
        "offload",
        "dtype",
        "stage",
        "time_ms",
    ]
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def init_clusterkv(model, args, dtype, device, input_len):
    sink = resolve_sparse_params(args, input_len)
    max_seq_len = input_len + args.max_new_tokens + 512
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
        offload=args.offload,
    )
    return {
        "sink": sink,
        "token_budget": token_budget,
        "max_seq_len": max_seq_len,
    }


def timed_generate(model, tokenizer, input_ids, max_new_tokens):
    torch.cuda.synchronize()
    begin = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=None,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )[0]
    torch.cuda.synchronize()
    end = time.perf_counter()
    prompt_len = input_ids.shape[1]
    generated_text = tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)
    return end - begin, generated_text


def main():
    args = parse_args()
    seed_everything(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for breakdown measurement.")

    print("=" * 80, flush=True)
    print("MyLlama ClusterKV Component Breakdown Test", flush=True)
    print(f"Model          : {args.model}", flush=True)
    print(f"Input lens     : {args.input_lens}", flush=True)
    print(f"Output len     : {args.max_new_tokens}", flush=True)
    print(f"Budget         : {args.budget}", flush=True)
    print(f"Offload        : {args.offload}", flush=True)
    print(f"Warmup/Measure : {args.warmup_rounds}/{args.measure_rounds}", flush=True)
    print(f"CSV            : {args.csv}", flush=True)
    print("=" * 80, flush=True)

    model, tokenizer, dtype, device = load_model(args)
    input_ids_all = build_input(tokenizer, args.input_file, max(args.input_lens), device)
    init_timer(model.config.num_hidden_layers)
    init_all_stages()
    set_recording_state(True)

    total_rounds = args.warmup_rounds + args.measure_rounds
    rows = []
    for round_idx in range(total_rounds):
        is_warmup = round_idx < args.warmup_rounds
        round_name = "warmup" if is_warmup else "measure"
        print(f"\n===== Round {round_idx + 1}/{total_rounds} ({round_name}) =====", flush=True)

        for input_len in args.input_lens:
            if getattr(model.model, "cache", None) is not None:
                model.clusterkv_clear()
            runtime_cfg = init_clusterkv(model, args, dtype, device, input_len)
            input_ids = input_ids_all[:, :input_len]
            reset_timer()
            elapsed_s, generated_text = timed_generate(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
            )
            times = get_all_stages_time()
            print(f"input_len={input_len}, total={elapsed_s:.4f}s", flush=True)
            for stage_name, t in sorted(times.items()):
                print(f"  {stage_name:25s}: {t:8.2f} ms", flush=True)
            print(f"  generated: {generated_text}", flush=True)

            if is_warmup:
                continue
            for stage_name, t in sorted(times.items()):
                rows.append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model": args.model,
                    "input_len": input_len,
                    "max_new_tokens": args.max_new_tokens,
                    "budget": args.budget,
                    "token_budget": runtime_cfg["token_budget"],
                    "sink": runtime_cfg["sink"],
                    "window": args.window,
                    "window_nlist": args.window_nlist,
                    "nlist": args.nlist,
                    "niter": args.niter,
                    "offload": int(args.offload),
                    "dtype": args.dtype,
                    "stage": stage_name,
                    "time_ms": f"{t:.4f}",
                })
        torch.cuda.empty_cache()

    groups = {}
    counts = {}
    for row in rows:
        key = (row["input_len"], row["stage"])
        groups[key] = groups.get(key, 0.0) + float(row["time_ms"])
        counts[key] = counts.get(key, 0) + 1

    print("\n" + "=" * 80, flush=True)
    print("Averaged Stage Times Per Input Length", flush=True)
    print("=" * 80, flush=True)
    for input_len in sorted({row["input_len"] for row in rows}):
        print(f"\n--- input_len={input_len} ---", flush=True)
        template = next(row for row in rows if row["input_len"] == input_len)
        stage_names = sorted({row["stage"] for row in rows if row["input_len"] == input_len})
        for stage_name in stage_names:
            avg_ms = groups[(input_len, stage_name)] / counts[(input_len, stage_name)]
            print(f"  {stage_name:25s}: {avg_ms:8.2f} ms", flush=True)
            out = dict(template)
            out["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            out["stage"] = stage_name
            out["time_ms"] = f"{avg_ms:.4f}"
            append_csv(args.csv, out)

    del model
    torch.cuda.empty_cache()
    logger.info("myllama ClusterKV breakdown test done.")


if __name__ == "__main__":
    main()
