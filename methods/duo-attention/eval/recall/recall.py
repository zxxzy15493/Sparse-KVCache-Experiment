"""
DuoAttention recall evaluation script.

Measures recall@100, recall@k, and selected_attn_ratio for DuoAttention's
head-level KV compression on LongBench and RULER datasets.

Usage:
    # LongBench
    python recall.py --model Qwen/Qwen2.5-7B-Instruct \\
        --dataset ../../../benchmarks/Longbench_recall --dataset_name qasper \\
        --attn_load_dir ../../attn_patterns/Qwen2.5-7B-Instruct --sparsity 0.5

    # RULER
    python recall.py --model Qwen/Qwen2.5-7B-Instruct \\
        --dataset /path/to/ruler/data --task fwe \\
        --attn_load_dir /path/to/attn_pattern --sparsity 0.5
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Add duo-attention project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
# Fix for PyTorch meta device nonzero() issue in reorder_linear_weights
import torch.fx.experimental._config as fx_config
fx_config.meta_nonzero_assume_all_nonzero = True
import argparse
import json
import csv
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import duo_recall  # use module reference for current_sample_stats
from duo_recall import (
    enable_duo_recall,
    stats_registry,
    init_new_sample_registry,
)

# ============================================================
# Default output base
# ============================================================
base_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recall_results")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help="Model name or path")
    parser.add_argument('--dataset', type=str,
                        default="../../../benchmarks/Longbench_recall",
                        help="The folder of dataset to evaluate on")
    parser.add_argument('--dataset_name', type=str, default=None,
                        help="Dataset name for LongBench (loads {dataset}/{name}.jsonl)")
    parser.add_argument('--task', type=str, default=None,
                        help="Task name for RULER (loads {dataset}/{task}/validation.jsonl)")

    # DuoAttention args
    parser.add_argument('--attn_load_dir', type=str, default=None,
                        help="Attention pattern directory")
    parser.add_argument("--sink_size", type=int, default=64)
    parser.add_argument("--recent_size", type=int, default=256)
    parser.add_argument("--sparsity", type=float, default=0.5)

    # Generation args
    parser.add_argument('--max_length', type=int, default=16384,
                        help="Max input length")
    parser.add_argument('--max_gen', type=int, default=64,
                        help="Max generation length")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Output directory (default: recall_results/<model>/<dataset>)")

    return parser.parse_args()


def build_chat(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False
    )


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def get_output_path(args):
    """Determine output file path (JSONL for preds, CSV for recall metrics)."""
    model_short = args.model.rstrip("/")
    if "/" in model_short:
        model_short = model_short.split("/")[-1]

    sparsity_dir = f"sp{args.sparsity}"

    # Separate RULER and LongBench into distinct subdirectories
    if args.task:
        sub_dir = "ruler"
    elif args.dataset_name:
        sub_dir = "longbench"
    else:
        sub_dir = os.path.basename(os.path.dirname(args.dataset)) if args.dataset else ""

    output_dir = os.path.join(base_output, model_short, sparsity_dir, sub_dir)
    if args.output_dir is not None:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Use dataset_name or task as the filename prefix
    name = args.dataset_name if args.dataset_name else args.task
    if name is None:
        name = "unknown"
    return os.path.join(output_dir, f"{name}.jsonl"), output_dir


def reset_sample_cache_state(model):
    """Seal current sample stats and reset per-layer recall state."""
    init_new_sample_registry()
    for module in model.modules():
        if hasattr(module, "_recall_enabled"):
            module._recall_prefill_done = False
            module._recall_step = 0


@torch.inference_mode()
def get_pred_single_gpu(data, max_length, max_gen, prompt_format, dataset,
                        model_name, output_jsonl, output_dir, args):
    print(f"Loading model from {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="flash_attention_2",
    ).eval()

    device = model.device
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    gen_config = GenerationConfig.from_pretrained(model_name)
    eos_token_ids = gen_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    # ---- Load attention pattern and enable DuoAttention ----
    # Inline helper functions to avoid tensor_parallel dependency in duo_attn.utils
    import numpy as np

    def _load_attn_pattern(attn_load_dir):
        full_attention_heads = np.loadtxt(
            os.path.join(attn_load_dir, "full_attention_heads.tsv"),
            dtype=float, delimiter="\t",
        )
        full_attention_heads = np.clip(full_attention_heads, 0, 1)
        config = json.load(open(os.path.join(attn_load_dir, "config.json")))
        return full_attention_heads, config["sink_size"], config["recent_size"]

    def _sparsify_attention_heads(full_attention_heads, sparsity):
        full_attention_heads = full_attention_heads + np.random.uniform(0, 1e-6, full_attention_heads.shape)
        threshold = np.quantile(full_attention_heads, sparsity)
        if sparsity >= 1: threshold = 2
        if sparsity <= 0: threshold = -1
        full_attention_heads = (full_attention_heads >= threshold).astype(float)
        true_sparsity = 1 - full_attention_heads.mean()
        return full_attention_heads, true_sparsity

    def _to_device(inputs, device):
        if isinstance(inputs, dict):
            return {k: _to_device(v, device) for k, v in inputs.items()}
        return inputs.to(device)

    # enable_qwen2_duo_attention_eval already calls enable_tuple_kv_cache internally
    from duo_attn.patch.qwen2 import enable_qwen2_duo_attention_eval

    print(f"Loading attention pattern from {args.attn_load_dir} with sparsity {args.sparsity}")
    full_attention_heads, sink_size, recent_size = _load_attn_pattern(args.attn_load_dir)
    sink_size = args.sink_size if args.sink_size else sink_size
    recent_size = args.recent_size if args.recent_size else recent_size

    full_attention_heads, true_sparsity = _sparsify_attention_heads(
         full_attention_heads, args.sparsity
     )
    print(f"True sparsity: {true_sparsity}")

    enable_qwen2_duo_attention_eval(model, full_attention_heads, sink_size, recent_size)

    # ---- Enable recall tracking ----
    enable_duo_recall(model, sink_size=sink_size, recent_size=recent_size)

    # ---- Resume support ----
    done_count = 0
    if os.path.exists(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            done_count = sum(1 for line in f if line.strip())

    remaining_data = data[done_count:]
    if done_count > 0:
        print(f"Resuming from sample {done_count}, {len(remaining_data)} remaining.")

    # ---- Main evaluation loop ----
    with open(output_jsonl, "a", encoding="utf-8") as fout:
        for json_obj in tqdm(remaining_data):
            reset_sample_cache_state(model)

            prompt = prompt_format.format(**json_obj)
            answers = json_obj.get("answers", json_obj.get("outputs", []))
            all_classes = json_obj.get("all_classes", [])
            length = json_obj.get("length", 0)

            # Truncate prompt to max_length
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = max_length // 2
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                         tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p",
                                "fwe", "vt", "niah_single_3"]:
                prompt = build_chat(tokenizer, prompt)

            input_tensor = tokenizer(prompt, truncation=False, return_tensors="pt")
            input_tensor = _to_device(input_tensor, device)
            context_length = input_tensor.input_ids.shape[-1]

            # Generation
            gen_kwargs = {
                "max_new_tokens": max_gen,
                "num_beams": 1,
                "do_sample": False,
                "eos_token_id": eos_token_ids,
            }

            with torch.no_grad():
                output = model.generate(**input_tensor, **gen_kwargs)[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True).strip()

            json.dump({
                "pred": pred, "answers": answers,
                "all_classes": all_classes, "length": length
            }, fout, ensure_ascii=False)
            fout.write('\n')
            fout.flush()
            torch.cuda.empty_cache()

    # ---- Finalize stats ----
    if duo_recall.current_sample_stats:
        stats_registry.append(duo_recall.current_sample_stats)

    save_recall_csv(output_dir, dataset)


def save_recall_csv(output_dir, dataset_name):
    """Flatten stats_registry into CSV."""
    csv_path = os.path.join(output_dir, f"{dataset_name}_attn_ratios.csv")
    print(f"\n[INFO] Saving recall diagnostics to CSV: {csv_path}")

    rows = []
    total_rows = 0
    for sample_idx, sample_stats in enumerate(stats_registry):
        for step_idx, step_data in enumerate(sample_stats):
            # step_data is a dict: head_idx -> (recall_100, recall_k, selected_attn_ratio, total_len)
            for head_idx, metrics in step_data.items():
                r100, rk, sar, seq_len = metrics
                rows.append([sample_idx, step_idx, head_idx, seq_len,
                             f"{r100:.6f}", f"{rk:.6f}", f"{sar:.6f}"])
                total_rows += 1

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "step_idx", "head_idx", "seq_len",
                         "recall_100", "recall_k", "selected_attn_ratio"])
        for row in rows:
            writer.writerow(row)

    print(f"[SUCCESS] CSV written! Total rows: {total_rows}")
    print(f"  Path: {csv_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    seed_everything(42)

    # Load dataset configs (reuse PyramidKV configs)
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    dataset2prompt = {}
    dataset2maxlen = {}
    for cfg_file in ["dataset2prompt.json", "dataset2maxlen.json"]:
        cfg_path = os.path.join(config_dir, cfg_file)
        if os.path.exists(cfg_path):
            cfg_data = json.load(open(cfg_path, "r"))
            if cfg_file == "dataset2prompt.json":
                dataset2prompt = cfg_data
            else:
                dataset2maxlen = cfg_data

    # Determine dataset name and path
    dataset_name = args.dataset_name
    is_ruler = args.task is not None
    if is_ruler:
        dataset_name = args.task
        dataset_path = os.path.join(args.dataset, dataset_name, "validation.jsonl")
    else:
        if dataset_name is None:
            dataset_name = "qasper"
        dataset_path = os.path.join(args.dataset, dataset_name + ".jsonl")

    prompt_format = dataset2prompt.get(dataset_name, "{input}{answer_prefix}" if is_ruler else "{context}{input}")
    max_len = args.max_length or dataset2maxlen.get(dataset_name, 100000)
    # Use dataset2maxlen for max_gen if not explicitly set
    if args.max_gen == 64 and dataset2maxlen.get(dataset_name, 64) != 64:
        args.max_gen = dataset2maxlen.get(dataset_name, 64)

    print(f"Loading dataset from {dataset_path}...")
    data = [json.loads(line) for line in open(dataset_path, "r", encoding="utf-8") if line.strip()]
    print(f"Loaded {len(data)} samples.")

    output_jsonl, output_dir = get_output_path(args)

    get_pred_single_gpu(
        data, max_len, args.max_gen, prompt_format,
        dataset_name, args.model, output_jsonl, output_dir, args,
    )