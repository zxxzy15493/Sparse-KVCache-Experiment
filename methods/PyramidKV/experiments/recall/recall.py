import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Add PyramidKV project root to path so 'pyramidkv' module is importable
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import argparse
import json
import csv
import random
import time
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from pyramid_recall import enable_pyramid_recall
import pyramid_recall  # for current_sample_stats / stats_registry (must use module ref to see reassignments)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--compress_args_path', type=str, default=None, help="Path to the compress args JSON")
    parser.add_argument('--dataset', type=str, default="",
                        help="The folder of dataset to evaluate on")
    parser.add_argument('--dataset_name', type=str, default='qasper',
                        help="The name of dataset to evaluate on")
    parser.add_argument('--task', type=str, default=None,
                        help="RULER task name (e.g. fwe, vt, niah_single_3)")
    return parser.parse_args(args)

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

def reset_sample_cache_state(model, sample_idx=None):
    """Reset per-layer recall state for a new sample."""
    for module in model.modules():
        if hasattr(module, "_recall_enabled"):
            module._recall_step = 0
            module.kv_seq_len = 0  # critical: reset for next sample (Qwen doesn't do this)
            # Clean attention module
            if hasattr(module, "_recall_shadow_key"):
                module._recall_shadow_key = None
            if hasattr(module, "_recall_selected_indices"):
                module._recall_selected_indices = None
            if hasattr(module, "_sel_tensor_cache"):
                module._sel_tensor_cache = None
            # Clean kv_cluster (where _recall_key/_query/_selected_indices live)
            if hasattr(module, "kv_cluster"):
                kv = module.kv_cluster
                for attr in ("_recall_key", "_recall_query", "_selected_indices"):
                    if hasattr(kv, attr):
                        delattr(kv, attr)
                del module.kv_cluster

def set_model_config(model, **params):
    """Set PyramidKV config attributes on model's config object."""
    valid_keys = {"window_size", "max_capacity_prompt", "kernel_size", "pooling", "pyram_beta"}
    for k, v in params.items():
        if k in valid_keys:
            setattr(model.config, k, v)
            print(f"  Set config.{k} = {v}")

def enable_pyramidkv_recall(model, **compress_args):
    """
    Enable PyramidKV recall tracking.

    1. Set model config attributes from compress_args
    2. Run PyramidKV's class-level monkeypatch
    3. Override instance-level attention forwards with recall tracking
    """
    from pyramidkv.monkeypatch import replace_llama, replace_qwen2

    # Step 1: Set config attributes for init_pyramidkv to read
    print("[PyramidKV Recall] Setting config attributes...")
    set_model_config(model, **compress_args)

    # Step 2: PyramidKV class-level monkeypatch (only for matching model type)
    print("[PyramidKV Recall] Applying PyramidKV monkeypatch...")
    replace_llama("pyramidkv")
    if "qwen" in model.config.model_type.lower():
        replace_qwen2("pyramidkv")

    # Step 3: Instance-level recall tracking
    print("[PyramidKV Recall] Applying recall tracking hooks...")
    enable_pyramid_recall(model, check_recall=True, **compress_args)

def build_output_path(model_name, dataset_name, dataset_path, compress_args=None, task=None):
    base_output = "./recall_results"

    # Extract budget from compress_args
    budget_dir = "budget_full"
    if compress_args and 'max_capacity_prompt' in compress_args:
        cap = compress_args['max_capacity_prompt']
        budget_num = cap[0] if isinstance(cap, list) else cap
        budget_dir = f"budget{budget_num}"

    model_dir_name = os.path.basename(model_name.rstrip("/"))

    sub_dir = ""
    if dataset_path and "evict_ruler" in dataset_path:
        sub_dir = os.path.basename(os.path.dirname(dataset_path))

    output_dir = os.path.join(base_output, model_dir_name, budget_dir, sub_dir)
    os.makedirs(output_dir, exist_ok=True)
    filename = task if task else dataset_name
    return os.path.join(output_dir, f"{filename}.jsonl")

@torch.inference_mode()
def get_pred_single_gpu(data, max_length, max_gen, prompt_format, dataset, dataset_name,
                        model_name, model2path, output_path, compress_args=None):
    print(f"Loading model from {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="flash_attention_2"
    ).eval()

    device = model.device
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # Enable PyramidKV recall
    if compress_args is not None:
        enable_pyramidkv_recall(model, **compress_args)

    # Resume support
    done_count = 0
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            done_count = sum(1 for line in f if line.strip())

    remaining_data = data[done_count:]
    if done_count > 0:
        print(f"Resuming from sample {done_count}, {len(remaining_data)} remaining.")

    with open(output_path, "a", encoding="utf-8") as fout:
        for sample_idx, json_obj in enumerate(tqdm(remaining_data), start=done_count):
            reset_sample_cache_state(model, sample_idx=sample_idx)

            if "recall_test" in dataset_path:
                prompt = prompt_format.format(**json_obj)
                answers = json_obj.get("outputs", [])
                all_classes = json_obj.get("all_classes", [])
                length = json_obj.get("length", len(json_obj.get("input", "")))
            elif "Longbench_recall" in dataset_path:
                prompt = prompt_format.format(**json_obj)
                answers = json_obj.get("answers", [])
                all_classes = json_obj.get("all_classes", [])
                length = json_obj.get("length", len(json_obj.get("input", "")))
            else:
                prompt = prompt_format.format(**json_obj)
                answers = json_obj.get("answers", json_obj.get("outputs", []))
                all_classes = json_obj.get("all_classes", [])
                length = json_obj.get("length", 0)

            if "Longbench_recall" in dataset_path and dataset_name not in \
                    ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                prompt = build_chat(tokenizer, prompt)

            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            orig_len = len(tokenized_prompt)
            print(f"Dataset: {dataset_name} | Sample Index: {data.index(json_obj)} | "
                  f"Original Length: {orig_len}")

            if orig_len > max_length:
                half = int(max_length / 2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                         tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

            if "LongBench" in dataset and dataset_name not in \
                    ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                prompt = build_chat(tokenizer, prompt)

            input_tensor = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input_tensor.input_ids.shape[-1]

            gen_kwargs = {
                "max_new_tokens": max_gen, "num_beams": 1, "do_sample": False,
                "min_length": context_length + 1
            }
            if dataset_name == "samsum":
                gen_kwargs["pad_token_id"] = tokenizer.eos_token_id
                gen_kwargs["eos_token_id"] = [
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n", add_special_tokens=False)[-1]
                ]
            else:
                gen_kwargs["temperature"] = 1.0

            _t0 = time.time()
            with torch.no_grad():
                output = model.generate(**input_tensor, **gen_kwargs)[0]
            _t1 = time.time()
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True).strip()
            print(f"  [SAMPLE-TIME] sample {sample_idx}: generate={_t1-_t0:.1f}s output_len={len(output)-context_length}")

            json.dump({
                "pred": pred, "answers": answers,
                "all_classes": all_classes, "length": length
            }, fout, ensure_ascii=False)
            fout.write('\n')
            fout.flush()
            torch.cuda.empty_cache()

            # Save recall stats for this sample
            pyramid_recall.seal_and_save_stats(sample_idx=sample_idx)

    # Finalize: no more samples to save
    pyramid_recall.seal_and_save_stats(sample_idx=None)


def save_recall_csv(output_path):
    """Flatten stats_registry into CSV (same format as HeadKV/CakeKV)."""
    stats_path_csv = output_path.replace(".jsonl", "_attn_ratios.csv")
    print(f"\n[INFO] Saving recall diagnostics to CSV...")

    existing_rows = []
    sample_offset = 0
    if os.path.exists(stats_path_csv):
        with open(stats_path_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row:
                    existing_rows.append(row)
        if existing_rows:
            sample_offset = int(existing_rows[-1][0]) + 1
            print(f"[INFO] Found {len(existing_rows)} existing CSV rows, offset={sample_offset}.")

    try:
        row_count = 0
        with open(stats_path_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "sample_idx", "step_idx", "layer_idx", "head_idx",
                "recall_100", "recall_k", "selected_attn_ratio"
            ])
            for row in existing_rows:
                writer.writerow(row)
                row_count += 1

            for s_idx, entry in enumerate(pyramid_recall.stats_registry):
                # Each entry is (sample_idx, data) tuple
                if isinstance(entry, tuple):
                    stored_sample_idx, sample_data = entry
                else:
                    # Fallback for legacy format
                    stored_sample_idx, sample_data = sample_offset + s_idx, entry

                if stored_sample_idx is None:
                    continue

                if hasattr(sample_data, 'tolist'):
                    sample_data = sample_data.tolist()
                for step_idx, step in enumerate(sample_data):
                    if not isinstance(step, dict):
                        continue
                    for l_idx, matrix in step.items():
                        matrix = np.array(matrix)
                        if matrix.ndim == 2 and matrix.shape[1] == 3:
                            for h_idx in range(matrix.shape[0]):
                                writer.writerow([
                                    stored_sample_idx, step_idx, l_idx, h_idx,
                                    f"{matrix[h_idx, 0]:.6f}",
                                    f"{matrix[h_idx, 1]:.6f}",
                                    f"{matrix[h_idx, 2]:.6f}"
                                ])
                                row_count += 1

        print(f"[SUCCESS] CSV written! Total rows: {row_count}")
        print(f"  Path: {stats_path_csv}")

        npz_path = output_path.replace(".jsonl", "_attn_ratios.npz")
        try:
            np.savez_compressed(npz_path, data=pyramid_recall.stats_registry)
        except Exception as e:
            print(f"[WARN] NPZ save failed (CSV still OK): {e}")

    except Exception as e:
        print(f"[ERROR] CSV export failed: {e}")


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    # Load configs
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    model_name = args.model
    max_length = model2maxlen.get(os.path.basename(model_name.rstrip("/")), 1000000)

    dataset = args.dataset
    dataset_name = args.dataset_name
    task = args.task

    if task:
        dataset_path = os.path.join(dataset, task, "validation.jsonl")
        prompt_format = dataset2prompt.get(task, "{input}{answer_prefix}")
        max_gen = dataset2maxlen.get(task, 128)
    else:
        dataset_path = os.path.join(dataset, dataset_name + ".jsonl")
        prompt_format = dataset2prompt.get(dataset_name, "{context}{input}")
        max_gen = dataset2maxlen.get(dataset_name, 128)

    if args.compress_args_path:
        compress_args = json.load(open(args.compress_args_path, "r"))
    else:
        compress_args = None

    output_path = build_output_path(
        model_name, dataset_name, dataset_path, compress_args, task=task
    )

    data_all = [json.loads(line) for line in open(dataset_path, "r", encoding="utf-8") if line.strip()]
    eval_name = task if task else dataset_name

    get_pred_single_gpu(
        data_all, max_length, max_gen, prompt_format, dataset,
        eval_name, model_name, model2path, output_path, compress_args
    )

    save_recall_csv(output_path)