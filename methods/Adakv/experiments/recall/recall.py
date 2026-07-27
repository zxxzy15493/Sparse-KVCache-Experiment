import os
import warnings
warnings.filterwarnings("ignore")

import torch
import argparse
import json
import csv
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from headkv_recall import enable_headkv_recall, stats_registry, init_new_sample_registry

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--compress_args_path', type=str, default=None, help="Path to the compress args")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--dataset', type=str, default=None, help="The folder of dataset to evaluate on")
    parser.add_argument('--dataset_name', type=str, default='qasper', help="The name of dataset to evaluate on")
    parser.add_argument('--method', type=str, default='adativekv', help='Method: adativekv or reasonkv')
    parser.add_argument('--task', type=str, default=None, help="RULER task name (e.g. fwe, vt, niah_single_3) for directory-structured data")
    return parser.parse_args(args)

def build_chat(tokenizer, prompt):
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def reset_sample_cache_state(model):
    """seal stats and clear variable-length shadow tensors before each sample to prevent memory accumulation"""
    init_new_sample_registry()
    for module in model.modules():
        if hasattr(module, "_recall_shadow_key"):
            module._recall_shadow_key = None
            module._recall_selected = None
            module._recall_step = 0
            module.kv_seq_len = 0

def build_output_path(model_name, dataset_name, dataset_path, compress_args=None, task=None, method=None):
    base_output = "./recall_results"

    # dynamically capture Budget: if a list, use first element as directory label
    budget_dir = "budget_full"
    if compress_args:
        caps = compress_args.get('base_capacity') or compress_args.get('max_capacity_prompts')
        if caps is not None:
            budget_num = caps[0] if isinstance(caps, list) else caps
            budget_dir = f"budget{budget_num}"
    model_dir_name = os.path.basename(model_name.rstrip("/"))
    sub_dir = ""
    if dataset_path and "evict_ruler" in dataset_path:
        sub_dir = os.path.basename(os.path.dirname(dataset_path))

    m = method if method else (compress_args.get('method', 'adativekv') if compress_args else 'adativekv')
    output_dir = os.path.join(base_output, m, model_dir_name, budget_dir, sub_dir)
    os.makedirs(output_dir, exist_ok=True)
    filename = task if task else dataset_name
    return os.path.join(output_dir, f"{filename}.jsonl")

@torch.inference_mode()
def get_pred_single_gpu(data, max_length, max_gen, prompt_format, dataset, dataset_name, model_name, model2path, output_path, compress_args=None):
    print(f"Loading model from {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="flash_attention_2"
    ).eval()

    device = model.device
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # CakeKV Recall hook
    if compress_args is not None:
        enable_headkv_recall(model, check_recall=True, **compress_args)

    done_count = 0
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            done_count = sum(1 for line in f if line.strip())

    remaining_data = data[done_count:]
    if done_count > 0:
        print(f"Resuming from sample {done_count}, {len(remaining_data)} remaining.")

    with open(output_path, "a", encoding="utf-8") as fout:
        for json_obj in tqdm(remaining_data):
            reset_sample_cache_state(model)

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

            if "Longbench_recall" in dataset_path and dataset_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                prompt = build_chat(tokenizer, prompt)

            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            print(f"Dataset: {dataset_name} | Sample Index: {data.index(json_obj)} | Original Length: {len(tokenized_prompt)}")

            if len(tokenized_prompt) > max_length:
                half = int(max_length / 2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

            if "LongBench" in dataset and dataset_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                prompt = build_chat(tokenizer, prompt)

            input_tensor = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input_tensor.input_ids.shape[-1]

            gen_kwargs = {
                "max_new_tokens": max_gen, "num_beams": 1, "do_sample": False, "min_length": context_length + 1
            }
            if dataset_name == "samsum":
                gen_kwargs["pad_token_id"] = tokenizer.eos_token_id
                gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]]
            else:
                gen_kwargs["temperature"] = 1.0

            output = model.generate(**input_tensor, **gen_kwargs)[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True).strip()

            json.dump({"pred": pred, "answers": answers, "all_classes": all_classes, "length": length}, fout, ensure_ascii=False)
            fout.write('\n')
            fout.flush()
            torch.cuda.empty_cache()

    init_new_sample_registry()

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    model_name = args.model
    max_length = 1000000

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    dataset = args.dataset
    dataset_name = args.dataset_name
    task = args.task

    if task:
        # RULER directory structure: {dataset}/{task}/validation.jsonl
        dataset_path = os.path.join(dataset, task, "validation.jsonl")
        prompt_format = dataset2prompt.get(task, "{input}{answer_prefix}")
        max_gen = dataset2maxlen.get(task, 128)
    else:
        dataset_path = os.path.join(dataset, dataset_name + ".jsonl")
        prompt_format = dataset2prompt[dataset_name]
        max_gen = dataset2maxlen[dataset_name]

    if args.compress_args_path:
        compress_args = json.load(open(args.compress_args_path, "r"))
    else:
        compress_args = None

    output_path = build_output_path(model_name, dataset_name, dataset_path, compress_args, task=task, method=args.method)

    data_all = [json.loads(line) for line in open(dataset_path, "r", encoding="utf-8") if line.strip()]

    eval_name = task if task else dataset_name
    get_pred_single_gpu(data_all, max_length, max_gen, prompt_format, dataset, eval_name, model_name, model2path, output_path, compress_args)

    # ==========================================
    # flatten data to disk: expand directly into standard 2D CSV table
    # ==========================================
    stats_path_csv = output_path.replace(".jsonl", "_attn_ratios.csv")
    print(f"\n[INFO] Flattening and saving diagnostic results to CSV...")

    # read existing CSV rows (preserve data on resume)
    existing_rows = []
    sample_offset = 0
    if os.path.exists(stats_path_csv):
        with open(stats_path_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header
            for row in reader:
                if row:
                    existing_rows.append(row)
        if existing_rows:
            sample_offset = int(existing_rows[-1][0]) + 1
            print(f"[INFO] Found {len(existing_rows)} existing CSV rows, new samples offset by {sample_offset}.")

    try:
        row_count = 0
        with open(stats_path_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["sample_idx", "step_idx", "layer_idx", "head_idx", "recall_100", "recall_k", "selected_attn_ratio"])

            # write existing rows first
            for row in existing_rows:
                writer.writerow(row)
                row_count += 1

            # then write new rows, with sample_idx offset
            for s_idx, sample in enumerate(stats_registry):
                if hasattr(sample, 'tolist'):
                    sample = sample.tolist()
                for step_idx, step in enumerate(sample):
                    if not isinstance(step, dict):
                        continue
                    for l_idx, matrix in step.items():
                        matrix = np.array(matrix)
                        if matrix.ndim == 2 and matrix.shape[1] == 3:
                            for h_idx in range(matrix.shape[0]):
                                writer.writerow([
                                    sample_offset + s_idx, step_idx, l_idx, h_idx,
                                    f"{matrix[h_idx, 0]:.6f}", f"{matrix[h_idx, 1]:.6f}", f"{matrix[h_idx, 2]:.6f}"
                                ])
                                row_count += 1

        print(f"[SUCCESS] CSV file written successfully! Total rows: {row_count} ({len(existing_rows)} existing + {row_count - len(existing_rows)} new).")
        print(f"Target Path: {stats_path_csv}")

        try:
            np.savez_compressed(output_path.replace(".jsonl", "_attn_ratios.npz"), data=stats_registry)
        except Exception as e2:
            print(f"[WARN] NPZ save failed (CSV still OK): {e2}")

    except Exception as e:
        print(f"[ERROR] CSV exporting failed: {e}")