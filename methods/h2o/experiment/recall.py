import warnings
warnings.filterwarnings("ignore")

import torch
import argparse
import json
import csv
import os
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from h2o_kv.h2o_recall import enable_h2o_recall, stats_registry, init_new_sample_registry

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def reset_sample_cache_state(model):
    init_new_sample_registry()

    for module in model.modules():
        if hasattr(module, "kv_cache") and module.kv_cache is not None:
            module.kv_cache.hh_score = None
            module.kv_cache.select_idx = None
            module.kv_cache.absolute_indices = None
            module.kv_cache.total_processed_tokens = 0
            module.kv_cache._global_step = 0
            module.kv_cache.shadow_key = None

def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    if "glm" in model_name_or_path.lower():
        config_cls = get_class_from_dynamic_module("configuration_chatglm.ChatGLMConfig", model_name_or_path)
        model_cls = get_class_from_dynamic_module("modeling_chatglm.ChatGLMForConditionalGeneration", model_name_or_path)
        config = config_cls.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = model_cls.from_pretrained(
            model_name_or_path, config=config, device_map="auto",
            torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="flash_attention_2"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2"
        )
        
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
            
    model.eval()
    return model, tokenizer

def build_chat(tokenizer, prompt, model_name):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def build_output_path(model_name_or_path, dataset_name, args, dataset_path=None):
    base_output = "./recall_results"
    budget = args.recent_size + args.heavy_hitter_size
    budget_dir = f"budget{budget}"
    model_dir_name = os.path.basename(model_name_or_path.rstrip("/"))
    sub_dir = ""
    if dataset_path and "evict_ruler" in dataset_path:
        parts = dataset_path.split(os.sep)
        try:
            idx = parts.index("evict_ruler")
            if idx + 1 < len(parts): sub_dir = parts[idx + 1]
        except ValueError: pass
            
    output_dir = os.path.join(base_output,budget_dir, model_dir_name, sub_dir)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{dataset_name}.jsonl")

@torch.no_grad()
def streaming_inference_heavy_hitter(model, model_name_or_path, tokenizer, dataset_path, dataset_name, samples, output_path, prompt_format, args, max_gen_len=1000):
    device = model.device
    done_count = 0
    
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): done_count += 1
    
    remaining_samples = samples[done_count:]
    if done_count > 0:
        print(f"Resuming from sample {done_count}, {len(remaining_samples)} remaining.")

    with open(output_path, "a", encoding="utf-8") as fout:
        for sample in tqdm(remaining_samples):
            reset_sample_cache_state(model)
            
            if "recall_test" in dataset_path:
                prompt = prompt_format.format(**sample)
                answers = sample.get("outputs", [])
                all_classes = sample.get("all_classes", [])
                length = sample.get("length", len(sample.get("input", "")))
            elif "Longbench_recall" in dataset_path:
                prompt = prompt_format.format(**sample)
                answers = sample["answers"]
                all_classes = sample["all_classes"]
                length = sample["length"]
                
            if "Longbench_recall" in dataset_path and dataset_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                prompt = build_chat(tokenizer, prompt, model_name=model_name_or_path)
                
            input_tensor = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input_tensor.input_ids.shape[1]
            
            gen_kwargs = {
                "max_new_tokens": max_gen_len,
                "num_beams": 1,
                "do_sample": False,
                "min_length": context_length + 1,
                "num_logits_to_keep": 1,
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
            
    init_new_sample_registry()

def main(args):
    seed_everything(42)
    model, tokenizer = load(args.model_name_or_path)
    
    enable_h2o_recall(model, args)
        
    data_path = os.path.join(args.data_root + ".jsonl")
    print(f"Loading data from {data_path} ...")
    data = load_dataset("json", data_files=data_path, split="train")
    
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    output_path = build_output_path(args.model_name_or_path, args.dataset_name,args, args.data_root)

    streaming_inference_heavy_hitter(
        model=model, model_name_or_path=args.model_name_or_path, tokenizer=tokenizer,
        dataset_path=args.data_root, dataset_name=args.dataset_name, samples=list(data),
        output_path=output_path, prompt_format=dataset2prompt[args.dataset_name],
        args=args, max_gen_len=dataset2maxlen[args.dataset_name],
    )

    stats_path_csv = output_path.replace(".jsonl", "_attn_ratios.csv")
    print(f"\nWriting directly to CSV...")
    
    try:
        row_count = 0
        with open(stats_path_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["sample_idx", "step_idx", "layer_idx", "head_idx", "recall_100", "recall_k", "selected_attn_ratio"])
            
            for s_idx, sample in enumerate(stats_registry):
                if hasattr(sample, 'tolist'):
                    sample = sample.tolist()
                for step_idx, step in enumerate(sample):
                    if not isinstance(step, dict): continue
                    for l_idx, matrix in step.items():
                        matrix = np.array(matrix)
                        if matrix.ndim == 2 and matrix.shape[1] == 3:
                            for h_idx in range(matrix.shape[0]):
                                writer.writerow([
                                    s_idx, step_idx, l_idx, h_idx, 
                                    f"{matrix[h_idx, 0]:.6f}", 
                                    f"{matrix[h_idx, 1]:.6f}", 
                                    f"{matrix[h_idx, 2]:.6f}"
                                ])
                                row_count += 1
                                
        print(f"CSV file generated successfully! Total rows written: {row_count}.")
        print(f"Path: {stats_path_csv}")
        
        np.savez_compressed(output_path.replace(".jsonl", "_attn_ratios.npz"), data=stats_registry)
        
    except Exception as e:
        print(f"CSV generation failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="LongBench")
    parser.add_argument("--dataset_name", type=str, default="cwe")
    parser.add_argument("--heavy_hitter_size", type=int, default=992)
    parser.add_argument("--recent_size", type=int, default=32)
    args = parser.parse_args()

    args.enable_h2o_cache = True
    args.check_recall = True
    main(args)