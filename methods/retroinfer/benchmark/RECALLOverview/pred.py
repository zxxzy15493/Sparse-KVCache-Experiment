import os
import sys
import torch
import json
from tqdm import tqdm
import numpy as np
import random
from pathlib import Path
import threading
import subprocess
import argparse
from utils import load_data
import time
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)
from model_hub import LlamaModel, QwenModel, GlmModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import generate_config, parse_attn_args

model2path = json.load(open("./config/model2path.json", "r"))
model2maxlen = json.load(open("./config/model2maxlen.json", "r"))
# we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
dataset2prompt = json.load(open("./config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("./config/dataset2maxlen.json", "r"))
import json
from typing import List, Dict

TASKS = {
    'niah': 128,
    'vt': 30,
    'cwe': 120,
    'fwe': 50,
    'qa': 32
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument("--attn_type", type=str, default="Full_Flash_Attn",                                                     \
                        choices=["Full_Flash_Attn", "RetroInfer"],                          \
                        help="Attention method")
    parser.add_argument("--prefill_method", type=str, default="Full_Flash_Attn",                                                     \
                        choices=["Full_Flash_Attn", "minfer"],                          \
                        help="Attention method for prefill phase, which determines the attention method used during the prefill phase. When set to 'minfer', it will use the minfer method with the best patterns for each layer. When set to 'Full_Flash_Attn', it will use full flash attention during the prefill phase.")
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Dtype")
    parser.add_argument("--benchmark", type=str, default="LongBench", help="benchmark name, can be LongBench or others, which determines the format of the output jsonl file")
    parser.add_argument("--num_samples", type=int, default=-1, help="num of example to evaluate. -1 for all.")
    parser.add_argument('--task', type=str, required=True, help="task name. work when --e is false")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--fixed_output_length", type=int, default=0, required=False)
    parser.add_argument("--recall",action="store_true", required=False)
    parser.add_argument("--measure_time", action="store_true", required=False)
    
    parser = parse_attn_args(parser)

    return parser.parse_args(args)

def load_dataset(args):
    # load data
    if args.benchmark == "LongBench":
        task_file = f'../../../../benchmarks/Longbench_recall/{args.task}.jsonl'
    elif args.benchmark == "synthetic":
        if 'llama' in args.model_name.lower():
            task_file = f'../../../../benchmarks/Ruler_recall/llama-3.1-8b/synthetic/65536/data/{args.task}/validation.jsonl'
        elif 'qwen' in args.model_name.lower():
            task_file = f'../../../../benchmarks/Ruler_recall/qwen-2.5-7b-1m/synthetic/65536/data/{args.task}/validation.jsonl'

    prefix = args.save_dir
    task_key = [key for key in TASKS.keys() if key in args.task][0]
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    if args.benchmark == "LongBench":
        out_path = f"{prefix}/{args.task}/RECALLOverview_{args.task}_top100.jsonl"
    elif args.benchmark == "synthetic":
        out_path = f"{prefix}/{task_key}/RECALLOverview_{args.task}_top100.jsonl"

    data = [sample for sample in load_data(task_file)]
    datas = data[:args.num_samples] if args.num_samples > 0 else data

    if os.path.exists(out_path):
        pred_index = [sample["index"] for sample in load_data(out_path)]
        if args.benchmark == "LongBench":
            data = [sample for sample in datas if sample["_id"] not in pred_index]
        elif args.benchmark == "synthetic":
            data = [sample for sample in datas if sample["index"] not in pred_index]
    else:
        data = datas

    if args.benchmark == "LongBench":
        tokenizer = AutoTokenizer.from_pretrained(
            model2path[model_name],
            trust_remote_code=('GLM' in model2path[model_name] or 'glm' in model2path[model_name])
        )
        prompt_format = dataset2prompt[args.task]
        for i in range(len(data)):
            json_obj = data[i]
            prompt = prompt_format.format(**json_obj)
            if args.task not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:  
                message = [{"role": "user", "content": prompt}]
                prompt = tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True  # assistant
                )
            data[i]["input"] = prompt
    return data

def get_pred(llm, data, max_new_tokens, model_name, out_path, args):
        
        if llm.tokenizer.eos_token is not None:
            llm.tokenizer.pad_token = llm.tokenizer.eos_token
        elif llm.tokenizer.pad_token_id is None and len(llm.eos_tokens) > 0:
            llm.tokenizer.pad_token_id = llm.eos_tokens[0]
        llm.tokenizer.padding_side = "left"

        inputs = llm.tokenizer([data['input']], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        attention_masks = inputs.attention_mask
        print("="*80)
        print("="*80)
        attn_config = generate_config(
            model_name, 
            input_ids.shape[1], 
            args.attn_type,
            budget_ratio=args.budget_ratio,
            budget=args.budget,
            estimate_ratio=args.estimate_ratio,
            # RECALL=1 if args.recall else 0,
            ratio_or_fixed=args.ratio_or_fixed,
        )

        out = llm.generate(
            attention_type=args.attn_type,
            inputs_ids = input_ids.to(llm.layers[0].device),
            attention_masks = attention_masks.to(llm.layers[0].device),
            max_new_length=max_new_tokens, 
            attn_config=attn_config,
            prefill_method=args.prefill_method
        )

        print("="*80)
        print("="*80)
        
        output = llm.tokenizer.batch_decode(out, skip_special_tokens=True)
        
        torch.cuda.empty_cache()
        
        task_key = [key for key in TASKS.keys() if key in args.task][0]
        if args.benchmark == "LongBench":
            out_path = os.path.join(out_path, f"{args.task}")
        elif args.benchmark == "synthetic":
            out_path = os.path.join(out_path, f"{task_key}")
        os.makedirs(out_path, exist_ok=True)

        out_path_budget = Path(out_path) / f"RECALLOverview_{args.task}_{args.budget}.jsonl"
        pred = output[0]
        with open(out_path_budget, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": pred,
                    "total_step": len(llm.kv_cache.decodeStepList_topBudget) if hasattr(llm, "kv_cache") else None,
                    "decodeStepList": llm.kv_cache.decodeStepList_topBudget if hasattr(llm, "kv_cache") else None,
                    "index":data["_id"] if args.benchmark == "LongBench" else data["index"]
                }, 
                f, 
                ensure_ascii=False
            )
            f.write('\n')
        
        out_path_top100 = Path(out_path) / f"RECALLOverview_{args.task}_top100.jsonl"
        with open(out_path_top100, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": pred,
                    "total_step": len(llm.kv_cache.decodeStepList_top100) if hasattr(llm, "kv_cache") else None,
                    "decodeStepList": llm.kv_cache.decodeStepList_top100 if hasattr(llm, "kv_cache") else None,
                    "index":data["_id"] if args.benchmark == "LongBench" else data["index"]
                }, 
                f, 
                ensure_ascii=False
            )
            f.write('\n')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model(model_path, max_len, dtype, device, max_new_tokens, args):
    if 'Llama' in model_path:
        llm = LlamaModel(model_path,
            max_length=max_len + max_new_tokens,
            dtype=dtype,
            device_map=device,
            RECALL=args.recall,
            budget=args.budget
            )
    elif 'Qwen' in model_path:
        llm = QwenModel(model_path,
            max_length=max_len + max_new_tokens,
            dtype=dtype,
            device_map=device,
            RECALL=args.recall,
            budget=args.budget
            )
    elif 'GLM' in model_path or 'glm' in model_path:
        llm = GlmModel(model_path,
            max_length=max_len + max_new_tokens,
            dtype=dtype,
            device_map=device,
            RECALL=args.recall,
            )
    else:
        raise ValueError(f"Unsupported model: {model_path}")

    if llm.tokenizer.eos_token is not None:
        llm.tokenizer.pad_token = llm.tokenizer.eos_token
    elif llm.tokenizer.pad_token_id is None and len(llm.eos_tokens) > 0:
        llm.tokenizer.pad_token_id = llm.eos_tokens[0]
    llm.tokenizer.padding_side = "left"
    
    return llm

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    model_name = args.model_name 
    dataset = args.task
    device = args.device
    dtype = torch.bfloat16
    pred_dir = args.save_dir

    data = load_dataset(args)

    if args.benchmark == "LongBench":
        max_length = model2maxlen[model_name]
        model_path = model2path[model_name]
        max_new_tokens = dataset2maxlen[dataset]
    elif args.benchmark == "synthetic":
        max_new_tokens = next((TASKS[key] for key in TASKS.keys() if key in dataset ), 0)
        model_path = model2path[model_name]
        max_length = 65536 + max_new_tokens


    llm = load_model(model_path, max_length, dtype, device, max_new_tokens, args)
    for data_sample in tqdm(data):
            get_pred(
                llm,
                data_sample,
                max_new_tokens,
                model_path,
                pred_dir,
                args,
            )
