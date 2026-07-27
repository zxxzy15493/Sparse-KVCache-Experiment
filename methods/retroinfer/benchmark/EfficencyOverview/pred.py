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
# from model_hub import QwenModel, GlmModel

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
    parser.add_argument("--max_len", type=int, default=1024, help="Length of the context for attention computation")
    parser.add_argument("--data_dir", type=Path, default="", help="Data directory")
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=str, default="LongBench", help="benchmark name, can be LongBench or others, which determines the format of the output jsonl file")
    parser.add_argument("--num_samples", type=int, default=-1, help="num of example to evaluate. -1 for all.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Dtype")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--fixed_output_length", type=int, default=0, required=False)
    parser.add_argument("--recall",action="store_true", required=False)
    parser.add_argument("--warmup", type=int, default=7)
    parser.add_argument("--input_max_token", type=int, default=1024)
    parser.add_argument("--measure_time", action="store_true", required=False)

    parser.add_argument("--monitor_gpu", type=int, default=1, help="whether to monitor GPU stats during inference, 1 to enable, 0 to disable")
    parser.add_argument("--gpu_monitor_interval", type=float, default=0.5, help="GPU monitor sampling interval in seconds")
    parser.add_argument("--gpu_monitor_verbose", type=int, default=0, help="whether to print GPU monitor backend and warning messages")

    parser = parse_attn_args(parser)

    return parser.parse_args(args)

def load_dataset(args):
    # ， data 
    with open("../../../../benchmarks/myinput.txt", "r", encoding="utf-8") as f:
        data = f.read()  # ：read()  data
    return data

def get_pred(llm, data, max_new_tokens, model_name, out_path, args):
        
        if llm.tokenizer.eos_token is not None:
            llm.tokenizer.pad_token = llm.tokenizer.eos_token
        elif llm.tokenizer.pad_token_id is None and len(llm.eos_tokens) > 0:
            llm.tokenizer.pad_token_id = llm.eos_tokens[0]
        llm.tokenizer.padding_side = "left"

        inputs = llm.tokenizer([data], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids[:, :args.input_max_token]
        attention_masks = inputs.attention_mask[:, :args.input_max_token]

        attn_config = generate_config(
            model_name, 
            input_ids.shape[1], 
            args.attn_type,
            budget_ratio=args.budget_ratio,
            budget=args.budget,
            estimate_ratio=args.estimate_ratio,
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
                
        out_path = Path(out_path) / f"EfficencyOverview_{args.input_max_token}_{args.fixed_output_length}_{args.budget_ratio}_{args.estimate_ratio}_{args.budget}.jsonl"
        pred = output[0]
        if args.benchmark == "LongBench":
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump(
                    {
                        "pred": pred, 
                        "answers": data["answers"], 
                        "all_classes": data["all_classes"], 
                        "length": data["length"],
                        "index":data["_id"]
                    }, 
                    f, 
                    ensure_ascii=False
                )
                f.write('\n')
        else:
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump(
                    {
                        "ttft": llm.prefill_latency,
                        "TPOT": llm.TPOT,
                        "decode_latency": llm.decode_latency,
                        "latency": llm.prefill_latency + llm.decode_latency,
                        "pred": pred,
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
            fixed_output_length=args.fixed_output_length,
            measure_time=args.measure_time
            )
    elif 'Qwen' in model_path:
        llm = QwenModel(model_path,
            max_length=max_len + max_new_tokens,
            dtype=dtype,
            device_map=device,
            # RECALL=args.recall,
            fixed_output_length=args.fixed_output_length,
            # measure_time=args.measure_time
            )
    elif 'GLM' in model_path or 'glm' in model_path:
        llm = GlmModel(model_path,
            max_length=max_len + max_new_tokens,
            dtype=dtype,
            device_map=device,
            RECALL=args.recall,
            fixed_output_length=args.fixed_output_length,
            measure_time=args.measure_time
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
    device = args.device
    dtype = torch.bfloat16
    pred_dir = args.save_dir

    data = load_dataset(args)


    model_path = model2path[model_name]
    max_new_tokens = 32
    max_length = args.input_max_token

    llm = load_model(model_path, max_length, dtype, device, max_new_tokens, args)
    for i in range(args.warmup+1):
        get_pred(
            llm,
            data,
            max_new_tokens,
            model_path,
            pred_dir,
            args,
        )
