import argparse
import json
import os
import subprocess
import sys
import threading
import time
from tqdm import tqdm
from pathlib import Path
from utils import load_data
import torch
import torch.distributed as dist

_MAGICPIG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _MAGICPIG_ROOT not in sys.path:
    sys.path.insert(0, _MAGICPIG_ROOT)

from models_single.llama import LlamaModel
from models_single.qwen import Qwen2Model

from transformers import AutoTokenizer

dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)
DEVICE = torch.device("cuda", local_rank)

TASKS = {
    'niah': 128,
    'vt': 30,
    'cwe': 120,
    'fwe': 50,
    'qa': 32
}

dataset2prompt = json.load(open("./config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("./config/dataset2maxlen.json", "r"))
model2path = json.load(open("./config/model2path.json", "r"))
model2maxlen = json.load(open("./config/model2maxlen.json", "r"))

def is_glm_model(model_name: str) -> bool:
    return "glm" in model_name.lower()


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=is_glm_model(model_name))

def parse_args(args=None):
    parser = argparse.ArgumentParser()
     # Data
    parser.add_argument("--data_dir", type=Path, required=True, help='path to load the dataset jsonl files')
    parser.add_argument("--save_dir", type=Path, required=True, help='path to save the prediction jsonl files')
    parser.add_argument("--num_sample", type=int, default=-1)
    parser.add_argument("--model_name", type=str, default='Qwen2.5-7B-Instruct', 
                        help='supported models from OpenAI or HF (provide a key or a local path to the checkpoint)')
    parser.add_argument("--K", type=int, default=9)
    parser.add_argument("--L", type=int, default=200)
    parser.add_argument("--recall", type=int, required=False)
    parser.add_argument("--fixed_budget", type=int, default=0, required=False)
    parser.add_argument("--fixed_output_length", type=int, default=0, required=False)

    
    parser.add_argument("--measure_time", type=int, default=0, required=False)
    parser.add_argument("--max_seq_length", type=int, default=32768, help='max sequence length including all input tokens and generated tokens.')


    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--input_max_token", type=int, default=1024)
    
    args = parser.parse_args()
    return args

def load_dataset(args):
    with open("../../../../benchmarks/myinput.txt", "r", encoding="utf-8") as f:
        data = f.read()  # ：read()  data
    return data

def load_model(model_name, K, L, batch_size, max_length, device, dtype, args):
    recall = True if args.recall == 1 else False
    fixed_output_length = args.fixed_output_length 

    if 'llama' in model_name.lower():
        llm = LlamaModel(model_name=model_name, 
                  K=K, 
                  L=L, 
                  batch_size=batch_size,
                max_length=max_length, 
                device=device, 
                dtype=dtype,
                RECALL=recall,
                fixed_output_length=fixed_output_length,
                )
    elif 'qwen' in model_name.lower():
        llm = Qwen2Model(model_name=model_name, 
                  K=K, 
                  L=L, 
                  batch_size=batch_size,
                max_length=max_length, 
                device=device, 
                dtype=dtype,
                RECALL=recall,
                fixed_output_length=fixed_output_length,
              )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return llm

def get_pred(llm, data, max_new_tokens, pred_dir, args):

    tokenizer = load_tokenizer(model2path[args.model_name])

    inputs = tokenizer([data], return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids[:, :args.input_max_token]
    print("="*80)
    print("="*80)

    output = llm.generate(input_ids=input_ids, max_tokens=max_new_tokens)
    generated_text = tokenizer.decode(output[args.input_max_token:], skip_special_tokens=True)
    print("="*80)
    print("="*80)

    torch.cuda.empty_cache()
    out_path = Path(pred_dir) / f"EfficencyOverview_{args.input_max_token}_{args.K}_{args.L}.jsonl"
    pred = generated_text
    # print(f"-------- pred result is : {pred} --------")
    
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

def main(args):
    
    dtype = torch.bfloat16
    pred_dir = args.save_dir
    K = args.K
    L = args.L

    data = load_dataset(args)
    
    model_path = model2path[args.model_name]

    max_new_tokens = 32

    
    llm = load_model(model_path, K, L, 1, args.input_max_token + max_new_tokens + 256, DEVICE, dtype, args)

    avg_ratio = 0
    avg_token = 0
    count = 0
    for i in range(args.warmup+1):
        get_pred(
            llm,
            data,
            max_new_tokens,
            pred_dir,
            args,
        )
        avg_ratio += (llm.attention_server.avg_nnz / llm.attention_server.count_nnz / args.input_max_token)
        avg_token += llm.attention_server.avg_nnz / llm.attention_server.count_nnz
        count += 1
        llm.attention_server.avg_nnz = 0
        llm.attention_server.count_nnz = 0


if __name__ == '__main__':
    args = parse_args()
    main(args)
