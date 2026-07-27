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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./")))
from models.llama_dist_time import LlamaModel
from models.qwen_dist_time import Qwen2Model
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

dataset2prompt = json.load(open("../LongBench/config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("../LongBench/config/dataset2maxlen.json", "r"))
model2path = json.load(open("../LongBench/config/model2path.json", "r"))
model2maxlen = json.load(open("../LongBench/config/model2maxlen.json", "r"))

def time_message(llm):
    print(f"="*100)
    print(f"Total prefill latency: {llm.prefill_latency}s")
    print(f"Total decode latency: {llm.decode_latency}s")
    print(f"="*100)
    print(f"Prefill latency breakdown")
    print(f"     - LLM")
    print(f"        - outer_layer_prefill_time is {llm.outer_layer_prefill_time}s")
    print(f"        - inner_layer_prefill_time is {llm.inner_layer_prefill_time}s")
    print(f"        - prefill attention computation used {llm.prefill_attn_time}s")
    print(f"        - prefill attention computation used {llm.prefill_attn_time_old}s measured by old method")
    print(f"        - prefill feed-forward computation used {llm.prefill_ffn_time}s")
    print(f"        - prefill feed-forward computation used {llm.prefill_ffn_time_old}s measured by old method")
    print(f"     - attenServer")
    print(f"        - unload operation used {llm.unload_time}s")
    print(f"        - unload operation used {llm.unload_time_old}s measured by old method")
    print(f"        - construct operation used {llm.construct_time}s")
    print(f"        - construct operation used {llm.construct_time_old}s measured by old method")
    print(f"        - build table operation used {llm.build_table_time}s")
    print(f"        - build table operation used {llm.build_table_time_old}s measured by old method")

    print(f"="*100)
    print(f"Total prefill latency: {llm.prefill_latency}s")
    print(f"Total decode latency: {llm.decode_latency}s")
    print(f"="*100)
    print(f"Decode latency breakdown")
    print(f"     - LLM")
    print(f"        - pre attention computation operation used {llm.pre_attention_compute_time}s")
    print(f"        - post attention computation operation used {llm.post_attention_compute_time}s")
    print(f"        - inner decode method used {llm.inner_decode_method_time}s")
    print(f"        - outer decode method used {llm.outer_decode_method_time}s")
    print(f"        - decode attention used (cpu + gpu + merge) {llm.decode_attn_time}s")
    print(f"        - decode attention used (cpu + gpu + merge) {llm.decode_attn_time_old}s measured by old method")
    print(f"        - retrieve operation used {llm.retrieve_time}s")
    print(f"     - attenServer")
    print(f"        - decode time except merge used {llm.attention_server.decode_time_except_merge}s")
    print(f"        - decode time except merge and cpu used {llm.attention_server.decode_time_except_merge_cpu}s")
    print(f"        - decode time except merge, cpu and fixed budget used {llm.attention_server.decode_time_except_merge_cpu_fix_budge}s")
    print(f"        - decode time except merge, cpu, fixed budget and retrieve used {llm.attention_server.decode_time_except_merge_cpu_fix_budge_retrieve}s")
    print(f"        - cpu decode attention used {llm.cpu_decode_attn_time}s")
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
    fixed_budget = args.fixed_budget
    fixed_output_length = args.fixed_output_length 
    measure_time = True if args.measure_time == 1 else False

    if 'llama' in model_name.lower():
        llm = LLM(model_name=model_name, 
                  K=K, 
                  L=L, 
                  batch_size=batch_size,
                max_length=max_length, 
                device=device, 
                dtype=dtype,
                RECALL=recall,
                fixed_budget=fixed_budget,
                fixed_output_length=fixed_output_length,
                measure_time=measure_time
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
                fixed_budget=fixed_budget,
                fixed_output_length=fixed_output_length,
                measure_time=measure_time
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
    args.warmup = 0
    for i in range(args.warmup+1):
        get_pred(
            llm,
            data,
            max_new_tokens,
            pred_dir,
            args,
        )
        llm.attention_server.avg_nnz = 0
        llm.attention_server.count_nnz = 0
    time_message(llm)
        
if __name__ == '__main__':
    args = parse_args()
    main(args)
