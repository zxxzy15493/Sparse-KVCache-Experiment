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


from model_recall.llama_recall import LlamaModel
from model_recall.qwen_recall import Qwen2Model

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

    parser.add_argument("--save_dir", type=Path, required=True, help='path to save the prediction jsonl files')
    parser.add_argument("--num_sample", type=int, default=-1)
    parser.add_argument("--model_name", type=str, default='Qwen2.5-7B-Instruct', 
                        help='supported models from OpenAI or HF (provide a key or a local path to the checkpoint)')
    parser.add_argument("--K", type=int, default=9)
    parser.add_argument("--L", type=int, default=200)
    parser.add_argument("--recall", type=int, required=False)
    parser.add_argument("--benchmark", type=str, default="synthetic", choices=["LongBench", "synthetic"])
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--fixed_budget", type=int, default=0, required=False)
    parser.add_argument("--fixed_output_length", type=int, default=0, required=False)

    parser.add_argument("--measure_time", type=int, default=0, required=False)
    parser.add_argument("--max_seq_length", type=int, default=65536, help='max sequence length including all input tokens and generated tokens.')
    args = parser.parse_args()
    return args

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
    datas = data[:args.num_sample] if args.num_sample > 0 else data

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
            model2path[args.model_name],
            trust_remote_code=('GLM' in model2path[args.model_name] or 'glm' in model2path[args.model_name])
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

def load_model(model_name, K, L, batch_size, max_length, device, dtype, args):
    recall = True if args.recall == 1 else False
    fixed_budget = args.fixed_budget
    fixed_output_length = args.fixed_output_length 
    measure_time = True if args.measure_time == 1 else False

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
    inputs = tokenizer([data['input']], return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]
    output = llm.generate(input_ids=input_ids, max_tokens=max_new_tokens)
    
    generated_text = tokenizer.decode(output[seq_len:], skip_special_tokens=True)


    torch.cuda.empty_cache()
    task_key = [key for key in TASKS.keys() if key in args.task][0]
    if args.benchmark == "LongBench":
        pred_dir = os.path.join(pred_dir, f"{args.task}")
    elif args.benchmark == "synthetic":
        pred_dir = os.path.join(pred_dir, f"{task_key}")
    out_path_budget = Path(pred_dir) / f"RecallOverview_{args.task}.jsonl"
    pred = generated_text
    # print(f"-------- pred result is : {pred} --------")
    # pred_dir（）
    os.makedirs(pred_dir, exist_ok=True)
    with open(out_path_budget, "a", encoding="utf-8") as f:
        json.dump(
            {
                "pred": pred,
                "total_step": len(llm.attention_server.decodeStepList_topBudget),
                "decodeStepList": llm.attention_server.decodeStepList_topBudget, 
                "index":data["_id"] if args.benchmark == "LongBench" else data["index"]
            }, 
            f, 
            ensure_ascii=False
        )
        f.write('\n')
    out_path_top100 = Path(pred_dir) / f"RECALLOverview_{args.task}_top100.jsonl"
    with open(out_path_top100, "a", encoding="utf-8") as f:
        json.dump(
            {
                "pred": pred,
                "total_step": len(llm.attention_server.decodeStepList_top100),
                "decodeStepList": llm.attention_server.decodeStepList_top100,
                "index":data["_id"] if args.benchmark == "LongBench" else data["index"]
            }, 
            f, 
            ensure_ascii=False
        )
        f.write('\n')
    return seq_len

def main(args):
    dtype = torch.bfloat16
    pred_dir = args.save_dir
    K = args.K
    L = args.L

    data = load_dataset(args)
    if args.benchmark == "LongBench":
        max_length = model2maxlen[args.model_name]
        model_path = model2path[args.model_name]
        max_new_tokens = dataset2maxlen[args.task]
    elif args.benchmark == "synthetic":
        max_new_tokens = next((TASKS[key] for key in TASKS.keys() if key in args.task ), 0)
        model_path = model2path[args.model_name]

    ### ：
    ###  max_length: kvcache， input_max_token + max_new_tokens
    ###  generation_buffer: kvcache， max_new_tokens
    if args.benchmark == "LongBench":
        llm = load_model(model_path, K, L, 1, max_length, DEVICE, dtype, args)
    else:
        llm = load_model(model_path, K, L, 1, args.max_seq_length + 256, DEVICE, dtype, args)

    avg_ratio = 0
    avg_token = 0
    count = 0
    for data_sample in tqdm(data):
        seq_len = get_pred(
            llm,
            data_sample,
            max_new_tokens,
            pred_dir,
            args,
        )
        avg_ratio += (llm.attention_server.avg_nnz / llm.attention_server.count_nnz / seq_len)
        avg_token += llm.attention_server.avg_nnz / llm.attention_server.count_nnz
        count += 1
        llm.attention_server.avg_nnz = 0
        llm.attention_server.count_nnz = 0


if __name__ == '__main__':
    args = parse_args()
    main(args)
