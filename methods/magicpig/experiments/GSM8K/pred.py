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
import re
import torch.distributed as dist
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from models_deepseek.deepseek_dist import DeepSeekModel
from transformers import AutoTokenizer
from examples import get_examples


dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)
DEVICE = torch.device("cuda", local_rank)

model2path = json.load(open("./config/model2path.json", "r"))
model2maxlen = json.load(open("./config/model2maxlen.json", "r"))
# we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
dataset2prompt = json.load(open("./config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("./config/dataset2maxlen.json", "r"))
import json

TASKS = {
    'niah': 128,
    'vt': 30,
    'cwe': 120,
    'fwe': 50,
    'qa': 32
}

EXAMPLES = get_examples()

def load_prompt(prompt_name, num_shots):
    if not num_shots:
        return []
    return EXAMPLES[prompt_name][:num_shots]

def construct_prompt(example, args):
    demos = load_prompt('gsm8k-cot', 8)
    demo_prompt = "".join(
        [
            q + "\n" + a
            for q, a in demos
        ]
    )
    return demo_prompt + "\nQuestion: " + example["question"] + "\n"

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=Path, required=True, help='path to save the prediction jsonl files')
    parser.add_argument("--num_sample", type=int, default=-1)
    parser.add_argument("--model_name", type=str, default='', 
                        help='supported models from OpenAI or HF (provide a key or a local path to the checkpoint)')
    parser.add_argument("--K", type=int, default=9)
    parser.add_argument("--L", type=int, default=200)
    parser.add_argument("--recall", type=int, required=False)
    parser.add_argument("--fixed_output_length", type=int, default=0, required=False)

    parser.add_argument("--measure_time", type=int, default=0, required=False)
    parser.add_argument("--max_seq_length", type=int, default=32768, help='max sequence length including all input tokens and generated tokens.')
    args = parser.parse_args()
    return args

def load_dataset(dataset, pred_dir, prompt_cot=None):
    # load data
    data_file = f'./data/{dataset}.jsonl'
    datas = load_data(data_file)
    for i, data in enumerate(datas):
        data.setdefault('index', i)

    out_path = Path(pred_dir) / "gsm8k.jsonl"

    if os.path.exists(out_path):
        pred_index = [sample["index"] for sample in load_data(out_path)]
        data = [sample for sample in datas if sample["index"] not in pred_index]
    else:
        data = datas

    return data

def get_pred(llm, message, data, max_new_tokens, model_name, out_path, args):  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = [{"role": "user", "content": message}]
    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    if isinstance(inputs, torch.Tensor):
        input_ids = inputs
    else:
        input_ids = inputs["input_ids"]
    input_ids = input_ids.to(DEVICE)
    seq_len = input_ids.shape[1]
    output = llm.generate(input_ids=input_ids, max_tokens=max_new_tokens)
    
    print(f"\noutput length is {len(output) - seq_len}")
    generated_text = tokenizer.decode(output[seq_len:], skip_special_tokens=True)


    torch.cuda.empty_cache()
    out_path = Path(pred_dir) / f"gsm8k.jsonl"
    pred = generated_text

    pattern = r'<answer>(.*?)</answer>'

    match_answer = re.search(pattern, pred)
    if match_answer:
        match_answer = match_answer.group(1)

    pattern_final_answer = r"#### (\d{1,3}(?:,\d{3})*(?:\.?\d+)?)"
    final_answer = re.search(pattern_final_answer, data['answer'])
    if final_answer:
        final_answer = final_answer.group(1)

    with open(out_path, "a", encoding="utf-8") as f:
        json.dump(
            {
                "index": data.get("index"),
                "match_result": match_answer,
                "final_answer": final_answer,
                "pred": pred,
            }, 
            f, 
            ensure_ascii=False
        )
        f.write('\n')
    return seq_len

def load_model(model_name, K, L, batch_size, max_length, max_new_tokens, device, dtype, args):
    recall = True if args.recall == 1 else False
    fixed_output_length = args.fixed_output_length
    measure_time = True if args.measure_time == 1 else False
    if 'deepseek' in model_name.lower():
        llm = DeepSeekModel(model_name=model_name,
                  K=K,
                  L=L,
                  batch_size=batch_size,
                max_length=max_length,
                num_sink_tokens=16,
                num_local_tokens=32,
                generation_buffer=max_new_tokens,
                device=device,
                dtype=dtype,
                RECALL=recall,
                fixed_output_length=fixed_output_length,
                measure_time=measure_time
              )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return llm

if __name__ == "__main__":  
    args = parse_args()
    dtype = torch.bfloat16
    pred_dir = args.save_dir
    K = args.K
    L = args.L

    datasets = "gsm8k_test"
    data = load_dataset(datasets, pred_dir)
    
    model_path = model2path[args.model_name]
    max_new_tokens = 3000
    llm = load_model(model_path, K, L, 1, 1000+max_new_tokens, max_new_tokens, "cuda:0", dtype, args)

    avg_ratio = 0
    avg_token = 0
    count = 0
    for data_sample in tqdm(data):
        message = construct_prompt(data_sample, args)
        if data_sample['index'] in [20,46,87,99]:
            seq_len = get_pred(
                llm,
                message,
                data_sample,
                max_new_tokens,
                model_path,
                pred_dir,
                args,
            )
            print(f"\n Average utilized token count for this sample: {llm.attention_server.avg_nnz / llm.attention_server.count_nnz if llm.attention_server.count_nnz > 0 else 0} \n")
            avg_ratio += (llm.attention_server.avg_nnz / llm.attention_server.count_nnz / seq_len) if llm.attention_server.count_nnz > 0 else 0
            avg_token += llm.attention_server.avg_nnz / llm.attention_server.count_nnz if llm.attention_server.count_nnz > 0 else 0
            count += 1
            llm.attention_server.avg_nnz = 0
            llm.attention_server.count_nnz = 0

    print(f"\n Average utilized token count: {avg_token / count if count > 0 else 0} \n")
    print(f"\n Average token utilization ratio: {avg_ratio / count if count > 0 else 0} \n")
