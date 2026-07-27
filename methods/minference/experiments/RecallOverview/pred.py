import argparse
import json
from pathlib import Path
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM,  GenerationConfig
import sys
import torch
import os
from utils import load_data
from tqdm import tqdm
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from minference_recall import MInference
from minference_recall import recall_json

model2path = json.load(open("./config/model2path.json", "r"))
model2maxlen = json.load(open("./config/model2maxlen.json", "r"))
# we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
dataset2prompt = json.load(open("./config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("./config/dataset2maxlen.json", "r"))

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
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=str, default="LongBench", help="benchmark name, can be LongBench or others, which determines the format of the output jsonl file")
    parser.add_argument('--task', type=str, required=True, help="task name. work when --e is false")
    parser.add_argument('--num_samples', type=int, default=-1)
    return parser.parse_args(args)

def load_dataset(args):
    model_name = args.model_name
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

    out_path = f"{prefix}/RecallOverview_{args.task}.jsonl"


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

def get_pred(llm, data, model_name, max_new_tokens, pred_dir, args):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer([data['input']], return_tensors="pt", return_attention_mask=True).to("cuda:0")
    seq_len = inputs.input_ids.shape[1]
    print(f"input_ids length: {seq_len}")
    out = llm.generate(**inputs, max_new_tokens=max_new_tokens, temperature=1.0, top_p=1.0, repetition_penalty=1.0)

    pred = tokenizer.decode(out[0][seq_len:], skip_special_tokens=True)

    pred_dir = Path(pred_dir) / f'RecallOverview_{args.task}.jsonl'
    with open(pred_dir, "a", encoding="utf-8") as f:
        json.dump(
            {
                "pred": pred, 
                "recall": recall_json,
                "index":data["_id"] if args.benchmark == "LongBench" else data["index"]
            }, 
            f, 
            ensure_ascii=False
        )
        f.write('\n')
    recall_json.clear()
    print(pred)
    
    torch.cuda.empty_cache()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def get_llm(model_path): 
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map="cuda:0",_attn_implementation="flash_attention_2")
    minference_patch = MInference(model_path)

    model = minference_patch(model, RECALL=True)

    return model


def main():
    
    seed_everything(42)
    args = parse_args()

    model_name = args.model_name 
    dataset = args.task
    pred_dir = args.save_dir

    data = load_dataset(args)

    if args.benchmark == "LongBench":
        model_path = model2path[model_name]
        max_new_tokens = dataset2maxlen[dataset]
    elif args.benchmark == "synthetic":
        max_new_tokens = next((TASKS[key] for key in TASKS.keys() if key in dataset ), 0)
        model_path = model2path[model_name]

    llm = get_llm(model_path)
    for data_sample in tqdm(data):
            get_pred(
                llm,
                data_sample,
                model_path,
                max_new_tokens,
                pred_dir,
                args,
            )

if __name__ == "__main__":
    main()
