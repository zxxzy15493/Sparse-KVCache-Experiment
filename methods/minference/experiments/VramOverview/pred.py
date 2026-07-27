import argparse
import json
from pathlib import Path
import sys
from xml.parsers.expat import model
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import yaml
import numpy as np
import random
import time
from tqdm import tqdm
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from minference import MInference
import torch
    

model2path = json.load(open("./config/model2path.json", "r"))
model2maxlen = json.load(open("./config/model2maxlen.json", "r"))
# we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
dataset2prompt = json.load(open("./config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("./config/dataset2maxlen.json", "r"))

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--save_dir",type=Path,required=True,help="path to save the prediction jsonl files",)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--fixed_output_length", type=int, default=0, required=False)
    parser.add_argument("--input_max_token", type=int, default=1024)
    # MInference
    parser.add_argument("--model_name", type=str, required=True, help="model name or path for MInferenceModel")

    return parser.parse_args()

def load_dataset(args):
    with open("../../../../benchmarks/myinput.txt", "r", encoding="utf-8") as f:
        data = f.read()  # ：read()  data
    return data

def get_pred(llm, data, model_name, pred_dir, args):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(data, return_tensors="pt", return_attention_mask=True).to(llm.device)

    inputs['input_ids'] = inputs.input_ids[:, :args.input_max_token]
    if "attention_mask" in inputs:
        inputs['attention_mask'] = inputs.attention_mask[:, :args.input_max_token]

    seq_len = inputs.input_ids.shape[1]

    print(f"input_ids length: {seq_len}")
    torch.cuda.memory.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history()

    # out = llm.generate(**inputs, min_new_tokens=args.fixed_output_length, max_new_tokens=args.fixed_output_length, top_k=1, temperature=1.0, top_p=1.0, repetition_penalty=1.0)
    out = llm.generate(**inputs, min_new_tokens=args.fixed_output_length, max_new_tokens=args.fixed_output_length, do_sample=False, temperature=1.0, top_p=1.0, repetition_penalty=1.0)

    pred = tokenizer.decode(out[0][seq_len:], skip_special_tokens=True)

    peak_memory = torch.cuda.memory.max_memory_allocated() / (1024 ** 2)  # Convert to MB
    # pred_dir = Path(pred_dir) / f'VramOverview_{args.fixed_output_length}.jsonl'
    model_chao = 'llama' if 'llama' in args.model_name.lower() else 'qwen'
    output_path = Path(f'./results/VramOverview/{model_chao}/')
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    with open(Path(output_path) / f'VramOverview_{args.input_max_token}_{args.fixed_output_length}.jsonl', "a", encoding="utf-8") as f:
        json.dump(
            {   
                "peak_memory_MB": peak_memory,
                "pred": pred, 
            }, 
            f, 
            ensure_ascii=False
        )
        f.write('\n')
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
    model = minference_patch(model)
    return model

def main():
    seed_everything(42)

    args = parse_args()
    data = load_dataset(args)
    # model_path = model2path[args.model_name]
    model_path = args.model_name

    # Load api
    llm = get_llm(model_path)
    for _ in range(args.warmup):
        # seed_everything(42)
        get_pred(
            llm,
            data,
            model_path,
            args.save_dir,
            args,
        )

if __name__ == "__main__":
    main()
