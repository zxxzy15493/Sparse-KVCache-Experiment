import argparse
import json
from pathlib import Path
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import yaml
import numpy as np
import random
import time
from tqdm import tqdm
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from minference_time import MInference, time_manager, full_time_manager
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
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--fixed_output_length", type=int, default=0, required=False)
    parser.add_argument("--input_max_token", type=int, default=1024)
    # MInference
    parser.add_argument("--model_name", type=str, required=True, help="model name or path for MInferenceModel")
    parser.add_argument("--full", action="store_true", help="whether to use full time manager or not")
    return parser.parse_args()

def load_dataset(args):
    with open("../../../../benchmarks/myinput.txt", "r", encoding="utf-8") as f:
        data = f.read()  # ：read()  data
    return data

def get_pred(model, data, model_name, pred_dir, args):
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(data, return_tensors="pt", return_attention_mask=True).to(model.device)
    print(f"input length: {inputs.input_ids.shape[1]}")
    inputs['input_ids'] = inputs.input_ids[:, :args.input_max_token]
    if "attention_mask" in inputs:
        inputs['attention_mask'] = inputs.attention_mask[:, :args.input_max_token]

    if args.full:
        if "llama" in model_name.lower():
            time_infos = full_time_manager.myinit(args.fixed_output_length, 32)
        elif "qwen" in model_name.lower():
            time_infos = full_time_manager.myinit(args.fixed_output_length, 28)
    else:
        if "llama" in model_name.lower():
            time_infos = time_manager.myinit(args.fixed_output_length, 32)
        elif "qwen" in model_name.lower():
            time_infos = time_manager.myinit(args.fixed_output_length, 28)
    for run_idx in range(args.warmup):
        decode_latency, past_key_values, current_input_ids = [], None, inputs['input_ids']
        with torch.no_grad():
            for i in range(args.fixed_output_length):
                torch.cuda.synchronize()
                start_ts = time.perf_counter()
                outputs = model(input_ids=current_input_ids, past_key_values=past_key_values, use_cache=True, num_logits_to_keep=1)
                past_key_values = outputs.past_key_values
                next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
                torch.cuda.synchronize()
                decode_latency.append(time.perf_counter() - start_ts)
                current_input_ids = next_token_id
        ttft = decode_latency[0]
        tpot = np.mean(decode_latency[1:]) if len(decode_latency) > 1 else 0
        latency = sum(decode_latency)

        save_data = {
            "run": run_idx + 1,  
            "ttft": ttft,
            "tpot": tpot*1000,
            "latency": latency,
            "decode_latency": decode_latency
        }
        prefill_latency = ttft
        decode_latency = np.sum(decode_latency[1:]) if len(decode_latency) > 1 else 0
        if args.full:
            if "llama" in model_name.lower():
                time_infos = full_time_manager.get_final_time(prefill_latency, decode_latency)
            elif "qwen" in model_name.lower():
                time_infos = full_time_manager.get_final_time(prefill_latency, decode_latency)
        else:
            if "llama" in model_name.lower():
                time_infos = time_manager.get_final_time(prefill_latency, decode_latency)
            elif "qwen" in model_name.lower():
                time_infos = time_manager.get_final_time(prefill_latency, decode_latency)
        
   
        
        model_chao = 'llama' if 'llama' in args.model_name.lower() else 'qwen'

        output_path = Path(f'./results/SelectTimeBreakDown/{model_chao}{"_full" if args.full else ""}/')
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / f"Efficency_{args.input_max_token}_{args.fixed_output_length}_{'full' if args.full else ''}.jsonl", "a", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False)
            f.write("\n")

        with open(output_path / f"SelectTimeBreakDown_{args.input_max_token}_{args.fixed_output_length}_{'full' if args.full else ''}.jsonl", "a", encoding="utf-8") as f:
            json.dump(time_infos, f, ensure_ascii=False, indent=4)
            f.write("\n")
        del outputs,past_key_values
        torch.cuda.empty_cache()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def get_llm(args, model_path): 
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map="cuda:0",_attn_implementation="flash_attention_2")
    minference_patch = MInference(model_path)
    # model = minference_patch(model)
    if args.full:
        model = minference_patch.full_patch(model)
    else:
        model = minference_patch(model)
    return model

def main():
    seed_everything(42)

    args = parse_args()
    data = load_dataset(args)
    model_path = args.model_name

    # Load api
    llm = get_llm(args, model_path)
    get_pred(
        llm,
        data,
        model_path,
        args.save_dir,
        args,
    )

if __name__ == "__main__":
    main()
