import warnings
warnings.filterwarnings("ignore")
import torch
import argparse
import json
import os
import time
import numpy as np
import random
from streaming_llm.utils import load
from streaming_llm.enable_streaming_llm import enable_streaming_llm

def build_chat(tokenizer, prompt, model_name):
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def build_output_path(model_name_or_path, out_len, budget):
    model_dir_name = os.path.basename(model_name_or_path.rstrip("/"))
    base_output = "output/ef"
    output_dir = os.path.join(base_output, f"budget{budget}", model_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"out{out_len}.jsonl"
    return os.path.join(output_dir, file_name)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def main(args):
    seed_everything(42)
    
    model_key = args.model_name_or_path
    model2path = json.load(open("config/model2path.json", "r"))
    actual_model_path = model_key
    model, tokenizer = load(actual_model_path)
    
    if args.enable_streaming:
        model.config.start_size = args.start_size
        model.config.recent_size = args.recent_size
        enable_streaming_llm(model, args)

    with open(args.data_root, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()
    
    input_ids_full = tokenizer(raw_text, return_tensors="pt").input_ids
    
    in_len = args.test_len
    total_budget = args.start_size + args.recent_size
    output_path = build_output_path(args.model_name_or_path, args.out_len, total_budget)

    for run_idx in range(args.num_runs):
        if in_len > input_ids_full.shape[1]:
            break
        
        print(f"Run {run_idx + 1} - Input Length: {in_len}, Budget: {total_budget}")
            
        input_ids = input_ids_full[:, :in_len].to(model.device)
        decode_latency = []
        past_key_values = None
        current_input_ids = input_ids

        with torch.no_grad():
            for i in range(args.out_len):
                torch.cuda.synchronize()
                start_ts = time.perf_counter()
                
                outputs = model(
                    input_ids=current_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    num_logits_to_keep=1
                )
                
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
            "in_len": in_len,
            "budget": total_budget,
            "ttft": ttft,
            "tpot": tpot,
            "latency": latency,
            "decode_latency": decode_latency
        }

        with open(output_path, "a", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False)
            f.write("\n")

        del past_key_values
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data_root", type=str, default="myinput.txt")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=16)
    parser.add_argument("--recent_size", type=int, default=1008)
    parser.add_argument("--out_len", type=int, default=32)
    parser.add_argument("--num_runs", type=int, default=4)
    parser.add_argument("--test_len", type=int, default=4096)
    args = parser.parse_args()
    main(args)