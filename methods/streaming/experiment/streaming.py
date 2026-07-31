import warnings

warnings.filterwarnings("ignore")

import torch
import argparse
import json
import os
import time
import re
import sys
import numpy as np
import random

from tqdm import tqdm
from transformers import DynamicCache
from streaming_llm.utils import load, download_url, load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm
from datasets import load_dataset

import torch

@torch.no_grad()
def streaming_inference(model,model_name_or_path, tokenizer,dataset,dataset_name, samples,output_path,prompt_format,max_gen_len):
    device = model.device
    done_count = 0
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): 
                    done_count += 1
    remaining_samples = samples[done_count:]
    with open(output_path, "a", encoding="utf-8") as fout:
        for  sample in tqdm(remaining_samples):
            if "ruler" in dataset:  
                prompt = prompt_format.format(**sample)
                answers = sample.get("outputs", [])
                all_classes = sample.get("all_classes", [])
                length = sample.get("length", len(sample.get("input", "")))
            elif "LongBench" in dataset:
                prompt = prompt_format.format(**sample)
                answers = sample["answers"]
                all_classes = sample["all_classes"]
                length = sample["length"]
            if "LongBench" in dataset and dataset_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
                prompt = build_chat(tokenizer, prompt, model_name=model_name_or_path)
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

            context_length = input.input_ids.shape[1]
            print(f"\n[DEBUG] Sample Prompt Length: {context_length} tokens")
            if dataset_name == "samsum": 
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen_len,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen_len,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                )[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True).strip()

            with open(output_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": answers , "all_classes": all_classes, "length": length}, f, ensure_ascii=False)
                f.write('\n')
            torch.cuda.empty_cache()

def build_chat(tokenizer, prompt, model_name):
   
    messages = [
            {"role": "user", "content": prompt}
    ]
    prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    return prompt

def build_output_path(model_name_or_path, dataset_name, dataset_path=None, budget=None):

    model_dir_name = os.path.basename(model_name_or_path.rstrip("/"))

    base_output = "./output"

    sub_dir = ""

    if budget is not None:
        output_dir = os.path.join("budget", f"budget{budget}", model_dir_name, sub_dir)
    else:
        output_dir = os.path.join(base_output, model_dir_name, sub_dir)
    os.makedirs(output_dir, exist_ok=True)

    return os.path.join(output_dir, f"{dataset_name}.jsonl")

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
    model=args.model_name_or_path
    model2path = json.load(open("config/model2path.json", "r"))
    model_name_or_path = model2path[model]
    model, tokenizer = load(model_name_or_path)

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    dataset_name = args.dataset_name
    prompt_format = dataset2prompt[dataset_name]
    max_gen_len = dataset2maxlen[dataset_name]

    subset = f"{dataset_name}_e" if getattr(args, 'extended', False) else dataset_name
    print(f"Loading data from THUDM/LongBench ({subset}) ...")
    data = load_dataset("THUDM/LongBench", subset, split="test")
    data_all = [data_sample for data_sample in data]

    if args.enable_streaming:
        model.config.start_size = args.start_size
        model.config.recent_size = args.recent_size

        from streaming_llm.enable_streaming_llm import enable_streaming_llm
        enable_streaming_llm(model,args)
        print(f"StreamingLLM Enabled: Sink={args.start_size}, Recent={args.recent_size}")
    else:
        kv_cache = None

    output_path = build_output_path(
        args.model_name_or_path,
        args.dataset_name,
        "LongBench",
        budget=getattr(args, 'budget', None),
    )

    streaming_inference(
        model,
        model_name_or_path,
        tokenizer,
        "LongBench",
        dataset_name,
        data_all,
        output_path,
        prompt_format,
        max_gen_len=max_gen_len,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=16)
    parser.add_argument("--recent_size", type=int, default=1008)
    parser.add_argument("--dataset_name", type=str, default="qasper")
    parser.add_argument("--output_path", type=str, default="predictions.jsonl")
    parser.add_argument("--budget", type=int, default=None)
    args = parser.parse_args()
 

    main(args)
