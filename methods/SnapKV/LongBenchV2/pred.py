import os
import csv
import json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import tiktoken
import torch
import numpy as np
import random

from snapkv.monkeypatch.monkeypatch import replace_llama, replace_qwen, replace_glm

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, torch_dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, 
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    model = model.eval()
    return model, tokenizer

def set_snapkv_layers(model, compress_args, check_recall=False):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model_layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "encoder") and hasattr(model.transformer.encoder, "layers"):
        model_layers = model.transformer.encoder.layers
    else:
        raise ValueError("Could not find layers in model")
    
    layers = len(model_layers)
    window_sizes = compress_args.get("window_sizes", 32)
    max_capacity_prompts = compress_args.get("max_capacity_prompts", 512)
    kernel_sizes = compress_args.get("kernel_sizes", 5)
    pooling = compress_args.get("pooling", "avg")

    if not isinstance(window_sizes, list):
        window_sizes = [window_sizes] * layers
    if not isinstance(max_capacity_prompts, list):
        max_capacity_prompts = [max_capacity_prompts] * layers
    if not isinstance(kernel_sizes, list):
        kernel_sizes = [kernel_sizes] * layers

    for i in range(layers):
        attn = getattr(model_layers[i], "self_attn", getattr(model_layers[i], "self_attention", None))
        if attn is not None:
            attn.config.window_size = window_sizes[i]
            attn.config.max_capacity_prompt = max_capacity_prompts[i]
            attn.config.kernel_size = kernel_sizes[i]
            attn.config.pooling = pooling
            attn.check_recall = check_recall

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def get_pred_local(data, args, out_file, model, tokenizer, compress_args=None):
    device = model.device
    max_len = 10000000
    
    with open(out_file, 'a', encoding='utf-8') as fout:
        for item in tqdm(data, desc="Evaluating"):
            if compress_args:
                set_snapkv_layers(model, compress_args, args.check_recall)

            context = item['context']
            if args.rag > 0:
                template = template_rag
                retrieved = item["retrieved_context"][:args.rag]
                retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
                context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
            elif args.no_context:
                template = template_no_context
            elif args.cot:
                template = template_0shot_cot
            else:
                template = template_0shot
                
            prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
            
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
                prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
            
            def local_generate(p, max_tokens):
                inputs = tokenizer(p, return_tensors="pt").to(device)
                try:
                    ctx = torch.inference_mode()
                except AttributeError:
                    ctx = torch.no_grad()
                with ctx:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0 if max_tokens == 128 else 0.1
                    )
                return tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

            if args.cot:
                output = local_generate(prompt, 1024)
            else:
                output = local_generate(prompt, 128)
                
            if output == '':
                continue
            if args.cot: 
                response = output.strip()
                item['response_cot'] = response
                prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
                output = local_generate(prompt, 128)
                if output == '':
                    continue
                    
            response = output.strip()
            item['response'] = response
            item['pred'] = extract_answer(response)
            item['judge'] = item['pred'] == item['answer']
            item['context'] = context[:1000]
            
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
            fout.flush()

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    
    if args.compress_args_path:
        comp_path = args.compress_args_path
        if not os.path.isabs(comp_path) and not os.path.exists(comp_path):
            comp_path = os.path.join('config', comp_path)
        compress_args = json.load(open(comp_path, "r", encoding='utf-8'))
        replace_qwen()

    else:
        compress_args = None

    model_name = os.path.basename(args.model)
    out_file = os.path.join(args.save_dir, model_name + ".jsonl")

    dataset = load_dataset('json', data_files=os.path.join(os.path.dirname(__file__), '../../../benchmarks/longbenchv2/filtered_longbench_v2_64k-192k.jsonl'), split='train')
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
            
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    print(f"Loading local model from {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    get_pred_local(data, args, out_file, model, tokenizer, compress_args)

if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--cot", "-cot", action='store_true') 
    parser.add_argument("--no_context", "-nc", action='store_true') 
    parser.add_argument("--rag", "-rag", type=int, default=0) 
    parser.add_argument('--compress_args_path', type=str, default=None, help="Path to the compress args")
    parser.add_argument('--check_recall', action='store_true', help="Enable recall calculation during decoding")
    args = parser.parse_args()
    main(args)