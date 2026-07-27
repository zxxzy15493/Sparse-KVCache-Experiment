import os, csv, json
import argparse
from pathlib import Path
import sys
import time
from tqdm import tqdm
import re
import torch
from transformers import AutoTokenizer
import tiktoken
from utils import load_data


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)
from model_hub import LlamaModel, QwenModel, GlmModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import generate_config, parse_attn_args

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def query_llm(prompt, llm, model, max_new_tokens=128, stop=None):
    # truncate
    if llm.tokenizer.eos_token is not None:
        llm.tokenizer.pad_token = llm.tokenizer.eos_token
    elif llm.tokenizer.pad_token_id is None and len(llm.eos_tokens) > 0:
        llm.tokenizer.pad_token_id = llm.eos_tokens[0]
    llm.tokenizer.padding_side = "left"
    

    messages = [{"role": "user", "content": prompt}]
    prompt = llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    max_len = maxlen_map[model]
    if model in model_map:
        input_ids = llm.tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            print("n" + "=" * 80)
            print(f"input_ids length is {len(input_ids)} before truncation.")
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            print("=" * 80)
            print(f"input_ids length is {len(input_ids)} after truncation.")
            print("=" * 80 + "\n")
            prompt = llm.tokenizer.decode(input_ids, skip_special_tokens=True)
    else:
        input_ids = llm.tokenizer.encode(prompt, disallowed_special=())
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = llm.tokenizer.decode(input_ids)

    inputs = llm.tokenizer(prompt, return_tensors="pt", padding=True)

    input_ids = inputs.input_ids
    attention_masks = inputs.attention_mask

    attn_config = generate_config(
        model, 
        input_ids.shape[1], 
        args.attn_type,
        budget_ratio=args.budget_ratio,
        budget=args.budget,
        estimate_ratio=args.estimate_ratio,
        ratio_or_fixed=args.ratio_or_fixed,
    )

    out = llm.generate(
        attention_type=args.attn_type,
        inputs_ids = input_ids.to(llm.layers[0].device),
        attention_masks = attention_masks.to(llm.layers[0].device),
        max_new_length=max_new_tokens, 
        attn_config=attn_config,
        prefill_method=args.prefill_method
    )

    output = llm.tokenizer.batch_decode(out, skip_special_tokens=True)
    
    torch.cuda.empty_cache()
            
    return  output[0]

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

def load_model(model_path, max_len, dtype, device, max_new_tokens, args):
    if 'Llama' in model_path:
        llm = LlamaModel(model_path,
            max_length=max_len + max_new_tokens,
            dtype=dtype,
            device_map=device,
            RECALL=args.recall
            )
    elif 'Qwen' in model_path:
        llm = QwenModel(model_path,
            max_length=max_len + max_new_tokens,
            dtype=dtype,
            device_map=device,
            RECALL=args.recall
            )
    elif 'GLM' in model_path or 'glm' in model_path:
        llm = GlmModel(model_path,
            max_length=max_len + max_new_tokens,
            dtype=dtype,
            device_map=device,
            RECALL=args.recall,
            )
    else:
        raise ValueError(f"Unsupported model: {model_path}")

    if llm.tokenizer.eos_token is not None:
        llm.tokenizer.pad_token = llm.tokenizer.eos_token
    elif llm.tokenizer.pad_token_id is None and len(llm.eos_tokens) > 0:
        llm.tokenizer.pad_token_id = llm.eos_tokens[0]
    llm.tokenizer.padding_side = "left"
    
    return llm

def get_pred(data, args, fout):
    model = args.model

    dtype = torch.bfloat16
    llm = load_model(model_map[model], maxlen_map[model], dtype, args.device, 1024, args)
    for item in tqdm(data):
        context = item['context']
        
        if args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        prompt = template\
                .replace('$DOC$', context.strip())\
                .replace('$Q$', item['question'].strip())\
                .replace('$C_A$', item['choice_A'].strip())\
                .replace('$C_B$', item['choice_B'].strip())\
                .replace('$C_C$', item['choice_C'].strip())\
                .replace('$C_D$', item['choice_D'].strip())
        if args.cot:
            output = query_llm(prompt, llm, model, max_new_tokens=1024)
        else:
            output = query_llm(prompt, llm, model, max_new_tokens=128)
        if output == '':
            continue
        if args.cot: # extract answer
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans\
                .replace('$DOC$', context.strip())\
                .replace('$Q$', item['question']\
                .strip())\
                .replace('$C_A$', item['choice_A'].strip())\
                .replace('$C_B$', item['choice_B'].strip())\
                .replace('$C_C$', item['choice_C'].strip())\
                .replace('$C_D$', item['choice_D'].strip())\
                .replace('$COT$', response)
            output = query_llm(prompt, llm, model, max_new_tokens=128)
            if output == '':
                continue
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    if args.no_context:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_no_context.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_cot.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + ".jsonl")

    dataset = load_data('../../../../benchmarks/longbenchv2/filtered_longbench_v2_64k-192k.jsonl')
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]
    
    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    get_pred(data, args, fout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)

    # Retroinfer
    parser.add_argument("--attn_type", type=str, default="Full_Flash_Attn",                                                     \
                        choices=["Full_Flash_Attn", "RetroInfer"],                          \
                        help="Attention method")
    parser.add_argument("--prefill_method", type=str, default="Full_Flash_Attn",                                                     \
                        choices=["Full_Flash_Attn", "minfer"],                          \
                        help="Attention method for prefill phase, which determines the attention method used during the prefill phase. When set to 'minfer', it will use the minfer method with the best patterns for each layer. When set to 'Full_Flash_Attn', it will use full flash attention during the prefill phase.")
    parser.add_argument("--max_len", type=int, default=1024, help="Length of the context for attention computation")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Dtype")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--fixed_output_length", type=int, default=0, required=False)
    parser.add_argument("--recall",action="store_true", required=False)
    parser.add_argument("--measure_time", action="store_true", required=False)
    parser = parse_attn_args(parser)
    args = parser.parse_args()
    main()