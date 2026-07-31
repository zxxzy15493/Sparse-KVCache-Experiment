import os, csv, json
import argparse
from pathlib import Path
from pyexpat import model
import sys
import time
from tqdm import tqdm
import re
import torch
from transformers import AutoTokenizer
import tiktoken

_MAGICPIG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _MAGICPIG_ROOT not in sys.path:
    sys.path.insert(0, _MAGICPIG_ROOT)


from models_single.llama import LlamaModel
from models_single.qwen import Qwen2Model
from utils import load_data
import torch.distributed as dist

dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)
DEVICE = torch.device("cuda", local_rank)


model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def query_llm(prompt, llm, tokenizer, model, max_new_tokens=128, stop=None):

    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    input_ids = inputs["input_ids"]
    seq_len = inputs["input_ids"].shape[1]
    output = llm.generate(input_ids=input_ids, max_tokens=max_new_tokens)

    avg_token = llm.attention_server.avg_nnz / llm.attention_server.count_nnz
    llm.attention_server.avg_nnz = 0
    llm.attention_server.count_nnz = 0

    generated_text = tokenizer.decode(output[seq_len:], skip_special_tokens=True)
    
    torch.cuda.empty_cache()
            
    return generated_text, avg_token, avg_token / seq_len

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

def load_model(model_name, K, L, batch_size, max_length, device, dtype, args):
    recall = True if args.recall == 1 else False
    fixed_budget = args.fixed_budget
    fixed_output_length = args.fixed_output_length 
    measure_time = True if args.measure_time == 1 else False

    if 'Llama' in model_name:
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
    elif 'Qwen' in model_name:
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

def get_pred(data, args, fout):
    model = args.model
    if "gpt" in model or "o1" in model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_map[model], trust_remote_code=True)
    dtype = torch.bfloat16
    llm = load_model(model_map[model], args.K, args.L, 1, maxlen_map[model], "cuda:0", dtype, args)
    avg_ratio = 0
    avg_token = 0
    count = 0
    for item in tqdm(data):
        try:
            context = item['context']
            if args.cot:
                template = template_0shot_cot
            else:
                template = template_0shot
            prompt = template\
                    .replace('$DOC$', context.strip())\
                    .replace('$Q$', item['question'].strip())\
                    .replace('$C_A$', item['choice_A']\
                    .strip())\
                    .replace('$C_B$', item['choice_B'].strip())\
                    .replace('$C_C$', item['choice_C'].strip())\
                    .replace('$C_D$', item['choice_D'].strip())

            output, token_cnt, token_ratio = query_llm(prompt, llm, tokenizer, model, max_new_tokens=1024)
            if output == '':
                continue
            avg_token += token_cnt
            avg_ratio += token_ratio
            count += 1
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
            output, token_cnt, token_ratio = query_llm(prompt, llm, tokenizer, model, max_new_tokens=128)
            if output == '':
                continue
            response = output.strip()
            item['response'] = response
            item['pred'] = extract_answer(response)
            item['judge'] = item['pred'] == item['answer']
            item['context'] = context[:1000]
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
            fout.flush()
        except Exception as e:
            print(f"Error processing item with _id {item['_id']}: {e}")
            continue
        finally:
            torch.cuda.empty_cache()

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

    parser.add_argument("--K", type=int, default=9)
    parser.add_argument("--L", type=int, default=200)
    parser.add_argument("--recall", type=int, required=False)
    parser.add_argument("--fixed_budget", type=int, default=0, required=False)
    parser.add_argument("--fixed_output_length", type=int, default=0, required=False)
    parser.add_argument("--measure_time", type=int, default=0, required=False)
    args = parser.parse_args()
    main()  