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

template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

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

@torch.no_grad()
def streaming_inference_v2(model, model_name_or_path, tokenizer, samples, output_path, args, max_len):
    device = model.device
    done_count = 0
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_count += 1
    remaining_samples = samples[done_count:]

    with open(output_path, "a", encoding="utf-8") as fout:
        for sample in tqdm(remaining_samples, desc="Streaming Inference LongBench-v2"):
            context = sample['context']

            if args.rag > 0:
                template = template_rag
                retrieved = sample["retrieved_context"][:args.rag]
                retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
                context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
            elif args.no_context:
                template = template_no_context
            elif args.cot:
                template = template_0shot_cot
            else:
                template = template_0shot

            prompt = template.replace('$DOC$', context.strip())\
                             .replace('$Q$', sample['question'].strip())\
                             .replace('$C_A$', sample['choice_A'].strip())\
                             .replace('$C_B$', sample['choice_B'].strip())\
                             .replace('$C_C$', sample['choice_C'].strip())\
                             .replace('$C_D$', sample['choice_D'].strip())

            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
                prompt = tokenizer.decode(input_ids, skip_special_tokens=True)

            def local_stream_generate(p, max_tokens):
                formatted_prompt = build_chat(tokenizer, p, model_name=model_name_or_path)
                inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
                context_length = inputs.input_ids.shape[1]

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_beams=1,
                    do_sample=False,
                    temperature=0.1 if args.cot else 1.0,
                    min_length=context_length+1,
                )[0]
                return tokenizer.decode(outputs[context_length:], skip_special_tokens=True).strip()

            if args.cot:
                output = local_stream_generate(prompt, 1024)
            else:
                output = local_stream_generate(prompt, 128)

            if output == '':
                continue

            if args.cot:
                response_cot = output.strip()
                sample['response_cot'] = response_cot
                prompt_ans = template_0shot_cot_ans.replace('$DOC$', context.strip())\
                                                   .replace('$Q$', sample['question'].strip())\
                                                   .replace('$C_A$', sample['choice_A'].strip())\
                                                   .replace('$C_B$', sample['choice_B'].strip())\
                                                   .replace('$C_C$', sample['choice_C'].strip())\
                                                   .replace('$C_D$', sample['choice_D'].strip())\
                                                   .replace('$COT$', response_cot)
                output = local_stream_generate(prompt_ans, 128)
                if output == '':
                    continue

            response = output.strip()
            sample['response'] = response
            sample['pred'] = extract_answer(response)
            sample['judge'] = sample['pred'] == sample['answer']
            sample['context'] = context[:1000]

            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
            fout.flush()
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

def build_output_path(model_name_or_path, dataset_name, dataset_path=None):

    model_dir_name = os.path.basename(model_name_or_path.rstrip("/"))

    base_output = "/streaming/experiment/output"

    sub_dir = ""
    if dataset_path and "evict_ruler" in dataset_path:
        parts = dataset_path.split(os.sep)
        try:
            idx = parts.index("evict_ruler")
            if idx + 1 < len(parts):
                sub_dir = parts[idx + 1]
        except ValueError:
            pass

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
    model_key=args.model_name_or_path
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    model_name_or_path = args.model_name_or_path

    max_length = 1000000
    model, tokenizer = load(model_name_or_path)

    print(f"Loading LongBench-v2 dataset from ../../../benchmarks/longbenchv2/filtered_longbench_v2_64k-192k.jsonl")
    dataset = load_dataset('json', data_files=os.path.join(os.path.dirname(__file__), '../../../benchmarks/longbenchv2/filtered_longbench_v2_64k-192k.jsonl'), split='train')
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"],
                 "difficulty": item["difficulty"], "length": item["length"], "question": item["question"],
                 "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"],
                 "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    if args.enable_streaming:
        model.config.start_size = args.start_size
        model.config.recent_size = args.recent_size

        from streaming_llm.enable_streaming_llm import enable_streaming_llm
        enable_streaming_llm(model, args)
        print(f"StreamingLLM Enabled: Sink={args.start_size}, Recent={args.recent_size}")

    model_name = os.path.basename(args.model_name_or_path.rstrip("/"))
    output_path = os.path.join(args.save_dir, model_name + ".jsonl")
    os.makedirs(args.save_dir, exist_ok=True)

    streaming_inference_v2(
        model=model,
        model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        samples=data_all,
        output_path=output_path,
        args=args,
        max_len=max_length
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--cot", "-cot", action='store_true')
    parser.add_argument("--no_context", "-nc", action='store_true')
    parser.add_argument("--rag", "-rag", type=int, default=0)
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=16)
    parser.add_argument("--recent_size", type=int, default=4080)
    args = parser.parse_args()

    main(args)
