import os
import json
import argparse
import re
import sys
from pathlib import Path
from tqdm import tqdm
# from datasets import load_dataset  # replaced with local jsonl file
import torch
import numpy as np
import random

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from duo_attn.patch import enable_duo_attention_eval
from duo_attn.patch.qwen2 import enable_qwen2_duo_attention_eval
from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache
from duo_attn.utils import load_attn_pattern, sparsify_attention_heads


def _read_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def _resolve_model_key(model_name):
    normalized = model_name.strip().lower()
    candidates = [normalized, normalized.split('/')[-1], normalized.replace('qwen/', '')]

    for candidate in candidates:
        for key in model_map:
            if key.lower() == candidate:
                return key

    for key in model_map:
        lowered = key.lower()
        if lowered in normalized or normalized in lowered:
            return key

    raise KeyError(f"Unknown model '{model_name}'. Available keys: {', '.join(model_map.keys())}")


model_map = _read_json(SCRIPT_DIR / 'config' / 'model2path.json')
maxlen_map = _read_json(SCRIPT_DIR / 'config' / 'model2maxlen.json')

template_rag = (SCRIPT_DIR / 'prompts' / '0shot_rag.txt').read_text(encoding='utf-8')
template_no_context = (SCRIPT_DIR / 'prompts' / '0shot_no_context.txt').read_text(encoding='utf-8')
template_0shot = (SCRIPT_DIR / 'prompts' / '0shot.txt').read_text(encoding='utf-8')
template_0shot_cot = (SCRIPT_DIR / 'prompts' / '0shot_cot.txt').read_text(encoding='utf-8')
template_0shot_cot_ans = (SCRIPT_DIR / 'prompts' / '0shot_cot_ans.txt').read_text(encoding='utf-8')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device, args):
    model_name_lower = model_name.lower()
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "").lower()
    is_chatglm = ("chatglm" in model_type) or ("glm" in model_name_lower)

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=is_chatglm)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=is_chatglm,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)

    if args.method == "duo_attn":
        if args.attn_load_dir is None:
            raise ValueError("When --method duo_attn, --attn_load_dir must be provided")

        full_attention_heads, sink_size, recent_size = load_attn_pattern(args.attn_load_dir)
        if args.sink_size is not None:
            sink_size = args.sink_size
        if args.recent_size is not None:
            recent_size = args.recent_size

        mask, sparsity = sparsify_attention_heads(full_attention_heads.copy(), None, sparsity=args.sparsity)
        print(f"True sparsity: {sparsity}")
        if "qwen" in model_name_lower or "qwen" in path.lower():
            enable_qwen2_duo_attention_eval(
                model,
                torch.tensor(mask, dtype=torch.float32),
                sink_size,
                recent_size,
            )
        else:
            enable_duo_attention_eval(
                model,
                torch.tensor(mask, dtype=torch.float32),
                sink_size,
                recent_size,
            )
    else:
        enable_tuple_kv_cache(model)

    return model.eval(), tokenizer
#modify end

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

#modify
@torch.inference_mode()
def get_pred_local(data, args, out_file, model, tokenizer, max_length):
    device = model.device
    
    with open(out_file, 'a', encoding='utf-8') as fout:
        for item in tqdm(data, desc="Evaluating"):
            torch.cuda.empty_cache()

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

            if len(input_ids) > max_length:
                half = max_length // 2
                input_ids = input_ids[:half] + input_ids[-half:]
                prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
            
            def local_generate(p, max_tokens):
                inputs = tokenizer(p, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0 if max_tokens == 128 else 0.1,
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

    model_name = args.model
    model_key = _resolve_model_key(model_name)
    out_file = os.path.join(args.save_dir, model_key + ".jsonl")


    _repo_root = Path(__file__).resolve().parents[4]
    dataset = [json.loads(line) for line in open(_repo_root / 'benchmarks' / 'longbenchv2' / 'filtered_longbench_v2_64k-192k.jsonl')]
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
            
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    max_length = int(maxlen_map.get(model_key, 10000000))

    model_path = model_map[model_key]
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    print(f"Loading local model from {model_path}...")
    model, tokenizer = load_model_and_tokenizer(model_path, model_key, device, args)
    
    get_pred_local(data, args, out_file, model, tokenizer, max_length)

if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="glm-4-9b-chat-1m")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument('--method', type=str, default='duo_attn', choices=['full', 'duo_attn'])
    parser.add_argument('--attn_load_dir', type=str, default='Qwen/qwen2.5-7b-instruct', help='attention pattern directory')
    parser.add_argument('--sink_size', type=int, default=None)
    parser.add_argument('--recent_size', type=int, default=None)
    parser.add_argument('--sparsity', type=float, default=0.5)
    parser.add_argument('--max_num_examples', type=int, default=None)
    parser.add_argument("--cot", "-cot", action='store_true') 
    parser.add_argument("--no_context", "-nc", action='store_true') 
    parser.add_argument("--rag", "-rag", type=int, default=0) 
    args = parser.parse_args()
    main(args)