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

from cake.cake_cache import CakeprefillKVCache
from cake.monkeypatch import (
    replace_flashllama_attn_with_cakeattn,
    replace_flashmistral_attn_with_cakeattn,
    replace_flashqwen2_attn_with_cakeattn,
    replace_chatglm_attn_with_cakeattn,
)
from cake.utils import CompressConfig


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
tau_map = _read_json(SCRIPT_DIR / 'config' / 'model2tau.json')

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

def _get_model_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers, "self_attn"
    if hasattr(model, "transformer") and hasattr(model.transformer, "encoder") and hasattr(model.transformer.encoder, "layers"):
        return model.transformer.encoder.layers, "self_attention"
    raise ValueError("Could not find transformer layers in model")
def load_model_and_tokenizer(path, model_name, device, compress_config):
    model_name_lower = model_name.lower()
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "").lower()
    is_chatglm = ("chatglm" in model_type) or ("glm" in model_name_lower)

    if compress_config and compress_config.compress:
        if "llama" in model_name_lower:
            replace_flashllama_attn_with_cakeattn()
        elif "mistral" in model_name_lower:
            replace_flashmistral_attn_with_cakeattn()
        elif "qwen2" in model_name_lower:
            replace_flashqwen2_attn_with_cakeattn()
        elif not is_chatglm:
            raise ValueError(f"Unsupported model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=is_chatglm)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=is_chatglm,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)

    if compress_config and compress_config.compress and is_chatglm:
        replace_chatglm_attn_with_cakeattn(model)

    if compress_config and compress_config.compress:
        model_layers, attn_attr = _get_model_layers(model)
        layer_count = len(model_layers)
        for idx in range(layer_count):
            attn = getattr(model_layers[idx], attn_attr)
            if not hasattr(attn, "config"):
                attn.config = model.config
            attn.config.key_size = [compress_config.cache_size - compress_config.window_size] * layer_count
            attn.config.window_size = [compress_config.window_size] * layer_count
            attn.config.prefill = [True] * layer_count
            attn.config.decoding_evict = [None] * layer_count
            attn.config.tau1 = compress_config.hyper[0]
            attn.config.tau2 = compress_config.hyper[1]
            attn.config.gamma = compress_config.hyper[2]

            if attn_attr == "self_attention":
                kv_heads = getattr(attn, "num_multi_query_groups_per_partition", getattr(attn, "num_attention_heads_per_partition", 1))
            else:
                kv_heads = getattr(attn, "num_heads", 1)

            attn.config.prefill_cake_evict = [
                CakeprefillKVCache(
                    cache_size=compress_config.cache_size,
                    window_size=compress_config.window_size,
                    k_seq_dim=2,
                    v_seq_dim=2,
                    num_heads=kv_heads,
                    num_layers=layer_count,
                    use_cascading=compress_config.cascading,
                )
            ] * layer_count

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
    
    compress_config = CompressConfig(
        compress=args.compress,
        cascading=args.cascading,
        cache_size=args.cache_size,
        window_size=args.window_size,
        hyper=[args.tau1, args.tau2, args.gamma],
    )

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

    if args.compress:
        cache_key = str(args.cache_size)
        try:
            tau_cfg = tau_map[model_key][cache_key]
            compress_config.hyper = [float(tau_cfg["tau1"]), float(tau_cfg["tau2"]), float(args.gamma)]
        except KeyError:
            compress_config.hyper = [float(args.tau1), float(args.tau2), float(args.gamma)]

    model_path = model_map[model_key]
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    print(f"Loading local model from {model_path}...")
    model, tokenizer = load_model_and_tokenizer(model_path, model_key, device, compress_config)
    
    get_pred_local(data, args, out_file, model, tokenizer, max_length)

if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="glm-4-9b-chat-1m")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument('--compress', action='store_true', help="Comrpess kv cache with CAKE")
    parser.add_argument('--cascading', action='store_true', help="Using cascading cache mangement")
    parser.add_argument('--cache_size', type=int, default=1024)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--tau1', type=float, default=1.0)
    parser.add_argument('--tau2', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=200.0)
    parser.add_argument("--cot", "-cot", action='store_true') 
    parser.add_argument("--no_context", "-nc", action='store_true') 
    parser.add_argument("--rag", "-rag", type=int, default=0) 
    args = parser.parse_args()
    main(args)