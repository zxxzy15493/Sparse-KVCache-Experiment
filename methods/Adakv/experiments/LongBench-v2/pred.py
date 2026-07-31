import os
import json
import argparse
import re
import html
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

from headkv.monkeypatch import replace_llama, replace_qwen2, replace_chatglm

# HeadKV adaptation: no CakeKV imports here


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

def _get_model_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers, "self_attn"
    if hasattr(model, "transformer") and hasattr(model.transformer, "encoder") and hasattr(model.transformer.encoder, "layers"):
        return model.transformer.encoder.layers, "self_attention"
    raise ValueError("Could not find transformer layers in model")

def _get_self_attention_module(layer, attn_attr_name):
    return getattr(layer, attn_attr_name)

class HeadKVConfigHolder:
    def __init__(self):
        self.window_size = None
        self.base_capacity = None
        self.head_choice = None
        self.beta = None
        self.temp = None
        self.kernel_size = None
        self.skip = None
        self.normalize = None
        self.pooling = None
        self.floor = None

def load_model_and_tokenizer(path, model_name, device):
    # Load model and tokenizer (no CakeKV modifications here)
    model_name_lower = model_name.lower()
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    is_chatglm = ("chatglm" in getattr(config, "model_type", "").lower()) or ("glm" in model_name_lower)

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=is_chatglm)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=is_chatglm,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)

    return model.eval(), tokenizer
#modify end

def _apply_headkv_monkeypatch(model_name, head_choice):
    name_lower = model_name.lower()
    if head_choice in {"reason", "copy"}:
        method = "ReasonKV"
    else:
        method = "AdativeKV"

    if "llama" in name_lower:
        replace_llama(method)
    elif "mistral" in name_lower:
        replace_mistral(method)
    elif "qwen" in name_lower:
        replace_qwen2(method)
    elif "glm" in name_lower or "chatglm" in name_lower:
        replace_chatglm(method, None)
    else:
        raise ValueError(f"Unsupported model for HeadKV monkeypatch: {model_name}")

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

def _strip_html(text: str) -> str:
    # Basic HTML tag removal to avoid div-heavy generations on web-sourced contexts.
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()

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

            if args.strip_html:
                context = _strip_html(context)
                
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

    # For LongBench-v2 we always read the unified dataset file
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
    _apply_headkv_monkeypatch(model_key, args.head_choice)
    model, tokenizer = load_model_and_tokenizer(model_path, model_key, device)

    # HeadKV-style config injection
    try:
        layers, attn_attr = _get_model_layers(model)
        holder = HeadKVConfigHolder()
        holder.window_size = args.window_size
        holder.base_capacity = args.max_capacity_prompts
        holder.head_choice = args.head_choice
        holder.beta = args.beta
        holder.temp = args.temp
        holder.kernel_size = args.kernel_size
        holder.skip = args.skip
        holder.normalize = args.normalize
        holder.pooling = args.pooling
        holder.floor = args.floor

        for layer in layers:
            attn = _get_self_attention_module(layer, attn_attr)
            if not hasattr(attn, 'config'):
                attn.config = model.config
            # apply headkv params to attention config
            attn.config.window_size = holder.window_size
            attn.config.base_capacity = holder.base_capacity
            attn.config.head_choice = holder.head_choice
            attn.config.beta = holder.beta
            attn.config.temp = holder.temp
            attn.config.kernel_size = holder.kernel_size
            attn.config.skip = holder.skip
            attn.config.normalize = holder.normalize
            attn.config.pooling = holder.pooling
            attn.config.floor = holder.floor
        # ReasonKV headscore loading is handled in ReasonSnapKVCluster.
    except Exception:
        pass

    # Debug: print head_choice and preview per-layer head config to diagnose identical outputs
    try:
        print(f"[DEBUG HEADKV] global head_choice={holder.head_choice}")
        preview = []
        for idx, layer in enumerate(layers):
            if idx >= 6:
                break
            attn = _get_self_attention_module(layer, attn_attr)
            hc = getattr(attn.config, 'head_choice', None)
            rh = getattr(attn.config, 'reason_heads', None)
            ch = getattr(attn.config, 'copy_heads', None)
            preview.append((idx, hc, None if rh is None else (len(rh), rh[:5] if isinstance(rh, (list, tuple)) else str(type(rh)))),)
        print(f"[DEBUG HEADKV] layer preview (idx, head_choice, reason_heads_preview): {preview}")
    except Exception as e:
        print(f"[DEBUG HEADKV] failed to produce preview: {e}")

    get_pred_local(data, args, out_file, model, tokenizer, max_length)

if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="glm-4-9b-chat-1m")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--max_capacity_prompts', type=int, default=1024)
    parser.add_argument('--head_choice', type=str, default='random', choices=['random', 'copy', 'reason'])
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--kernel_size', type=int, default=7)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--pooling', type=str, default='maxpool')
    parser.add_argument('--floor', type=float, default=0.2)
    parser.add_argument("--cot", "-cot", action='store_true') 
    parser.add_argument("--no_context", "-nc", action='store_true') 
    parser.add_argument("--rag", "-rag", type=int, default=0) 
    parser.add_argument("--strip_html", action='store_true')
    args = parser.parse_args()
    main(args)