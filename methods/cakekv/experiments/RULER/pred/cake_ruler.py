"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl: 
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""
import argparse
import json
import yaml
import os
import sys
import threading
import importlib
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoConfig, Gemma2ForCausalLM
import time
import torch
from threading import Thread
from transformers import TextIteratorStreamer
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional
import traceback
from cakekv.experiments.RULER.pred.utils import load_data
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(PROJECT_ROOT)
from cakekv.cake.cake_cache import CakeprefillKVCache 
from cakekv.cake.utils import CompressConfig 


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="/DATA/models/Qwen2.5-7B-Instruct")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--compress', action='store_true', help="Comrpess kv cache with CAKE")
    parser.add_argument('--cascading', action='store_true', help="Using cascading cache mangement")
    parser.add_argument('--pred_name', type=str, default="pred", help="Pred Output Name")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cache_size', type=int, default=1024)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--tau1', type=float, default=1.0)
    parser.add_argument('--tau2', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=200.0)

    parser.add_argument("--data_dir", type=Path, required=True, help='path to load the dataset jsonl files')
    parser.add_argument("--save_dir", type=Path, required=True, help='path to save the prediction jsonl files')
    parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
    parser.add_argument("--task", type=str, required=True, help='Options: tasks in benchmark')
    parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
    parser.add_argument("--chunk_idx", type=int, default=0, help='index of current split chunk')
    parser.add_argument("--chunk_amount", type=int, default=1, help='size of split chunk')

    # Server
    parser.add_argument("--server_type", default='nemo', action=ServerAction, choices=SERVER_TYPES)
    parser.add_argument("--server_host", type=str, default='127.0.0.1')
    parser.add_argument("--server_port", type=str, default='5000')
    parser.add_argument("--ssh_server", type=str)
    parser.add_argument("--ssh_key_path", type=str)

    
    parser.add_argument("--threads", type=int, default=4)

    parser.add_argument("--synthetic_len", type=int, required=True)
    parser.add_argument('--limit', type=int, default=50, help='最多处理的样本数')

    return parser.parse_args(args)


SERVER_TYPES = (
    'trtllm',
    'vllm',
    'openai',
    'gemini',
    'hf',
    'mamba',
)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device, compress_config):

    model_name_lower = model_name.lower()
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "").lower()
    is_chatglm = ("chatglm" in model_type) or ("glm" in model_name_lower)

    if compress_config.compress:
        if "llama" in model_name_lower:
            from cakekv.cake.monkeypatch import replace_flashllama_attn_with_cakeattn
            replace_flashllama_attn_with_cakeattn()
        elif "mistral" in model_name_lower:
            from cakekv.cake.monkeypatch import replace_flashmistral_attn_with_cakeattn
            replace_flashmistral_attn_with_cakeattn()
        elif "qwen2" in model_name_lower:
            from cakekv.cake.monkeypatch import replace_flashqwen2_attn_with_cakeattn
            replace_flashqwen2_attn_with_cakeattn()
        elif not is_chatglm:
            raise ValueError(f"Unsupported model_name: {model_name}. Must contain 'llama', 'mistral', or 'qwen2' (case-insensitive).")

    dtype = torch.bfloat16 if ("qwen2" in model_name.lower() or is_chatglm) else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=is_chatglm)
    attn_impl = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        trust_remote_code=is_chatglm,
    ).to(device)

    if compress_config.compress and is_chatglm:
        from cakekv.cake.monkeypatch import replace_chatglm_attn_with_cakeattn
        replace_chatglm_attn_with_cakeattn(model)

    if hasattr(config, 'num_hidden_layers'):
        layers = config.num_hidden_layers

    if compress_config.compress:
        if is_chatglm:
            target_layers = model.transformer.encoder.layers
            for i in range(layers):
                attn = target_layers[i].self_attention
                if not hasattr(attn, "config"):
                    attn.config = model.config
                attn.config.key_size = [compress_config.cache_size - compress_config.window_size] * layers
                attn.config.window_size = [compress_config.window_size] * layers
                attn.config.prefill = [True] * layers
                attn.config.decoding_evict = [None] * layers
                attn.config.tau1 = compress_config.hyper[0]
                attn.config.tau2 = compress_config.hyper[1]
                attn.config.gamma = compress_config.hyper[2]
                kv_heads = attn.num_multi_query_groups_per_partition if attn.multi_query_attention else attn.num_attention_heads_per_partition
                attn.config.prefill_cake_evict = [CakeprefillKVCache(
                    cache_size=compress_config.cache_size,
                    window_size=compress_config.window_size,
                    k_seq_dim=2,
                    v_seq_dim=2,
                    num_heads=kv_heads,
                    num_layers=layers,
                    use_cascading=compress_config.cascading,
                )] * layers
        else:
            for i in range(layers):
                model.model.layers[i].self_attn.config.key_size = [compress_config.cache_size - compress_config.window_size]*layers
                model.model.layers[i].self_attn.config.window_size = [compress_config.window_size]*layers
                model.model.layers[i].self_attn.config.prefill = [True]*layers
                model.model.layers[i].self_attn.config.decoding_evict = [None]*layers
                model.model.layers[i].self_attn.config.tau1 = compress_config.hyper[0]
                model.model.layers[i].self_attn.config.tau2 = compress_config.hyper[1] 
                model.model.layers[i].self_attn.config.gamma = compress_config.hyper[2] 
                model.model.layers[i].self_attn.config.prefill_cake_evict = [CakeprefillKVCache(
                    cache_size=compress_config.cache_size,
                    window_size=compress_config.window_size,
                    k_seq_dim=2,
                    v_seq_dim=2,
                    num_heads=model.model.layers[i].self_attn.num_heads,
                    num_layers=layers,
                    use_cascading=compress_config.cascading
                )]*layers

    model = model.eval()

    
    return model, tokenizer


class ServerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.server_type = values


def get_pred(
    llm,
    input_text: str,
    max_new_tokens: int,
    attn_type: str,
    model_name: str,
    budget_ratio: float,
    estimate_ratio: float,
    synthetic_len: int,
) -> str:
    
    llm.tokenizer.pad_token = llm.tokenizer.eos_token
    llm.tokenizer.padding_side = "left"
    inputs = llm.tokenizer([input_text], return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_masks = inputs.attention_mask

    attn_config = generate_config(
        model_name, 
        synthetic_len, 
        attn_type,
        budget_ratio=budget_ratio,
        estimate_ratio=estimate_ratio,
    )

    out = llm.generate(attention_type=attn_type,
        inputs_ids = input_ids.to(llm.layers[0].device),
        attention_masks = attention_masks.to(llm.layers[0].device),
        max_new_length=max_new_tokens, 
        attn_config=attn_config
    )

    output = llm.tokenizer.batch_decode(out[:,input_ids.shape[1]:], skip_special_tokens=True)
            
    print("Chunked generation:", output[0])
    return output[0]


def get_output(llm, outputs_parallel, idx, index, input, outputs, others, truncation, length):
    while True:
        try:
            pred = llm(prompt=input)
            break
        except Exception as e:
            traceback.print_exc()

    if len(pred['text']) > 0:
        outputs_parallel[idx] = {
            'index': index,
            'pred': pred['text'][0],
            'input': input,
            'outputs': outputs,
            'others': others,
            'truncation': truncation,
            'length': length,
        }


def main(args):
    start_time = time.time()
    
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    
    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")
        sys.exit(1)

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f"{args.task} is not found in {args.benchmark}.yaml. Available tasks: {list(tasks_customized.keys())}")
        
    config = tasks_customized.get(args.task)
    config.update(tasks_base[config['task']])

    task_file = args.data_dir / args.task / f'{args.subset}.jsonl'
    
    if args.chunk_amount > 1:
        pred_file = args.save_dir / f'{args.task}-{args.chunk_idx}.jsonl'
    else:
        pred_file = args.save_dir / f'{args.task}.jsonl'
        
    print(f'Predict {args.task} \nfrom {task_file}\nto {pred_file}')
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if os.path.exists(pred_file):
        pred_index = [sample['index'] for sample in load_data(pred_file)]
        data = [sample for sample in load_data(task_file) if sample['index'] not in pred_index]
    else:
        data = load_data(task_file)
    
    if args.limit is not None:
        data = data[:args.limit]
        
    # envs
    pred_name = args.pred_name
    model_name = args.model
    compress = args.compress
    cascading = args.cascading
    compress_config = CompressConfig(compress, cascading)
    max_length = args.synthetic_len


    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_config_path = os.path.join(script_dir, "../../experiments/LongBench/config/model2path.json")
    
    model2path = {}
    try:
        model2path = json.load(open(path_config_path, "r"))
    except FileNotFoundError:
        try:
            model2path = json.load(open("config/model2path.json", "r"))
        except FileNotFoundError:
            pass

 
    model_path = model_name
    if model_name in model2path:
        model_path = model2path[model_name]
        print(f"Mapping model name '{model_name}' to path '{model_path}'")
    elif model_name.lower() in [k.lower() for k in model2path.keys()]:
   
        for k, v in model2path.items():
            if k.lower() == model_name.lower():
                model_path = v
                print(f"Mapping model name '{model_name}' (case-insensitive) to path '{model_path}'")
                break
    elif any(k.lower() in model_name.lower() for k in model2path.keys()):

        for k, v in model2path.items():
            if k.lower() in model_name.lower():
                model_path = v
                print(f"Detected model key '{k}' in path, mapping to '{model_path}'")
                break

    if compress:
        compress_config.cache_size = args.cache_size
        compress_config.window_size = args.window_size
        cache_name = f"cache{args.cache_size}"
        
        config_path = os.path.join(script_dir, "../config/model2tau.json")
        
        try:
            model2tau = json.load(open(config_path, "r"))
        except FileNotFoundError:
             try:
                 model2tau = json.load(open("config/model2tau.json", "r"))
             except FileNotFoundError:
                 print(f"Warning: Could not find model2tau.json")
                 model2tau = {}
        
        try:
            # Try to match model name from path
            matched_key = None
            for key in model2tau.keys():
                if key.lower() in model_name.lower():
                    matched_key = key
                    break
            
            if matched_key:
                tau1 = model2tau[matched_key][f"{args.cache_size}"]["tau1"]
                tau2 = model2tau[matched_key][f"{args.cache_size}"]["tau2"]
            else:
                tau1 = model2tau[model_name][f"{args.cache_size}"]["tau1"]
                tau2 = model2tau[model_name][f"{args.cache_size}"]["tau2"]
            print(f"Loaded tau values for {matched_key or model_name}: {tau1}, {tau2}")
        except Exception as e:
            print(f"Error loading tau values: {e}")
            tau1, tau2 = 1.0, 1.0 
        gamma = args.gamma
        hyper = [tau1,tau2,gamma]
        compress_config.hyper = hyper
    else:
        cache_name ="cachefull"

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    llm, tokenizer = load_model_and_tokenizer(model_path, model_name, device, compress_config)

    tokens_to_generate = config["tokens_to_generate"]
    

    def get_output(idx, index, input, outputs, length, others=None, truncation=None, **kwargs):

        original_input = input  
        
        prompt = input
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        max_gen = tokens_to_generate
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                    tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        model_input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = model_input.input_ids.shape[-1]

       
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            **model_input,
            'max_new_tokens': max_gen,
            'num_beams': 1,
            'do_sample': False,
            'temperature': 1.0,
            'streamer': streamer,
        }

        t_start = time.perf_counter()
        thread = Thread(target=llm.generate, kwargs=gen_kwargs)
        thread.start()

        first_chunk_ts = None
        chunks = []
        for piece in streamer:
            if first_chunk_ts is None and piece:
                first_chunk_ts = time.perf_counter()
            chunks.append(piece)
        t_end = time.perf_counter()

        if compress:
            layers = len(llm.model.layers)
            for i in range(layers):
                llm.model.layers[i].self_attn.config.prefill = [True]*layers
                llm.model.layers[i].self_attn.config.decoding_evict = [None]*layers

        generated_text = "".join(chunks).strip()
        new_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))

     
        if len(generated_text) > 0:
            outputs_parallel[idx] = {
                'index': index,
            'pred': generated_text,
                'input': original_input,  
                'outputs': outputs,
                'others': others,
                'truncation': truncation,
                'length': length,
            }    

    outputs_parallel = [{} for _ in range(len(data))]
    for i, sample in tqdm(enumerate(data), total=len(data)):
        get_output(i, **sample)

    

    with open(pred_file, "at", encoding="utf-8") as f:
        for output in outputs_parallel:
            if output:
                f.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    main(args)