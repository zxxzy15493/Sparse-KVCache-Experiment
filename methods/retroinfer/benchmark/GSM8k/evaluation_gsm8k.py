from email.mime import text
import os
import sys
import torch
import json
from tqdm import tqdm
import numpy as np
import random
from pathlib import Path
import threading
import subprocess
import argparse
from examples import get_examples
from utils import load_data
import re
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)
from model_hub import LlamaModel, QwenModel, GlmModel, DeepSeekQwenModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import generate_config, parse_attn_args

model2path = json.load(open("./config/model2path.json", "r"))
model2maxlen = json.load(open("./config/model2maxlen.json", "r"))
# we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
dataset2prompt = json.load(open("./config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("./config/dataset2maxlen.json", "r"))
import json

TASKS = {
    'niah': 128,
    'vt': 30,
    'cwe': 120,
    'fwe': 50,
    'qa': 32
}

EXAMPLES = get_examples()

def load_prompt(prompt_name, num_shots):
    if not num_shots:
        return []
    return EXAMPLES[prompt_name][:num_shots]

def construct_prompt(example, args):
    demos = load_prompt('gsm8k-cot', 8)
    demo_prompt = "".join(
        [
            q + "\n" + a
            for q, a in demos
        ]
    )
    return demo_prompt + "\nQuestion: " + example["question"] + "\n"

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument("--attn_type", type=str, default="Full_Flash_Attn",                                                     \
                        choices=["Full_Flash_Attn", "RetroInfer"],                          \
                        help="Attention method")
    parser.add_argument("--prefill_method", type=str, default="Full_Flash_Attn",                                                     \
                        choices=["Full_Flash_Attn", "minfer"],                          \
                        help="Attention method for prefill phase, which determines the attention method used during the prefill phase. When set to 'minfer', it will use the minfer method with the best patterns for each layer. When set to 'Full_Flash_Attn', it will use full flash attention during the prefill phase.")
    parser.add_argument("--max_len", type=int, default=1024, help="Length of the context for attention computation")
    parser.add_argument("--data_dir", type=Path, default="", help="Data directory")
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Dtype")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--num_shots", type=int, default=8, help="number of shots for few-shot prompting")
    parser.add_argument("--cot_type", type=str, default="gsm8k-cot", help="type of chain-of-thought prompting")
    parser.add_argument("--fixed_output_length", type=int, default=0, required=False)
    parser.add_argument("--recall",action="store_true", required=False)
    parser.add_argument("--measure_time", action="store_true", required=False)
    parser = parse_attn_args(parser)

    return parser.parse_args(args)

def load_dataset(dataset, pred_dir, prompt_cot=None):
    # load data
    data_file = f'./data/{dataset}.jsonl'
    datas = load_data(data_file)
    for i, data in enumerate(datas):
        data.setdefault('index', i)

    out_path = Path(pred_dir) / "gsm8k.jsonl"

    if os.path.exists(out_path):
        pred_index = [sample["index"] for sample in load_data(out_path)]
        data = [sample for sample in datas if sample["index"] not in pred_index]
    else:
        data = datas

    return data

def get_pred(llm, message, data, max_new_tokens, model_name, out_path, args):  
    if llm.tokenizer.eos_token is not None:
        llm.tokenizer.pad_token = llm.tokenizer.eos_token
    elif llm.tokenizer.pad_token_id is None and len(llm.eos_tokens) > 0:
        llm.tokenizer.pad_token_id = llm.eos_tokens[0]
    llm.tokenizer.padding_side = "left"

    prompt = [{"role": "user", "content": message }]
    prompt = llm.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    print(prompt)

    inputs = llm.tokenizer([prompt], return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_masks = inputs.attention_mask

    attn_config = generate_config(
        model_name, 
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

    print("out length: is ", len(out[0]))
    output = llm.tokenizer.batch_decode(out, skip_special_tokens=True)

    torch.cuda.empty_cache()
            
    out_path = Path(out_path) / f"gsm8k.jsonl"
    pred = output[0]

    # （re.search：）
    pattern = r'<answer>(.*?)</answer>'

    # 2. （re.DOTALL .）
    match_answer = re.search(pattern, pred)
    if match_answer:
        match_answer = match_answer.group(1)

    pattern_final_answer = r"#### (\d{1,3}(?:,\d{3})*(?:\.?\d+)?)"
    final_answer = re.search(pattern_final_answer, data['answer'])
    if final_answer:
        final_answer = final_answer.group(1)

    with open(out_path, "a", encoding="utf-8") as f:
        json.dump(
            {
                "index": data.get("index"),
                "match_result": match_answer,
                "final_answer": final_answer,
                "pred": pred,
                'answer': data['answer'],
                'question': data['question']
            }, 
            f, 
            ensure_ascii=False
        )
        f.write('\n')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model(model_path, max_len, dtype, device, max_new_tokens, args):
    if 'Llama' in model_path:
        llm = LlamaModel(model_path,
            max_length=max_len + max_new_tokens,
            dtype=dtype,
            device_map=device,
            RECALL=args.recall,
            fixed_output_length=args.fixed_output_length,
            measure_time=args.measure_time
            )
    elif 'Qwen' in model_path:
        llm = QwenModel(model_path,
            max_length=max_len + max_new_tokens,
            dtype=dtype,
            device_map=device,
            # RECALL=args.recall,
            fixed_output_length=args.fixed_output_length,
            # measure_time=args.measure_time
            )
    elif 'deepseek' in model_path.lower():
        llm = DeepSeekQwenModel(model_path,
            max_length=max_len + max_new_tokens,
            dtype=dtype,
            device_map=device,
            # RECALL=args.recall,
            fixed_output_length=args.fixed_output_length,
            measure_time=args.measure_time
            )
    elif 'GLM' in model_path or 'glm' in model_path:
        llm = GlmModel(model_path,
            max_length=max_len + max_new_tokens,
            dtype=dtype,
            device_map=device,
            RECALL=args.recall,
            fixed_output_length=args.fixed_output_length,
            measure_time=args.measure_time
            )
    else:
        raise ValueError(f"Unsupported model: {model_path}")

    if llm.tokenizer.eos_token is not None:
        llm.tokenizer.pad_token = llm.tokenizer.eos_token
    elif llm.tokenizer.pad_token_id is None and len(llm.eos_tokens) > 0:
        llm.tokenizer.pad_token_id = llm.eos_tokens[0]
    llm.tokenizer.padding_side = "left"
    
    return llm

if __name__ == "__main__":
    seed_everything(42)    
    args = parse_args()

    # Setup output dir
    dataset = "gsm8k_test"
    
    model_name = args.model_name 
    device = args.device
    dtype = torch.bfloat16
    pred_dir = args.save_dir

    max_length = model2maxlen[model_name]
    model_path = model2path[model_name]
    max_new_tokens = 3000

    # Preprocessing the dataset.
    gsm8k_datas = load_dataset(dataset, pred_dir)

    # Load Model and Tokenizer
    llm = load_model(model_path, max_length, dtype, device, max_new_tokens, args)

    for data_sample in tqdm(gsm8k_datas):
            full_prompt = construct_prompt(data_sample, args)
            get_pred(
                llm,
                full_prompt,
                data_sample,
                max_new_tokens,
                model_path,
                pred_dir,
                args,
            )
