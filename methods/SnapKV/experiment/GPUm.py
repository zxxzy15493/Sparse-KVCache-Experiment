import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import os
import time
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snapkv.monkeypatch.monkeypatch import replace_glm, replace_llama, replace_qwen

def load(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model = model.eval()
    return model, tokenizer

def apply_snapkv_to_model(model, args):
    model_name = args.model_name_or_path.lower()
    if "llama" in model_name:
        replace_llama()
    if "qwen" in model_name:
        replace_qwen()
    model_type = getattr(getattr(model, "config", None), "model_type", "").lower()
    if "glm" in model_name or "glm" in model_type:
        replace_glm(model)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model_layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "encoder") and hasattr(model.transformer.encoder, "layers"):
        model_layers = model.transformer.encoder.layers
    else:
        raise ValueError("Could not find layers in model")
    for layer in model_layers:
        attn = getattr(layer, "self_attn", getattr(layer, "self_attention", None))
        if attn is None:
            continue
        attn.config.window_size = args.window_size
        attn.config.max_capacity_prompt = args.max_capacity_prompt
        attn.config.kernel_size = args.kernel_size

def build_chat(tokenizer, prompt, model_name):
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def build_output_path(model_name_or_path, out_len, budget):
    model_dir_name = os.path.basename(model_name_or_path.rstrip("/"))
    base_output = "./memory"
    output_dir = os.path.join(base_output, f"budget{budget}", model_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"out{out_len}.jsonl"
    return os.path.join(output_dir, file_name)

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
    model_key = args.model_name_or_path
    model2path = json.load(open("config/model2path.json", "r"))
    actual_model_path = model_key

    if args.compress_args_path:
        with open(args.compress_args_path, "r", encoding="utf-8") as f:
            compress_args = json.load(f)
        args.window_size = compress_args.get("window_sizes")[0] if isinstance(compress_args.get("window_sizes"), list) else compress_args.get("window_sizes")
        args.max_capacity_prompt = compress_args.get("max_capacity_prompts")[0] if isinstance(compress_args.get("max_capacity_prompts"), list) else compress_args.get("max_capacity_prompts")
        args.kernel_size = compress_args.get("kernel_sizes")[0] if isinstance(compress_args.get("kernel_sizes"), list) else compress_args.get("kernel_sizes")
        args.pooling = compress_args.get("pooling")

    if args.compress_args_path:
        model_path_lower = actual_model_path.lower()
        if "llama" in model_path_lower:
            replace_llama()
        if "qwen" in model_path_lower:
            replace_qwen()

    model, tokenizer = load(actual_model_path)

    if args.compress_args_path:
        apply_snapkv_to_model(model, args)

    with open(args.data_root, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()
    
    input_ids_full = tokenizer(raw_text, return_tensors="pt").input_ids
    
    in_len = args.test_len
    total_budget = getattr(args, "max_capacity_prompt", "full")
    output_path = build_output_path(args.model_name_or_path, args.out_len, total_budget)

    if in_len > input_ids_full.shape[1]:
        print("Warning: test_len exceeds available text length.")
        return

    print(f"Run 1 - Input Length: {in_len}, Budget: {total_budget}")
    input_ids = input_ids_full[:, :in_len].to(model.device)
    past_key_values, current_input_ids = None, input_ids

    torch.cuda.memory.reset_peak_memory_stats()
    if hasattr(torch.cuda.memory, "_record_memory_history"):
        torch.cuda.memory._record_memory_history()

    with torch.no_grad():
        for i in range(args.out_len):
            outputs = model(input_ids=current_input_ids, past_key_values=past_key_values, use_cache=True, num_logits_to_keep=1)
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            current_input_ids = next_token_id

    peak_alloc = torch.cuda.memory.max_memory_allocated()
    peak_alloc_mib = peak_alloc / 1024**2
    print(f"peak_alloc    = {peak_alloc_mib:.2f} MiB")

    save_data = {
        "run": 1,  
        "in_len": in_len,
        "budget": total_budget,
        "peak_alloc_mib": peak_alloc_mib
    }

    with open(output_path, "a", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False)
        f.write("\n")

    del past_key_values
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data_root", type=str, default="myinput.txt")
    parser.add_argument("--compress_args_path", type=str, default=None)
    parser.add_argument("--out_len", type=int, default=32)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--test_len", type=int, default=4096)
    args = parser.parse_args()
    main(args)