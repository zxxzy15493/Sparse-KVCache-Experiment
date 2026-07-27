import warnings

warnings.filterwarnings("ignore")

import torch
import argparse
import json
import os
import time
import re
import numpy as np
import random
import sys

from transformers import AutoConfig, AutoTokenizer
from tqdm import tqdm

from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from transformers.dynamic_module_utils  import get_class_from_dynamic_module


def iter_attention_modules(model):
    """Yield patched attention modules across llama/qwen/glm models."""
    target_names = {
        "SelfAttention",
        "LlamaAttention",
        "LlamaSdpaAttention",
        "LlamaFlashAttention2",
        "Qwen2Attention",
        "Qwen2SdpaAttention",
        "Qwen2FlashAttention2",
    }
    for _, module in model.named_modules():
        if module.__class__.__name__ in target_names:
            yield module

def reset_sample_cache_state(model):
    model_type = getattr(getattr(model, "config", None), "model_type", "")

    if "llama" in model_type:
        import keyformer_kv.modify_llama as modify_llama
    elif "glm" in model_type:
        import keyformer_kv.modify_glm as modify_glm

    for attn in iter_attention_modules(model):
        if hasattr(attn, "kv_cache") and attn.kv_cache is not None and hasattr(attn.kv_cache, "key_score"):
            attn.kv_cache.key_score = None
        if hasattr(attn, "kv_cache_evictor") and attn.kv_cache_evictor is not None and hasattr(attn.kv_cache_evictor, "hh_score"):
            attn.kv_cache_evictor.key_score = None

def load(model_name_or_path, args):
    print(f"Loading model from {model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
    )
    if "glm" in model_name_or_path.lower():

        config_cls = get_class_from_dynamic_module(
            "configuration_chatglm.ChatGLMConfig", model_name_or_path
        )
        model_cls = get_class_from_dynamic_module(
            "modeling_chatglm.ChatGLMForConditionalGeneration", model_name_or_path
        )    
        config = config_cls.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = model_cls.from_pretrained(
            model_name_or_path,
            config=config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    model.eval()
    return model, tokenizer

@torch.no_grad()
def keyformer_inference(model,model_name_or_path, tokenizer,dataset,dataset_name, samples,output_path,prompt_format,max_gen_len=1000):
    device = model.device
    done_count = 0
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): 
                    done_count += 1
    remaining_samples = samples[done_count:]
    with open(output_path, "a", encoding="utf-8") as fout:
        for  sample in tqdm(remaining_samples):
            reset_sample_cache_state(model)
            if "ruler" in dataset:  
                prompt = prompt_format.format(**sample)
                answers = sample.get("outputs", [])
                all_classes = sample.get("all_classes", []) 
                length = sample.get("length", len(sample.get("input", "")))
            elif "LongBench" in dataset:
                prompt = prompt_format.format(**sample)
                answers = sample["answers"]
                all_classes = sample["all_classes"]
                length = sample["length"]
            if "LongBench" in dataset and dataset_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                prompt = build_chat(tokenizer, prompt, model_name=model_name_or_path)
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input.input_ids.shape[1]
            print(f"\nSample Prompt Length: {context_length} tokens")
            if dataset_name == "samsum": 
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen_len,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen_len,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                )[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True).strip()


            with open(output_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": answers , "all_classes": all_classes, "length": length}, f, ensure_ascii=False)
                f.write('\n')

def build_chat(tokenizer, prompt, model_name):
   
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
                                    messages,
                                    tokenize=False,
                                    add_generation_prompt=True
                                )

    return prompt

def build_output_path(model_name_or_path, dataset_name, dataset_path=None):

    base_output = "./output"

    model_dir_name = os.path.basename(model_name_or_path.rstrip("/"))

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
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path,args)
    data_path=os.path.join(args.data_root, args.dataset_name + ".jsonl")
    print(f"Loading data from {data_path} ...")
    data = load_dataset("json", data_files=data_path,split="train")

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset_name=args.dataset_name
    data_root=args.data_root
    prompt_format = dataset2prompt[dataset_name]
    dataset2maxlen=json.load(open("config/dataset2maxlen.json", "r"))
    max_gen_len = dataset2maxlen[dataset_name]

    data_all = [data_sample for data_sample in data]
    output_path = build_output_path(
        args.model_name_or_path,
        args.dataset_name,
        args.data_root
    )
    print(f"Predictions will be saved to {output_path} ...")
    if args.keyformer:
        from keyformer_kv.enable_keyformer import enable_keyformer
        enable_keyformer(model, args)
        print(f"KeyFormer-LayerWise: {args.key_size}, {args.recent_size}")
        

    keyformer_inference(
        model,
        model_name_or_path,
        tokenizer,
        data_root,
        dataset_name,
        data_all,
        output_path,
        prompt_format, 
        max_gen_len=max_gen_len,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--data_root", type=str, default="LongBench")
    parser.add_argument("--keyformer", action="store_true")
    parser.add_argument("--recent_size", type=int, default=32)
    parser.add_argument("--dataset_name", type=str, default="qasper")
    parser.add_argument("--output_path", type=str, default="predictions.jsonl")
    parser.add_argument("--max_gen_len", type=int, default=1000)
    parser.add_argument("--key_size", type=int, default=992)
    parser.add_argument("--tau_init", type=float, default=1.0)
    parser.add_argument("--tau_delta", type=float, default=0.01)

    args = parser.parse_args()

    main(args)
