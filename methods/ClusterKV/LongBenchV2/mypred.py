import os
import re
import json
import argparse
import random
import time
from typing import Dict, Any, List, Optional
from requests.exceptions import ProxyError, SSLError

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    AutoConfig,
)

# ---------------------------------------------------------
# Import Patches and Cluster library specific to the second code
# ---------------------------------------------------------
from accuracy.patch import parse_common_args, enable_attention_eval, get_config_output_affix
from accuracy.cluster_attention import cluster_reset

# ---------------------------------------------------------
# First code: Prompt template loading
# ---------------------------------------------------------
try:
    template_rag = open("prompts/0shot_rag.txt", encoding="utf-8").read()
    template_no_context = open("prompts/0shot_no_context.txt", encoding="utf-8").read()
    template_0shot = open("prompts/0shot.txt", encoding="utf-8").read()
    template_0shot_cot = open("prompts/0shot_cot.txt", encoding="utf-8").read()
    template_0shot_cot_ans = open("prompts/0shot_cot_ans.txt", encoding="utf-8").read()
except FileNotFoundError:
    print("[Warning] Prompt files not found in 'prompts/'. Please ensure they exist.")
    template_rag, template_no_context, template_0shot, template_0shot_cot, template_0shot_cot_ans = "", "", "", "", ""

TOPP_SAVE_TOPK = os.environ.get("TOPP_SAVE_TOPK") 

# ---------------------------------------------------------
# Common helper functions (from the first code)
# ---------------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def extract_answer(response: str) -> Optional[str]:
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)

    match = re.search(r"The correct answer is ([A-D])", response)
    if match:
        return match.group(1)

    return None

def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device

def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")

# ---------------------------------------------------------
# Hybrid implementation: argument parsing (built on the second code's parse_common_args)
# ---------------------------------------------------------
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # inherit argument config from the second code (e.g. --model, --cluster, --quest)
    parser = parse_common_args(parser)
    
    # additional LongBench-V2-specific arguments from the first code
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--model_path", type=str, default=None, help="Model path; defaults to config/model2path.json if unset")
    parser.add_argument("--data_file", type=str, default="../../../benchmarks/longbenchv2/filtered_longbench_v2_64k-192k.jsonl", help="LongBenchV2 dataset path")
    parser.add_argument("--split", type=str, default="train")

    parser.add_argument("--max_context_len", type=int, default=204800)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--cot_max_new_tokens", type=int, default=1024)

    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--no_context", action="store_true")
    parser.add_argument("--rag", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    
    parsed_args = parser.parse_args(args)
    return parsed_args

# ---------------------------------------------------------
# Second code: model loading and Patch
# ---------------------------------------------------------
def load_model_and_tokenizer(path, model_name, device, args):
    print(f"Loading model from: {path}")
    if "glm4" in model_name:
        import importlib.util
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        spec = importlib.util.spec_from_file_location(
            "modeling_chatglm", os.path.join(path, "modeling_chatglm.py")
        )
        modeling_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modeling_module)
        model_class = getattr(modeling_module, config.auto_map["AutoModelForCausalLM"].split(".")[-1])
        model = model_class.from_pretrained(
            path, config=config, torch_dtype=torch.bfloat16,
            device_map="auto", low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2"
        ).to(device)
    elif "intern" in model_name or "qwen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map="auto", low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2", use_cache=True
        ).to(device)
    elif "llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2", use_cache=True
        )
    else:
        # Fallback to standard loading
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
    model = model.eval()

    # apply Quest or Cluster attention Patch (from the second code)
    if getattr(args, 'quest', False) or getattr(args, 'cluster', False):
        print(f"Applying attention evaluation patch for: {model_name}")
        enable_attention_eval(model_name, model, args)

    return model, tokenizer

def load_model_with_retry(model_path, model_name, device, args, retries=3, delay=1):
    for attempt in range(retries):
        try:
            return load_model_and_tokenizer(model_path, model_name, device, args)
        except (ProxyError, SSLError) as e:
            print(f"Attempt {attempt + 1} failed due to network error: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise

# ---------------------------------------------------------
# First code: data and Prompt handling logic
# ---------------------------------------------------------
def build_chat_prompt(tokenizer, prompt: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        return prompt

def truncate_token_ids(input_ids: List[int], max_len: int) -> List[int]:
    if len(input_ids) <= max_len:
        return input_ids
    keep_front = max_len // 2
    keep_back = max_len - keep_front
    return input_ids[:keep_front] + input_ids[-keep_back:]

def encode_prompt(tokenizer, prompt: str, max_context_len: int, device: torch.device):
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = truncate_token_ids(input_ids, max_context_len)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

def build_base_prompt(item: Dict[str, Any], args) -> str:
    context = item["context"]
    if args.rag > 0:
        template = template_rag
        retrieved = item["retrieved_context"][:args.rag]
        retrieved = sorted(retrieved, key=lambda x: x["c_idx"])
        context = "\n\n".join(
            [f"Retrieved chunk {idx + 1}: {x['content']}" for idx, x in enumerate(retrieved)]
        )
    elif args.no_context:
        template = template_no_context
    elif args.cot:
        template = template_0shot_cot
    else:
        template = template_0shot

    prompt = (
        template.replace("$DOC$", context.strip())
        .replace("$Q$", item["question"].strip())
        .replace("$C_A$", item["choice_A"].strip())
        .replace("$C_B$", item["choice_B"].strip())
        .replace("$C_C$", item["choice_C"].strip())
        .replace("$C_D$", item["choice_D"].strip())
    )
    return prompt

def generate_text(model, tokenizer, prompt: str, max_context_len: int, max_new_tokens: int, args) -> str:
    device = get_model_device(model)
    chat_prompt = build_chat_prompt(tokenizer, prompt)
    inputs = encode_prompt(tokenizer, chat_prompt, max_context_len, device)
    prompt_len = inputs["input_ids"].shape[-1]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    # key point: when using Cluster, reset state before each generation (from the second code)
    if getattr(args, 'cluster', False):
        cluster_reset(model)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            top_p=1.0,
            top_k=0,
            temperature=1.0,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]

    gen_ids = output_ids[prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    torch.cuda.empty_cache()
    return text

def load_data(args):
    dataset = load_dataset("json", data_files=args.data_file, split=args.split)
    data_all = []
    for item in dataset:
        sample = {
            "_id": item["_id"],
            "domain": item["domain"],
            "sub_domain": item["sub_domain"],
            "difficulty": item["difficulty"],
            "length": item["length"],
            "question": item["question"],
            "choice_A": item["choice_A"],
            "choice_B": item["choice_B"],
            "choice_C": item["choice_C"],
            "choice_D": item["choice_D"],
            "answer": item["answer"],
            "context": item["context"],
        }
        if "retrieved_context" in item:
            sample["retrieved_context"] = item["retrieved_context"]
        data_all.append(sample)
    return data_all

# ---------------------------------------------------------
# First code: core inference loop
# ---------------------------------------------------------
def run_longbenchv2_pred(model, tokenizer, data, args, fout):
    _id = 0
    for item in tqdm(data, desc="Running prediction"):
        if TOPP_SAVE_TOPK is not None:
            tmp = TOPP_SAVE_TOPK
            with open(f"record/{tmp}.txt", "a") as f:
                f.write(f"data:{_id}\n")
        try:
            prompt = build_base_prompt(item, args)

            if args.cot:
                cot_response = generate_text(
                    model=model, tokenizer=tokenizer, prompt=prompt,
                    max_context_len=args.max_context_len, max_new_tokens=args.cot_max_new_tokens, args=args
                )

                final_prompt = (
                    template_0shot_cot_ans
                    .replace("$DOC$", item["context"].strip())
                    .replace("$Q$", item["question"].strip())
                    .replace("$C_A$", item["choice_A"].strip())
                    .replace("$C_B$", item["choice_B"].strip())
                    .replace("$C_C$", item["choice_C"].strip())
                    .replace("$C_D$", item["choice_D"].strip())
                    .replace("$COT$", cot_response)
                )

                final_response = generate_text(
                    model=model, tokenizer=tokenizer, prompt=final_prompt,
                    max_context_len=args.max_context_len, max_new_tokens=args.max_new_tokens, args=args
                )

                output_item = dict(item)
                output_item["response_cot"] = cot_response
                output_item["response"] = final_response
            else:
                response = generate_text(
                    model=model, tokenizer=tokenizer, prompt=prompt,
                    max_context_len=args.max_context_len, max_new_tokens=args.max_new_tokens, args=args
                )

                output_item = dict(item)
                output_item["response"] = response

            output_item["pred"] = extract_answer(output_item["response"])
            output_item["judge"] = output_item["pred"] == output_item["answer"]
            output_item["context"] = output_item["context"][:1000]

            # preserve the original output field order
            ordered_keys = ["response", "pred", "answer", "judge", "response_cot", "_id", "domain", "sub_domain", "difficulty", "length", "choice_A", "choice_B", "choice_C", "choice_D", "question", "context"]
            ordered_item = {k: output_item[k] for k in ordered_keys if k in output_item}
            for k, v in output_item.items():
                if k not in ordered_item:
                    ordered_item[k] = v

            fout.write(json.dumps(ordered_item, ensure_ascii=False) + "\n")
            fout.flush()

        except Exception as e:
            print(f"[Error] sample_id={item.get('_id', 'unknown')} failed: {e}")
        _id += 1

# ---------------------------------------------------------
# main function
# ---------------------------------------------------------
def main():
    args = parse_args()
    seed_everything(args.seed)
    
    # mutual exclusion check
    if getattr(args, 'quest', False) and getattr(args, 'cluster', False):
        raise ValueError("Cannot enable both quest and cluster at the same time")

    # determine model name and path
    model_name = args.model
    if args.model_path is not None:
        model_path = args.model_path
    else:
        # if --model_path is not on the CLI, read it from the second code's config file
        try:
            model2path = json.load(open("../config/model2path.json", "r"))
            model_path = model2path[model_name]
        except Exception:
            raise ValueError("model_path is not provided and failed to load from ../config/model2path.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Parsed Arguments: {args}")

    os.makedirs(args.save_dir, exist_ok=True)
    if TOPP_SAVE_TOPK is not None:
        tmp = TOPP_SAVE_TOPK
        os.makedirs("record", exist_ok=True)
        with open(f"record/{tmp}.txt", "a") as f:
            f.write("begin\n")

    # build the output filename (merging suffix logic from both codes)
    suffix_parts = []
    if args.rag > 0:
        suffix_parts.append(f"rag_{args.rag}")
    if args.no_context:
        suffix_parts.append("no_context")
    if args.cot:
        suffix_parts.append("cot")
        
    lb_v2_suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    config_affix = get_config_output_affix(args) # method from the second code (carries quest/cluster info)

    out_file = os.path.join(
        args.save_dir,
        f"{safe_model_name(model_name)}{lb_v2_suffix}{config_affix}.jsonl"
    )

    data_all = load_data(args)

    # checkpoint resume logic
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    has_data[json.loads(line)["_id"]] = 0

    data = [item for item in data_all if item["_id"] not in has_data]

    print(f"Total samples   : {len(data_all)}")
    print(f"Remaining samples: {len(data)}")
    print(f"Output file     : {out_file}")

    if len(data) == 0:
        print("No remaining samples to run.")
        return

    # load model
    model, tokenizer = load_model_with_retry(model_path, model_name, device, args)

    # run prediction
    with open(out_file, "a", encoding="utf-8") as fout:
        run_longbenchv2_pred(
            model=model,
            tokenizer=tokenizer,
            data=data,
            args=args,
            fout=fout,
        )

if __name__ == "__main__":
    main()