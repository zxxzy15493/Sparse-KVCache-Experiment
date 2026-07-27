import os
import sys
import json
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)
from model_hub import LlamaModel, QwenModel, GlmModel, DeepSeekQwenModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import generate_config, parse_attn_args

from utils import load_data

# Load config files
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
dataset2prompt = json.load(open(os.path.join(_CONFIG_DIR, "dataset2prompt.json"), "r"))
dataset2maxlen = json.load(open(os.path.join(_CONFIG_DIR, "dataset2maxlen.json"), "r"))
model2path = json.load(open(os.path.join(_CONFIG_DIR, "model2path.json"), "r"))
model2maxlen = json.load(open(os.path.join(_CONFIG_DIR, "model2maxlen.json"), "r"))

# Tasks that use direct generation (no chat template)
GENERATION_TASKS = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]

# Synthetic task max generation tokens
SYNTHETIC_TASKS = {
    'niah': 128,
    'vt': 30,
    'cwe': 120,
    'fwe': 50,
    'qa': 32
}

# Ruler data root
RULER_DATA_ROOT = "../../../../benchmarks/ruler/benchmark_root"

# Local model path -> short name for Ruler data directory
LOCAL_PATH_TO_SHORT = {
    "llama-3.1": "llama3.1-8b",
    "qwen-2.5": "qwen2.5-7b",
    "glm": "glm-4-9b",
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model short name (key in config/model2path.json)")
    parser.add_argument("--attn_type", type=str, default="Full_Flash_Attn",
                        choices=["Full_Flash_Attn", "RetroInfer"],
                        help="Attention method")
    parser.add_argument("--prefill_method", type=str, default="Full_Flash_Attn",
                        choices=["Full_Flash_Attn", "minfer"],
                        help="Attention method for prefill phase")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["LongBench", "Synthetic"],
                        help="Benchmark type: LongBench or Synthetic")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (e.g. narrativeqa, niah_single_1, vt, etc.)")
    parser.add_argument("--max_len", type=int, default=None,
                        help="Max context length. If None, use model2maxlen config value.")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save prediction results")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"],
                        help="Model dtype")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (e.g. cuda:0, auto)")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Number of examples to evaluate. -1 for all.")
    parser.add_argument("--fixed_output_length", type=int, default=0,
                        help="Fixed output length for decoding. 0 means use task default.")
    parser.add_argument("--recall", action="store_true",
                        help="Enable recall measurement")
    parser.add_argument("--measure_time", action="store_true",
                        help="Enable detailed time measurement")

    parser = parse_attn_args(parser)

    return parser.parse_args(args)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_dataset(args):
    """Load data based on benchmark type, with resume support.

    Skips samples that have already been predicted (based on out_path).

    Returns:
        data: list of remaining data samples with 'input' field ready for generation
        max_new_tokens: max generation length for this task
        out_path: path to save predictions
    """
    model_name = args.model_name
    task = args.task
    benchmark = args.benchmark

    # Prepare output path
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"{task}.jsonl")

    if benchmark == "LongBench":
        from datasets import load_dataset as hf_load_dataset

        data = [dict(sample) for sample in hf_load_dataset("THUDM/LongBench", task, split="test")]
        max_new_tokens = dataset2maxlen.get(task, 128)

        # Apply chat template for non-generation tasks
        from transformers import AutoTokenizer
        model_path = model2path[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        prompt_format = dataset2prompt.get(task, "{input}")
        for i in range(len(data)):
            json_obj = data[i]
            prompt = prompt_format.format(**json_obj)
            if task not in GENERATION_TASKS:
                message = [{"role": "user", "content": prompt}]
                prompt = tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            data[i]["input"] = prompt

    elif benchmark == "Synthetic":
        model_short = LOCAL_PATH_TO_SHORT.get(model_name)
        if model_short is None:
            for key, val in LOCAL_PATH_TO_SHORT.items():
                if key in model_name.lower() or model_name.lower() in key:
                    model_short = val
                    break
        if model_short is None:
            raise ValueError(f"Cannot determine model short name for: {model_name}. "
                             f"Add it to LOCAL_PATH_TO_SHORT in pred.py.")

        task_file = f'{RULER_DATA_ROOT}/{model_short}/synthetic/65536/data/{task}/validation.jsonl'
        if not os.path.exists(task_file):
            raise FileNotFoundError(f"Synthetic task file not found: {task_file}")

        data = load_data(task_file)

        # Determine max_new_tokens from task type
        task_key = next((key for key in SYNTHETIC_TASKS.keys() if key in task), None)
        max_new_tokens = SYNTHETIC_TASKS.get(task_key, 128)

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    # Apply num_samples limit
    if args.num_samples > 0:
        data = data[:args.num_samples]

    # Resume support: skip already predicted indices
    pred_indices = set()
    if os.path.exists(out_path):
        for sample in load_data(out_path):
            pred_indices.add(sample.get("index", ""))

    remaining = []
    for sample in data:
        idx = sample.get("_id", "") if benchmark == "LongBench" else sample.get("index", "")
        if idx not in pred_indices:
            remaining.append(sample)

    print(f"[{task}] {len(remaining)} samples to predict, {len(pred_indices)} already done, max_new_tokens: {max_new_tokens}")

    return remaining, max_new_tokens, out_path


def load_model(model_name, max_len, dtype, device, args):
    """Load model via RetroInfer model_hub."""
    model_path = model2path[model_name]

    if 'DeepSeek' in model_path or 'deepseek' in model_path:
        llm = DeepSeekQwenModel(
            model_path,
            max_length=max_len,
            dtype=dtype,
            device_map=device,
        )
    elif 'Llama' in model_path:
        llm = LlamaModel(
            model_path,
            max_length=max_len,
            dtype=dtype,
            device_map=device,
        )
    elif 'Qwen' in model_path:
        llm = QwenModel(
            model_path,
            max_length=max_len,
            dtype=dtype,
            device_map=device,
        )
    elif 'GLM' in model_path or 'glm' in model_path:
        llm = GlmModel(
            model_path,
            max_length=max_len,
            dtype=dtype,
            device_map=device,
        )
    else:
        raise ValueError(f"Unsupported model: {model_path}")

    if llm.tokenizer.eos_token is not None:
        llm.tokenizer.pad_token = llm.tokenizer.eos_token
    elif llm.tokenizer.pad_token_id is None and len(llm.eos_tokens) > 0:
        llm.tokenizer.pad_token_id = llm.eos_tokens[0]
    llm.tokenizer.padding_side = "left"

    return llm


def get_pred(llm, data_sample, max_new_tokens, out_path, args):
    """Run prediction on a single data sample and save to jsonl."""
    prompt = data_sample["input"]
    model_path = model2path[args.model_name]

    inputs = llm.tokenizer([prompt], return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_masks = inputs.attention_mask

    input_len = input_ids.shape[1]
    print(f"Input token length: {input_len}")

    # Generate attention config dynamically
    attn_config = generate_config(
        model_path,
        input_len,
        args.attn_type,
        budget_ratio=args.budget_ratio,
        budget=args.budget,
        estimate_ratio=args.estimate_ratio,
        ratio_or_fixed=args.ratio_or_fixed,
    )

    if args.fixed_output_length > 0:
        gen_length = args.fixed_output_length
    else:
        gen_length = max_new_tokens

    out = llm.generate(
        attention_type=args.attn_type,
        inputs_ids=input_ids.to(llm.layers[0].device),
        attention_masks=attention_masks.to(llm.layers[0].device),
        max_new_length=gen_length,
        attn_config=attn_config,
        prefill_method=args.prefill_method,
    )

    generated_len = len(out[0])
    print(f"Output token length: {generated_len}")

    output = llm.tokenizer.batch_decode(out, skip_special_tokens=True)
    pred = output[0]

    torch.cuda.empty_cache()

    # Determine index key based on benchmark
    if args.benchmark == "LongBench":
        index = data_sample.get("_id", "")
    else:
        index = data_sample.get("index", "")

    with open(out_path, "a", encoding="utf-8") as f:
        json.dump(
            {
                "pred": pred,
                "answers": data_sample["answers"],
                "all_classes": data_sample["all_classes"],
                "length": data_sample["length"],
                "index": index,
            },
            f,
            ensure_ascii=False,
        )
        f.write("\n")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    # Determine max context length
    if args.max_len is not None:
        max_len = args.max_len
    else:
        max_len = model2maxlen.get(args.model_name, 130000)

    # Load data (with resume support)
    data, max_new_tokens, out_path = load_dataset(args)

    if len(data) == 0:
        print(f"[{args.task}] All examples already predicted, skipping.")
        sys.exit(0)

    # Load model
    print(f"Loading model: {args.model_name}")
    llm = load_model(args.model_name, max_len, dtype, args.device, args)
    print(f"Model loaded.")

    # Predict
    for data_sample in tqdm(data, desc=f"Predicting {args.task}"):
        get_pred(llm, data_sample, max_new_tokens, out_path, args)

    print(f"[{args.task}] Predictions saved to {out_path}")
