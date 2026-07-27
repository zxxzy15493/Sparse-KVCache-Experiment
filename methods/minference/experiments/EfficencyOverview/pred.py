# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import time
import numpy as np
from pathlib import Path
import sys

# Ensure the repository-local `minference_time/` package is importable when this
# script is launched from `experiments/EfficencyOverview`.
# File layout: <repo>/experiments/EfficencyOverview/pred.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minference_time import MInference

# we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
dataset2prompt = json.load(open("./config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("./config/dataset2maxlen.json", "r"))


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="path to save the prediction jsonl files",
    )
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--fixed_output_length", type=int, default=0, required=False)
    parser.add_argument("--input_max_token", type=int, default=1024)
    # MInference
    parser.add_argument("--model_name", type=str, required=True, help="model name or path")
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--starting_layer", type=int, default=-1)
    parser.add_argument("--kv_type", type=str, default="dense")
    parser.add_argument("--trust_remote_code", action="store_true")

    return parser.parse_args(args)


def load_dataset(args):
    with open("../../../../benchmarks/myinput.txt", "r", encoding="utf-8") as f:
        data = f.read()  # core: read() reads all content into data at once
    return data


def load_model(model_name):
    """Load model with MInference sparse attention patch."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cuda",
        _attn_implementation="flash_attention_2",
    )
    model.eval()

    # Apply MInference patch
    minference_patch = MInference(model_name)
    model = minference_patch(model)

    return model, tokenizer


def get_pred(model, tokenizer, data, args):
    """Run efficiency benchmark: manual decode loop with timing over multiple runs."""
    fixed_output_length = args.fixed_output_length
    input_max_token = args.input_max_token

    inputs = tokenizer(
        data, return_tensors="pt", return_attention_mask=True
    ).to(model.device)
    inputs['input_ids'] = inputs.input_ids[:, :input_max_token]
    if "attention_mask" in inputs:
        inputs['attention_mask'] = inputs.attention_mask[:, :input_max_token]

    model_chao = 'llama' if 'llama' in args.model_name.lower() else 'qwen'
    output_path = Path(f'./results/EfficencyOverview/{model_chao}/')
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    for run_idx in range(6):
        torch.cuda.empty_cache()

        decode_latency, past_key_values, current_input_ids = [], None, inputs['input_ids']

        with torch.no_grad():
            for i in range(fixed_output_length):
                torch.cuda.synchronize()
                start_ts = time.perf_counter()
                outputs = model(input_ids=current_input_ids, past_key_values=past_key_values, use_cache=True, num_logits_to_keep=1)
                past_key_values = outputs.past_key_values
                next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
                torch.cuda.synchronize()
                decode_latency.append(time.perf_counter() - start_ts)
                current_input_ids = next_token_id

        ttft = decode_latency[0]
        tpot = np.mean(decode_latency[1:]) if len(decode_latency) > 1 else 0
        latency = sum(decode_latency)

        save_data = {
            "run": run_idx + 1,
            "ttft": ttft,
            "tpot": tpot * 1000,
            "latency": latency,
            "decode_latency": decode_latency
        }
        with open(output_path / f"Efficency_{input_max_token}_{fixed_output_length}.jsonl", "a", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False)
            f.write("\n")

        del past_key_values
        torch.cuda.empty_cache()


def main():

    args = parse_args()

    data = load_dataset(args)

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name)
    print(f"Model loaded. Device: {model.device}")

    get_pred(model, tokenizer, data, args)


if __name__ == "__main__":
    main()
