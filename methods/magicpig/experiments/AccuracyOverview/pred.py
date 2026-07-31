# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer

_MAGICPIG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _MAGICPIG_ROOT not in sys.path:
    sys.path.insert(0, _MAGICPIG_ROOT)

from models_single.deepseek import DeepSeekModel
from models_single.glm import GLMModel
from models_single.llama import LlamaModel
from models_single.qwen import Qwen2Model
from utils import load_data


_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
dataset2prompt = json.load(open(os.path.join(_CONFIG_DIR, "dataset2prompt.json"), "r"))
dataset2maxlen = json.load(open(os.path.join(_CONFIG_DIR, "dataset2maxlen.json"), "r"))

GENERATION_TASKS = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]

SYNTHETIC_TASKS = {
    "niah": 128,
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa": 32,
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model path or HF repo id")
    parser.add_argument("--model_key", type=str, default="", help="Short model key used in configs/logs")
    parser.add_argument("--benchmark", type=str, required=True, choices=["LongBench", "Synthetic"])
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--data_dir", type=Path, default=Path("../../../../benchmarks"))
    parser.add_argument("--K", type=int, default=9)
    parser.add_argument("--L", type=int, default=150)
    parser.add_argument("--recall", type=int, default=0)
    parser.add_argument("--fixed_budget", type=int, default=0)
    parser.add_argument("--fixed_output_length", type=int, default=0)
    parser.add_argument("--measure_time", type=int, default=0)
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=65536,
        help="Maximum prompt length budget used to size MagicPig KV cache.",
    )
    return parser.parse_args(args)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    if not torch.cuda.is_available():
        raise RuntimeError("MagicPig AccuracyOverview requires CUDA.")

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank()))
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def is_glm_model(model_name):
    return "glm" in model_name.lower()


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=is_glm_model(model_name),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_dataset(args):
    """Load data and skip samples already present in the output jsonl."""
    model_name = args.model_name
    task = args.task
    benchmark = args.benchmark

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"{task}.jsonl")

    if benchmark == "LongBench":
        from datasets import load_dataset as hf_load_dataset

        data = [dict(sample) for sample in hf_load_dataset("THUDM/LongBench", task, split="test")]
        max_new_tokens = dataset2maxlen.get(task, 128)

        tokenizer = load_tokenizer(model_name)
        prompt_format = dataset2prompt.get(task, "{input}")
        for i, json_obj in enumerate(data):
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
        task_file = args.data_dir / "validation.jsonl"
        if not task_file.exists():
            raise FileNotFoundError(f"Synthetic task file not found: {task_file}")

        data = load_data(task_file)
        task_key = next((key for key in SYNTHETIC_TASKS if key in task), None)
        max_new_tokens = SYNTHETIC_TASKS.get(task_key, 128)

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    if args.num_samples > 0:
        data = data[: args.num_samples]

    pred_indices = set()
    if os.path.exists(out_path):
        for sample in load_data(out_path):
            pred_indices.add(sample.get("index", ""))

    remaining = []
    for sample in data:
        idx = sample.get("_id", "") if benchmark == "LongBench" else sample.get("index", "")
        if idx not in pred_indices:
            remaining.append(sample)

    print(
        f"[{task}] {len(remaining)} samples to predict, "
        f"{len(pred_indices)} already done, max_new_tokens: {max_new_tokens}"
    )
    return remaining, max_new_tokens, out_path


def _supported_kwargs(model_cls, kwargs):
    signature = inspect.signature(model_cls.__init__)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def load_model(model_name, K, L, batch_size, max_length, max_new_tokens, device, dtype, args):
    model_name_lower = model_name.lower()
    recall = args.recall == 1
    measure_time = args.measure_time == 1

    common_kwargs = {
        "model_name": model_name,
        "K": K,
        "L": L,
        "batch_size": batch_size,
        "max_length": max_length,
        "num_sink_tokens": 4,
        "num_local_tokens": 4,
        "generation_buffer": max_new_tokens,
        "device": device,
        "dtype": dtype,
        "RECALL": recall,
        "fixed_output_length": args.fixed_output_length,
    }

    if "llama" in model_name_lower:
        model_cls = LlamaModel
    elif "qwen" in model_name_lower:
        if "deepseek" in model_name_lower:
            model_cls = DeepSeekModel
        else:
            model_cls = Qwen2Model
    elif "glm" in model_name_lower:
        model_cls = GLMModel
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model_cls(**_supported_kwargs(model_cls, common_kwargs))


def get_pred(llm, tokenizer, data_sample, max_new_tokens, benchmark, out_path, device):
    prompt = data_sample["input"]
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    input_len = input_ids.shape[1]
    print(f"Input token length: {input_len}")

    output = llm.generate(input_ids=input_ids, max_tokens=max_new_tokens)
    pred = tokenizer.decode(output[input_len:], skip_special_tokens=True)
    print(f"Output token length: {len(output) - input_len}")

    index = data_sample.get("_id", "") if benchmark == "LongBench" else data_sample.get("index", "")
    answers = data_sample.get("answers", data_sample.get("outputs", []))

    if dist.get_rank() == 0:
        with open(out_path, "a", encoding="utf-8") as f:
            if benchmark == "LongBench":
                json.dump(
                    {
                        "pred": pred,
                        "answers": data_sample["answers"],
                        "all_classes": data_sample.get("all_classes", None),
                        "length": data_sample.get("length", len(pred)),
                        "index": index,
                    },
                    f,
                    ensure_ascii=False,
                )
            else:
                json.dump(
                    {
                        "pred": pred,
                        "outputs": data_sample["outputs"],
                        "input": data_sample["input"],
                        "others": data_sample.get("others", {}),
                        "truncation": data_sample.get("truncation", -1),
                        "length": data_sample.get("length", -1),
                        "index": index,
                    },
                    f,
                    ensure_ascii=False,
                )
            f.write("\n")

    attention_server = getattr(llm, "attention_server", None)
    if attention_server is not None:
        if hasattr(attention_server, "avg_nnz"):
            attention_server.avg_nnz = 0
        if hasattr(attention_server, "count_nnz"):
            attention_server.count_nnz = 0
    torch.cuda.empty_cache()


def main(args):
    seed_everything(42)
    device = setup_distributed()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    data, max_new_tokens, out_path = load_dataset(args)
    if len(data) == 0:
        print(f"[{args.task}] All examples already predicted, skipping.")
        return

    max_length = args.max_seq_length + max_new_tokens + 256
    if args.fixed_output_length > 0:
        max_length = args.max_seq_length + args.fixed_output_length + 256

    print(f"Loading MagicPig model: {args.model_name}")
    llm = load_model(
        args.model_name,
        args.K,
        args.L,
        1,
        max_length,
        max_new_tokens,
        device,
        dtype,
        args,
    )
    tokenizer = load_tokenizer(args.model_name)
    print(f"Model loaded. Device: {device}")

    for data_sample in tqdm(data, desc=f"Predicting {args.task}"):
        get_pred(llm, tokenizer, data_sample, max_new_tokens, args.benchmark, out_path, device)

    print(f"[{args.task}] Predictions saved to {out_path}")


if __name__ == "__main__":
    main(parse_args())
