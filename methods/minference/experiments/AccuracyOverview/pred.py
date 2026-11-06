# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import os
import sys
import json
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path for minference import
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from minference import MInference
from utils import load_data

# Load config files
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
dataset2prompt = json.load(open(os.path.join(_CONFIG_DIR, "dataset2prompt.json"), "r"))
dataset2maxlen = json.load(open(os.path.join(_CONFIG_DIR, "dataset2maxlen.json"), "r"))

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

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model path (local path)")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["LongBench", "Synthetic"],
                        help="Benchmark type: LongBench or Synthetic")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (e.g. narrativeqa, niah_single_1, vt, etc.)")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save prediction results")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"],
                        help="Model dtype")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (e.g. auto, cuda:0)")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Number of examples to evaluate. -1 for all.")
    parser.add_argument("--data_dir", type=str, default="../../../../benchmarks",
                        help="Directory containing the task data files (constructed by Accuracy.sh)")
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
        from datasets import DownloadConfig, load_dataset as hf_load_dataset

        # The LongBench script downloads a single large archive.  Default
        # datasets retries only once, which is too brittle for a transient HF
        # mirror/network timeout.  Prepared datasets still load from the cache
        # without a network request.
        download_retries = int(os.environ.get("LONG_BENCH_DOWNLOAD_RETRIES", "5"))
        if download_retries < 1:
            raise ValueError("LONG_BENCH_DOWNLOAD_RETRIES must be at least 1")
        download_config = DownloadConfig(max_retries=download_retries)

        # LongBench is distributed with a dataset loading script.  Accuracy.sh
        # runs non-interactively, so explicitly approve that script here rather
        # than letting datasets prompt and then fail with its default answer.
        data = [
            dict(sample)
            for sample in hf_load_dataset(
                "THUDM/LongBench",
                task,
                split="test",
                trust_remote_code=True,
                download_config=download_config,
            )
        ]
        max_new_tokens = dataset2maxlen.get(task, 128)

        # Apply chat template for non-generation tasks
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Qwen3 enables thinking by default.  LongBench evaluates the final
        # answer and does not need hidden chain-of-thought tokens, so use the
        # model's hard template switch to prevent a <think> block entirely.
        # Keep all other model families on their existing chat template path.
        is_qwen3 = "qwen3" in model_name.lower() or "qwen3" in str(
            getattr(tokenizer, "name_or_path", "")
        ).lower()
        chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
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
                    **chat_template_kwargs,
                )
            data[i]["input"] = prompt

    elif benchmark == "Synthetic":
        task_file = f'{args.data_dir}/validation.jsonl'
        if not os.path.exists(task_file):
            raise FileNotFoundError(f"Synthetic task file not found: {task_file}")

        data = load_data(task_file)

        # Determine max_new_tokens from task type
        task_key = next((key for key in SYNTHETIC_TASKS.keys() if key in task), None)
        max_new_tokens = SYNTHETIC_TASKS.get(task_key, 128)

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    # A pattern search is meaningful only for a long prefill.  Select one
    # longest available sample instead of relying on the dataset's first row;
    # regular evaluation retains its normal --num_samples behavior.
    is_search = os.environ.get("is_search", "false").lower() == "true"
    if is_search:
        if not data:
            raise ValueError(f"No samples available for {task} pattern search")
        max_search_length = int(os.environ.get("MINFERENCE_SEARCH_MAX_LENGTH", "0"))
        if max_search_length < 0:
            raise ValueError("MINFERENCE_SEARCH_MAX_LENGTH must be non-negative")
        candidates = data
        if max_search_length:
            candidates = [
                sample for sample in data
                if sample.get("length", 0) <= max_search_length
            ]
            if not candidates:
                raise ValueError(
                    f"No {task} sample has length <= {max_search_length} for pattern search"
                )
        data = [max(candidates, key=lambda sample: sample.get("length", 0))]
        limit_note = f", limit={max_search_length}" if max_search_length else ""
        print(
            f"[{task}] Search mode selected longest eligible sample "
            f"(length={data[0].get('length', 'unknown')}{limit_note})"
        )
    elif args.num_samples > 0:
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


def load_model(model_name, dtype, device):
    """Load model with MInference sparse attention patch."""
    # Monkey-patch: GLM models define ChatGLMConfig in both modeling_chatglm.py
    # and configuration_chatglm.py, causing config_class mismatch in register().
    # Unify the model's config_class with the passed config class before the check.
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

    _orig_register = _BaseAutoModelClass.register.__func__

    def _patched_register(cls, config_class, model_class, exist_ok=False):
        if (
            hasattr(model_class, "config_class")
            and str(model_class.config_class) != str(config_class)
        ):
            model_class.config_class = config_class
        return _orig_register(cls, config_class, model_class, exist_ok=exist_ok)

    _BaseAutoModelClass.register = classmethod(_patched_register)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    finally:
        _BaseAutoModelClass.register = classmethod(_orig_register)

    model.eval()

    # Apply MInference patch
    minference_patch = MInference(model_name)
    model = minference_patch(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_pred(model, tokenizer, data_sample, max_new_tokens, benchmark, out_path):
    """Run prediction on a single data sample and save to jsonl."""
    prompt = data_sample["input"]

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    input_len = input_ids.shape[1]
    print(f"Input token length: {input_len}")

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=1.0,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens
    generated_ids = out[0, input_len:]
    pred = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"Output token length: {len(generated_ids)}")

    # Determine index key based on benchmark
    if benchmark == "LongBench":
        index = data_sample.get("_id", "")
    else:
        index = data_sample.get("index", "")

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

    torch.cuda.empty_cache()


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    model_name = args.model_name
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    # Load data (with resume support)
    data, max_new_tokens, out_path = load_dataset(args)

    if len(data) == 0:
        print(f"[{args.task}] All examples already predicted, skipping.")
        sys.exit(0)

    # Load model
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, dtype, args.device)
    print(f"Model loaded. Device: {model.device}")

    # Predict
    for data_sample in tqdm(data, desc=f"Predicting {args.task}"):
        get_pred(model, tokenizer, data_sample, max_new_tokens, args.benchmark, out_path)

    print(f"[{args.task}] Predictions saved to {out_path}")
