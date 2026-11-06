"""Batched Adakv (HeadKV/AdaKV) prediction for LongBench and RULER.

Each ``model.generate`` call receives at most ``batch_size`` distinct samples.
Prompts are tokenized with left padding and truncated to the model input
budget; generated tokens are sliced at the shared padded width. Completed
samples are skipped on re-runs, and every batch appends one line to
``<task>.jsonl`` and ``<task>.batch_stats.jsonl``.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
# METHODS_ROOT (methods/Adakv) contains the ``headkv`` package used by the monkey patch.
METHOD_ROOT = EXPERIMENT_ROOT.parents[1]
# REPO_ROOT is Sparse-KVCache-Experiment/ (holds benchmarks/ruler).
REPO_ROOT = EXPERIMENT_ROOT.parents[3]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from utils import load_data


def read_config(name):
    with (EXPERIMENT_ROOT / "config" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


dataset2prompt = read_config("dataset2prompt.json")
dataset2maxlen = read_config("dataset2maxlen.json")
model2path = read_config("model2path.json")
model2maxlen = read_config("model2maxlen.json")

# LongBench tasks that generate the answer directly; the rest go through the chat template.
GENERATION_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
# RULER template name -> tokens_to_generate, mirroring RULER/run_ruler.py TASKS.
SYNTHETIC_TASKS = {
    "niah": 128,
    "variable_tracking": 30,
    "vt": 30,
    "common_words_extraction": 120,
    "cwe": 120,
    "freq_words_extraction": 50,
    "fwe": 50,
    "qa": 32,
}
RULER_DATA_ROOT = REPO_ROOT / "benchmarks" / "ruler" / "benchmark_root"
RULER_TASK_YAML = REPO_ROOT / "benchmarks" / "ruler" / "synthetic.yaml"


def benchmark_name(value):
    if value.lower() == "longbench":
        return "LongBench"
    if value.lower() in {"ruler", "synthetic"}:
        return "RULER"
    raise argparse.ArgumentTypeError("benchmark must be longbench, ruler or synthetic")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Model alias in config/model2path.json, HF model ID, or local path")
    parser.add_argument("--benchmark", required=True, type=benchmark_name)
    parser.add_argument("--task", required=True)
    parser.add_argument("--save_dir", required=True, type=Path)
    parser.add_argument("--batch_size", type=positive_int, default=1,
                        help="Number of distinct samples per generate call (default: 1)")
    parser.add_argument("--context_length", type=positive_int, default=None,
                        help="RULER dataset context length (default: 65536)")
    parser.add_argument("--data_dir", type=Path, default=None,
                        help="Local task JSONL file or directory; otherwise use benchmark defaults")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Number of examples to evaluate; -1 for all")
    parser.add_argument("--method", default="AdativeKV", help="HeadKV method (fullkv disables the patch)")
    parser.add_argument("--head_choice", default="random", choices=["random", "copy", "reason"])
    parser.add_argument("--max_capacity_prompts", type=int, default=512)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--pooling", default="maxpool")
    parser.add_argument("--floor", type=float, default=0.2)
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--device", default="auto", help="Device map, e.g. auto or cuda:0")
    args = parser.parse_args(argv)
    if args.num_samples != -1 and args.num_samples <= 0:
        parser.error("--num_samples must be -1 or a positive integer")
    if args.benchmark == "RULER":
        args.context_length = args.context_length or 65536
    else:
        if args.context_length is not None:
            parser.error("--context_length is only applicable to RULER/Synthetic")
        if args.task not in dataset2prompt:
            parser.error(f"Unknown LongBench task: {args.task}")
    args.model = model2path.get(args.model, args.model)
    return args


def ruler_model_short(model_name):
    """Map the model to the prepared RULER data directory."""
    name = Path(model_name).name.lower()
    if "llama" in name:
        return "llama-3.1-8b"
    if "qwen3" in name:
        return "qwen3-32b"
    if "qwen" in name:
        return "qwen-2.5-7b-1m"
    raise ValueError(f"No RULER data mapping for {model_name}; supply --data_dir explicitly")


def ruler_task_tokens(task):
    """Resolve tokens_to_generate via benchmarks/ruler/synthetic.yaml + run_ruler TASKS."""
    base = None
    if RULER_TASK_YAML.is_file():
        import yaml

        customized = yaml.safe_load(RULER_TASK_YAML.read_text(encoding="utf-8"))
        base = customized.get(task, {}).get("task")
    if base is None:
        base = next((key for key in SYNTHETIC_TASKS if key in task), None)
    return SYNTHETIC_TASKS.get(base, 128)


def model_max_len(model_name):
    """Substring match against config/model2maxlen.json, longest key first."""
    lowered = model_name.lower()
    for key in sorted(model2maxlen, key=len, reverse=True):
        if key in lowered:
            return model2maxlen[key]
    return None


def sample_index(sample, benchmark, position):
    index = sample.get("_id") if benchmark == "LongBench" else sample.get("index")
    return position if index is None or index == "" else index


def load_dataset(args, tokenizer=None):
    """Prepare prompts and skip completed samples before forming batches."""
    args.save_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.save_dir / f"{args.task}.jsonl"
    input_budget = None
    if args.data_dir is not None:
        task_file = args.data_dir
        if task_file.is_dir():
            task_file /= f"{args.task}.jsonl" if args.benchmark == "LongBench" else "validation.jsonl"
        data = load_data(task_file)
    elif args.benchmark == "LongBench":
        from datasets import DownloadConfig, load_dataset as hf_load_dataset

        retries = int(os.environ.get("LONG_BENCH_DOWNLOAD_RETRIES", "5"))
        if retries < 1:
            raise ValueError("LONG_BENCH_DOWNLOAD_RETRIES must be at least 1")
        data = [dict(sample) for sample in hf_load_dataset(
            "THUDM/LongBench", args.task, split="test", trust_remote_code=True,
            download_config=DownloadConfig(max_retries=retries),
        )]
    else:
        task_file = (RULER_DATA_ROOT / ruler_model_short(args.model) / "synthetic"
                     / str(args.context_length) / "data" / args.task / "validation.jsonl")
        if not task_file.is_file():
            raise FileNotFoundError(f"RULER task file not found: {task_file}")
        data = load_data(task_file)

    if not data:
        raise ValueError(f"No samples available for {args.task}")
    if args.num_samples > 0:
        data = data[:args.num_samples]
    completed = {str(record["index"]) for record in load_data(out_path)} if out_path.exists() else set()
    remaining = []
    seen = set()
    for position, original in enumerate(data):
        index = sample_index(original, args.benchmark, position)
        if str(index) in seen:
            raise ValueError(f"Duplicate sample index: {index}")
        seen.add(str(index))
        if str(index) not in completed:
            sample = dict(original)
            sample["index"] = index
            remaining.append(sample)

    if args.benchmark == "LongBench":
        max_new_tokens = dataset2maxlen[args.task]
        if remaining and args.task not in GENERATION_TASKS and tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        for sample in remaining:
            prompt = dataset2prompt[args.task].format(**sample)
            if args.task not in GENERATION_TASKS:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False,
                )
            sample["input"] = prompt
        length = model_max_len(args.model)
        if length is not None:
            input_budget = max(1, int(length) - int(max_new_tokens))
    else:
        max_new_tokens = ruler_task_tokens(args.task)

    print(f"[{args.task}] {len(remaining)} remaining, {len(completed)} saved, "
          f"batch_size={args.batch_size}, max_new_tokens={max_new_tokens}, input_budget={input_budget}")
    return remaining, max_new_tokens, input_budget, out_path


def seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "encoder") \
            and hasattr(model.transformer.encoder, "layers"):
        return model.transformer.encoder.layers
    raise ValueError("Unsupported model architecture for layer traversal")


def get_self_attention_module(layer):
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if hasattr(layer, "self_attention"):
        return layer.self_attention
    raise ValueError("Unsupported layer attention module")


def get_config_holder(model):
    if hasattr(model, "model") and hasattr(model.model, "config"):
        return model.model.config
    if hasattr(model, "transformer") and hasattr(model.transformer, "config"):
        return model.transformer.config
    return model.config


def apply_headkv_runtime_config(model, args):
    """Set the same runtime HeadKV config as LonBench/run_longbench.py."""
    fields = {
        "window_size": args.window_size,
        "base_capacity": args.max_capacity_prompts,
        "head_choice": args.head_choice,
        "beta": args.beta,
        "temp": args.temp,
        "kernel_size": args.kernel_size,
        "skip": 0,
        "normalize": True,
        "pooling": args.pooling,
        "floor": args.floor,
    }
    config_holder = get_config_holder(model)
    for name, value in fields.items():
        setattr(config_holder, name, value)
    for layer in get_transformer_layers(model):
        attn_module = get_self_attention_module(layer)
        if not hasattr(attn_module, "config"):
            attn_module.config = model.config
        for name, value in fields.items():
            setattr(attn_module.config, name, value)


def load_model(args):
    """Load the Transformers model and apply the HeadKV patch exactly as run_longbench.py."""
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_path = args.model
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise ValueError("Batched generation requires a tokenizer pad_token or eos_token")

    method = (args.method or "fullkv").lower()
    attn_impl = "flash_attention_2"
    is_glm = "glm" in model_path.lower() or "chatglm" in model_path.lower()
    if method != "fullkv":
        from headkv.monkeypatch import replace_llama, replace_qwen2, replace_chatglm
        replace_llama(args.method)
        replace_qwen2(args.method)
    if is_glm:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.use_cache = True
        config._attn_implementation = attn_impl
        chatglm_cls = get_class_from_dynamic_module(
            "modeling_chatglm.ChatGLMForConditionalGeneration", model_path, local_files_only=True,
        )
        model = chatglm_cls.from_pretrained(
            model_path, config=config, torch_dtype=dtype, low_cpu_mem_usage=True,
            device_map=args.device, local_files_only=True,
        )
        if method != "fullkv":
            replace_chatglm(args.method, model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=args.device,
            use_cache=True, attn_implementation=attn_impl,
        )
    torch.cuda.empty_cache()
    model.eval()
    apply_headkv_runtime_config(model, args)
    return model, tokenizer


def iter_batches(data, batch_size):
    for start in range(0, len(data), batch_size):
        yield data[start:start + batch_size]


def model_cuda_devices(model):
    """Synchronize every participating GPU when timing a sharded model."""
    import torch

    placements = getattr(model, "hf_device_map", {}).values()
    devices = set()
    for placement in [model.device, *placements]:
        if placement == "disk":
            continue
        device = torch.device(f"cuda:{placement}" if isinstance(placement, int) else placement)
        if device.type == "cuda":
            devices.add(device)
    return devices


def trim_generated(tokens, eos_token_id):
    """Remove the first EOS and padding while another row is still generating."""
    eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
    for position, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[:position]
    return tokens


def get_pred(model, tokenizer, samples, max_new_tokens, benchmark, out_path,
             input_budget=None, batch_size=None, extra_gen_kwargs=None):
    """Run one real batch and preserve each row's sample ID and references."""
    import torch

    tokenizer.padding_side = "left"
    inputs = tokenizer(
        [sample["input"] for sample in samples], return_tensors="pt", padding="longest",
        truncation=input_budget is not None, max_length=input_budget, add_special_tokens=True,
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    input_lengths = attention_mask.sum(dim=-1).tolist()
    if any(length == 0 for length in input_lengths):
        raise ValueError("An empty prompt cannot be generated")
    padded_length = input_ids.shape[1]
    print(f"Batch size: {len(samples)}, input lengths: {input_lengths}, padded length: {padded_length}")
    gen_kwargs = dict(
        input_ids=input_ids, attention_mask=attention_mask,
        max_new_tokens=max_new_tokens, num_beams=1, do_sample=False,
        min_length=padded_length + 1, pad_token_id=tokenizer.pad_token_id,
    )
    if extra_gen_kwargs:
        gen_kwargs.update(extra_gen_kwargs)
    devices = model_cuda_devices(model)
    for device in devices:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**gen_kwargs)
    for device in devices:
        torch.cuda.synchronize(device)
    latency = time.perf_counter() - started
    if len(out) != len(samples):
        raise RuntimeError(f"Expected {len(samples)} generated rows, got {len(out)}")
    eos_ids = getattr(model.generation_config, "eos_token_id", None)
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    # generate returns the entire padded input prefix for every row. Slice at the
    # shared width rather than at each row's unpadded prompt length.
    outputs = [trim_generated(row, eos_ids) for row in out[:, padded_length:].tolist()]
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    with Path(out_path).open("a", encoding="utf-8") as handle:
        for sample, prediction, input_length, tokens in zip(samples, predictions, input_lengths, outputs):
            record = {"pred": prediction, "index": sample["index"], "batch_size": len(samples),
                      "input_tokens": input_length, "output_tokens": len(tokens)}
            if benchmark == "LongBench":
                record.update(answers=sample["answers"], all_classes=sample.get("all_classes"),
                              length=sample.get("length", input_length))
            else:
                record.update(outputs=sample["outputs"], input=sample["input"],
                              others=sample.get("others", {}), truncation=sample.get("truncation", -1),
                              length=sample.get("length", -1))
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats = {
        "indices": [sample["index"] for sample in samples],
        "batch_size": len(samples), "configured_batch_size": batch_size or len(samples),
        "padded_input_tokens": padded_length, "input_tokens": input_lengths,
        "output_tokens": list(map(len, outputs)), "latency_s": latency,
        "output_tokens_per_s": sum(map(len, outputs)) / latency if latency else 0.0,
    }
    stats_path = Path(out_path).with_suffix(".batch_stats.jsonl")
    with stats_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(stats, ensure_ascii=False) + "\n")
    return stats


def model_gen_kwargs(model_name):
    """Extra generate kwargs used by LonBench/run_longbench.py for GLM/Qwen."""
    lowered = model_name.lower()
    if "glm" in lowered or "chatglm" in lowered:
        return {"return_last_logit": True}
    if "qwen2" in lowered or "qwen2.5" in lowered:
        return {"num_logits_to_keep": 1}
    return {}


def main(argv=None):
    args = parse_args(argv)
    data, max_new_tokens, input_budget, out_path = load_dataset(args)
    if not data:
        print(f"[{args.task}] All selected examples already predicted, skipping model loading.")
        return
    from tqdm import tqdm

    seed_everything(42)
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args)
    extra_gen_kwargs = model_gen_kwargs(args.model)
    for samples in tqdm(iter_batches(data, args.batch_size),
                        total=(len(data) + args.batch_size - 1) // args.batch_size,
                        desc=f"Predicting {args.task}"):
        get_pred(model, tokenizer, samples, max_new_tokens, args.benchmark, out_path,
                 input_budget, args.batch_size, extra_gen_kwargs)
    print(f"[{args.task}] Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
