"""Shared batched inference for AdaKV (HeadKV) efficiency and VRAM experiments.

This adapts the RetroInfer / MInference ``BatchEfficency_VRAM`` design to AdaKV's
Transformers-based model loader (``run_longbench.py``) and HeadKV monkey patches.
Efficiency metrics are read from ``model.model._timing`` (written by
``adaptive_LlamaModel_forward``); for architectures whose hijack does not populate
``_timing`` (Qwen2 / ChatGLM) a lightweight forward timer reproduces the same
fields. VRAM metrics sample the CUDA allocator before cache release.
"""

import argparse
from contextlib import nullcontext
import csv
import hashlib
import inspect
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
METHOD_ROOT = EXPERIMENT_ROOT.parents[1]  # .../methods/Adakv  (contains headkv/)
REPO_ROOT = EXPERIMENT_ROOT.parents[3]    # .../Sparse-KVCache-Experiment
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from utils import load_data
from decode_memory import DecodeMemoryMonitor


def read_config(name):
    with (EXPERIMENT_ROOT / "config" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


dataset2prompt = read_config("dataset2prompt.json")
dataset2maxlen = read_config("dataset2maxlen.json")
model2path = read_config("model2path.json")
model2maxlen = read_config("model2maxlen.json")

GENERATION_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
SYNTHETIC_TASKS = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}
RULER_DATA_ROOT = REPO_ROOT / "benchmarks" / "ruler" / "benchmark_root"
SCHEMA_VERSION = 1
DECODE_MEMORY_FIELDS = {
    "decode_num_steps", "decode_avg_alloc", "decode_peak_alloc",
    "decode_avg_alloc_by_device", "decode_peak_alloc_by_device",
}
SEED = 42


def ruler_model_short(model_path):
    """Map a model path to the prepared RULER data directory name."""
    name = Path(model_path).name.lower()
    if "llama" in name:
        return "llama-3.1-8b"
    if "qwen3" in name:
        return "qwen3-32b"
    if "qwen" in name:
        return "qwen-2.5-7b-1m"
    if "glm" in name:
        return "glm-4-9b"
    raise ValueError(f"No RULER data mapping for {model_path}; supply --data_dir explicitly.")


def model_default_max_len(model_path):
    """Reuse run_longbench's substring model2maxlen lookup."""
    path = str(model_path).lower()
    for key in sorted(model2maxlen, key=len, reverse=True):
        if key in path:
            return model2maxlen[key]
    return 130000


def benchmark_name(value):
    if value.lower() == "longbench":
        return "LongBench"
    if value.lower() in {"ruler", "synthetic"}:
        return "RULER"
    raise argparse.ArgumentTypeError("benchmark must be LongBench, RULER or Synthetic")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args(argv=None, measurement="efficency"):
    if measurement not in {"efficency", "vram"}:
        raise ValueError(f"Unknown measurement: {measurement}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Alias in config/model2path.json, HF model ID, or local model path")
    parser.add_argument("--benchmark", required=True, type=benchmark_name)
    parser.add_argument("--task", required=True)
    parser.add_argument("--batch_size", type=positive_int, default=1,
                        help="Number of distinct requests in each generation call; keep the final partial batch")
    parser.add_argument("--context_length", type=positive_int, default=None,
                        help="RULER dataset length, e.g. 4096/8192/65536")
    parser.add_argument("--max_len", type=positive_int, default=None,
                        help="Prompt token limit; defaults to the model config for LongBench, "
                             "context_length for RULER")
    parser.add_argument("--data_dir", type=Path, default=None,
                        help="Optional local JSONL file or task directory; LongBench defaults to THUDM/LongBench")
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--device", default="auto", help="device_map, e.g. auto or cuda:0")
    parser.add_argument("--num_samples", type=int, default=-1, help="-1 for all; otherwise a positive sample count")
    parser.add_argument("--fixed_output_length", type=int, default=0,
                        help="0: task default and per-request EOS; positive: exactly this many output tokens")
    parser.add_argument("--warmup", type=int, default=7 if measurement == "efficency" else 2,
                        help="Warmup calls on the first batch; excluded from predictions and measured results")
    # HeadKV (AdaKV) parameters.
    parser.add_argument("--method", default="AdativeKV")
    parser.add_argument("--head_choice", default="random", choices=["random", "copy", "reason"])
    parser.add_argument("--max_capacity_prompts", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--pooling", default="maxpool")
    parser.add_argument("--floor", type=float, default=0.2)
    args = parser.parse_args(argv)
    args.measurement = measurement
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.num_samples != -1 and args.num_samples <= 0:
        parser.error("--num_samples must be -1 or a positive integer")
    if args.fixed_output_length < 0:
        parser.error("--fixed_output_length must be non-negative")
    args.model = model2path.get(args.model, args.model)
    if args.benchmark == "RULER":
        args.context_length = args.context_length or 65536
        args.max_len = args.max_len or args.context_length
        if args.max_len < args.context_length:
            parser.error("--max_len must be at least --context_length for RULER")
    else:
        if args.context_length is not None:
            parser.error("--context_length is only applicable to RULER/Synthetic")
        if args.task not in dataset2prompt:
            parser.error(f"Unknown LongBench task: {args.task}")
        args.max_len = args.max_len or model_default_max_len(args.model)
    return args


def sample_index(sample, benchmark, position):
    key = "_id" if benchmark == "LongBench" else "index"
    value = sample.get(key)
    return position if value is None or value == "" else value


def decode_throughput(batch_size, decode_steps, decode_latency):
    """Decode token slots per second, using the model log's batch-size rule."""
    return batch_size * decode_steps / decode_latency if decode_steps > 0 and decode_latency > 0 else 0.0


def efficiency_derived(record):
    """Add derived three-stage throughput fields to an efficiency record without retiming."""
    output_tokens = record.get("output_tokens")
    input_tokens = record.get("input_tokens")
    derived = {
        "prefill_tokens_per_s": (sum(input_tokens) / record["ttft"]
                                 if input_tokens and record.get("ttft") else None),
        "decode_tokens_per_s": (sum(c - 1 for c in output_tokens) / record["decode_latency"]
                                if output_tokens and record.get("decode_steps")
                                and record.get("decode_latency") else None),
        "output_tokens_per_s": (sum(output_tokens) / record["latency"]
                                if output_tokens and record.get("latency") else None),
    }
    if record.get("decode_steps") and record.get("decode_latency"):
        derived["throughput"] = decode_throughput(record.get("batch_size"), record["decode_steps"],
                                                  record["decode_latency"])
    return {key: value for key, value in derived.items() if value is not None}


def load_efficiency_records(path):
    """Add throughput and three-stage rates to old efficiency records from existing counters."""
    path = Path(path)
    records = load_data(path) if path.exists() else []
    changed = False
    for record in records:
        derived = efficiency_derived(record)
        for key, value in derived.items():
            if key not in record:
                record[key] = value
                changed = True
    if changed:
        write_jsonl_atomic(path, records)
    return records


def run_manifest_config(args, data_path, max_new_tokens, input_sha256):
    """Configuration recorded in the run manifest for reproducibility checks."""
    return {
        "schema_version": SCHEMA_VERSION, "measurement": args.measurement,
        "model": args.model, "benchmark": args.benchmark, "task": args.task,
        "batch_size": args.batch_size, "context_length": args.context_length,
        "max_len": args.max_len, "num_samples": args.num_samples, "dtype": args.dtype,
        "device": args.device, "fixed_output_length": args.fixed_output_length,
        "warmup": args.warmup, "method": args.method, "head_choice": args.head_choice,
        "max_capacity_prompts": args.max_capacity_prompts, "beta": args.beta,
        "temp": args.temp, "window_size": args.window_size, "kernel_size": args.kernel_size,
        "pooling": args.pooling, "floor": args.floor,
        "data_dir": str(args.data_dir) if args.data_dir is not None else None,
        "data_file": str(data_path.resolve()) if data_path is not None else None,
        "max_new_tokens": max_new_tokens, "input_sha256": input_sha256,
        "loader_fingerprint": LOADER_FINGERPRINT, "seed": SEED,
    }


def build_input_sha256(data):
    """Fingerprint the full planned dataset (not just remaining) for stable resume."""
    digest = hashlib.sha256()
    for sample in data:
        digest.update(json.dumps({"index": sample["index"], "input": sample.get("input", "")},
                                 ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def prepare_samples(data, benchmark):
    """Assign stable indices and reject duplicates across the full dataset."""
    seen = set()
    for position, sample in enumerate(data):
        index = sample_index(sample, benchmark, position)
        if str(index) in seen:
            raise ValueError(f"Duplicate sample index: {index}")
        seen.add(str(index))
        sample["index"] = index
    return data


def validate_manifest(args, manifest, metrics_path, config):
    """Enforce a single reproducible run per save_dir; never auto-overwrite a manifest."""
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        changed = [key for key in config if config[key] != previous.get(key)]
        if changed:
            raise ValueError(f"Run configuration or input changed ({', '.join(changed)}); use a new --save_dir")
    elif metrics_path.exists():
        raise ValueError(f"Existing measurements have no run manifest: {metrics_path}; use a new --save_dir")
    else:
        manifest.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_records(args, records, metrics_path):
    """Reject saved batches that are partial, incomplete, duplicated, or outside the plan."""
    planned = {tuple(str(sample["index"]) for sample in batch)
               for batch in iter_batches(args._planned_samples, args.batch_size)}
    completed = set()
    for record in records:
        indices = tuple(map(str, record.get("indices", [])))
        if (str(record.get("schema_version")) != str(SCHEMA_VERSION) or indices not in planned
                or len(indices) != record.get("batch_size") or completed.intersection(indices)):
            raise ValueError(f"Invalid or duplicated measurement batch in {metrics_path}")
        completed.update(indices)
    return completed


def load_dataset(args, tokenizer=None):
    """Prepare prompts/records, resume by stable sample ID, and write the run manifest.

    The input fingerprint covers the whole planned dataset so a resumed run is
    reproducible: any change in data or run configuration requires a new save_dir.
    """
    args.save_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.save_dir / f"{args.task}.{args.measurement}.pred.jsonl"
    data_path = None
    if args.data_dir is not None:
        data_path = args.data_dir
        if data_path.is_dir():
            filename = f"{args.task}.jsonl" if args.benchmark == "LongBench" else "validation.jsonl"
            data_path = data_path / filename
        data = load_data(data_path)
    elif args.benchmark == "LongBench":
        from datasets import load_dataset as hf_load_dataset

        data = [dict(sample) for sample in hf_load_dataset(
            "THUDM/LongBench", args.task, split="test", trust_remote_code=True,
        )]
    else:
        data_path = (RULER_DATA_ROOT / ruler_model_short(args.model) / "synthetic"
                     / str(args.context_length) / "data" / args.task / "validation.jsonl")
        if not data_path.is_file():
            raise FileNotFoundError(f"RULER task file not found: {data_path}")
        data = load_data(data_path)

    if args.num_samples > 0:
        data = data[:args.num_samples]

    # Build the final prompt/input for every sample (needed for the fingerprint).
    if args.benchmark == "LongBench":
        max_new_tokens = dataset2maxlen.get(args.task, 128)
        if data and args.task not in GENERATION_TASKS and tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        for sample in data:
            prompt = dataset2prompt[args.task].format(**sample)
            if args.task not in GENERATION_TASKS:
                template_kwargs = {"tokenize": False, "add_generation_prompt": True}
                if "qwen3" in str(args.model).lower():
                    template_kwargs["enable_thinking"] = False
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **template_kwargs)
            sample["input"] = prompt
    else:
        task_key = next((key for key in SYNTHETIC_TASKS if key in args.task), None)
        max_new_tokens = SYNTHETIC_TASKS.get(task_key, 128)

    prepare_samples(data, args.benchmark)
    args._planned_samples = list(data)

    metrics_path = args.save_dir / f"{args.task}.{args.measurement}.jsonl"
    manifest = args.save_dir / f"{args.task}.{args.measurement}.run.json"
    config = run_manifest_config(args, data_path, max_new_tokens, build_input_sha256(data))
    validate_manifest(args, manifest, metrics_path, config)

    # Measurements are the source of truth. A completed batch must not be
    # measured twice if prediction writing was interrupted.
    records = (load_efficiency_records(metrics_path) if args.measurement == "efficency"
               else load_data(metrics_path) if metrics_path.exists() else [])
    if args.measurement == "vram":
        # Legacy VRAM records need another generation to measure decode memory.
        records = [record for record in records if DECODE_MEMORY_FIELDS.issubset(record)]
    completed = validate_records(args, records, metrics_path)

    remaining = [sample for sample in data if str(sample["index"]) not in completed]
    print(f"[{args.task}] {len(remaining)} remaining, {len(completed)} saved, batch_size={args.batch_size}")
    return remaining, max_new_tokens, out_path


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if (hasattr(model, "transformer") and hasattr(model.transformer, "encoder")
            and hasattr(model.transformer.encoder, "layers")):
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


def get_base_module(model):
    """Return the patched language-model module whose ``forward`` carries ``_timing``."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if (hasattr(model, "transformer") and hasattr(model.transformer, "encoder")
            and hasattr(model.transformer.encoder, "layers")):
        return model.transformer.encoder
    raise ValueError("Unsupported model architecture for forward monitoring")


def apply_headkv_patch(args, model):
    if args.method.lower() == "fullkv":
        return
    from headkv.monkeypatch import replace_chatglm, replace_llama, replace_qwen2

    replace_llama(args.method)
    replace_qwen2(args.method)
    if "glm" in args.model.lower() or "chatglm" in args.model.lower():
        replace_chatglm(args.method, model)


def apply_headkv_runtime_config(model, args):
    layers_list = get_transformer_layers(model)
    config_holder = get_config_holder(model)
    config_holder.window_size = args.window_size
    config_holder.base_capacity = args.max_capacity_prompts
    config_holder.head_choice = args.head_choice
    config_holder.beta = args.beta
    config_holder.temp = args.temp
    config_holder.kernel_size = args.kernel_size
    config_holder.skip = args.skip
    config_holder.normalize = bool(args.normalize)
    config_holder.pooling = args.pooling
    config_holder.floor = args.floor
    for layer in layers_list:
        attn_module = get_self_attention_module(layer)
        if not hasattr(attn_module, "config"):
            attn_module.config = model.config
        attn_module.config.window_size = config_holder.window_size
        attn_module.config.base_capacity = config_holder.base_capacity
        attn_module.config.head_choice = config_holder.head_choice
        attn_module.config.beta = config_holder.beta
        attn_module.config.temp = config_holder.temp
        attn_module.config.kernel_size = config_holder.kernel_size
        attn_module.config.skip = config_holder.skip
        attn_module.config.normalize = config_holder.normalize
        attn_module.config.pooling = config_holder.pooling
        attn_module.config.floor = config_holder.floor


def ensure_pad_token(tokenizer, model):
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))


def load_model(args, dtype):
    """Load the same Transformers model and HeadKV patch as run_longbench.py."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_path = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")

    apply_headkv_patch(args, model=None)
    if "glm" in model_path.lower() or "chatglm" in model_path.lower():
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.use_cache = True
        chatglm_cls = get_class_from_dynamic_module(
            "modeling_chatglm.ChatGLMForConditionalGeneration", model_path, local_files_only=True,
        )
        model = chatglm_cls.from_pretrained(
            model_path, config=config, torch_dtype=dtype, low_cpu_mem_usage=True,
            device_map=args.device, trust_remote_code=True, local_files_only=True,
        )
        apply_headkv_patch(args, model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=args.device,
            use_cache=True, attn_implementation="flash_attention_2", trust_remote_code=True,
        )
    model.eval()
    ensure_pad_token(tokenizer, model)
    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True
    apply_headkv_runtime_config(model, args)
    return model, tokenizer


LOADER_FINGERPRINT = hashlib.sha256(inspect.getsource(load_model).encode("utf-8")).hexdigest()


def model_cuda_devices(model):
    """Discover every participating GPU, including parameters outside device_map."""
    import torch

    placements = [model.device, *getattr(model, "hf_device_map", {}).values()]
    for accessor in ("parameters", "buffers"):
        if callable(getattr(model, accessor, None)):
            placements.extend(tensor.device for tensor in getattr(model, accessor)())
    devices = set()
    for placement in placements:
        if placement is None or placement == "disk":
            continue
        device = torch.device(f"cuda:{placement}" if isinstance(placement, int) else placement)
        if device.type == "cuda":
            devices.add(torch.device("cuda", device.index if device.index is not None
                                     else torch.cuda.current_device()))
    return sorted(devices, key=lambda device: device.index)


def synchronize_devices(devices):
    import torch

    for device in devices:
        torch.cuda.synchronize(device)


class EfficiencyTimer:
    """Time the language-model forward when ``_timing`` is not populated.

    The first forward call of a generation is the prefill (TTFT), every later call
    is a decoding step. This mirrors ``adaptive_LlamaModel_forward``'s counters for
    architectures (Qwen2 / ChatGLM) whose hijack does not write ``_timing``.
    """

    def __init__(self, module):
        self.module = module
        self.ttft = None
        self.decode_time = 0.0
        self.decode_steps = 0
        self._call_index = 0

    def __enter__(self):
        self._original = self.module.forward
        self._had_override = "forward" in vars(self.module)
        self._request_start = time.perf_counter()
        self.module.forward = self._forward
        return self

    def _forward(self, *args, **kwargs):
        started = time.perf_counter()
        result = self._original(*args, **kwargs)
        elapsed = time.perf_counter() - started
        if self._call_index == 0:
            self.ttft = time.perf_counter() - self._request_start
        else:
            self.decode_time += elapsed
            self.decode_steps += 1
        self._call_index += 1
        return result

    def __exit__(self, exc_type, exc, traceback):
        if self._had_override:
            self.module.forward = self._original
        else:
            del self.module.forward
        return False

    def metrics(self):
        tpot = self.decode_time / self.decode_steps if self.decode_steps else None
        return {
            "ttft": self.ttft,
            "tpot": tpot,
            "decode_steps": self.decode_steps,
            "decode_time": self.decode_time,
            "latency": (self.ttft or 0.0) + self.decode_time,
        }


def iter_batches(data, batch_size):
    for start in range(0, len(data), batch_size):
        yield data[start:start + batch_size]


def tokenize_batch(tokenizer, samples, max_len):
    """Keep the beginning and end of long prompts, then left-pad the batch."""
    tokenized = tokenizer([sample["input"] for sample in samples], truncation=False)
    input_ids = []
    for tokens in tokenized["input_ids"]:
        if len(tokens) > max_len:
            head = max_len // 2
            tokens = tokens[:head] + tokens[-(max_len - head):]
        if not tokens:
            raise ValueError("An empty prompt cannot be generated")
        input_ids.append(tokens)
    return tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")


def stop_token_ids(model, tokenizer):
    values = set()
    for source in (getattr(model, "generation_config", None), tokenizer):
        value = getattr(source, "eos_token_id", None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            values.update(int(token) for token in value)
        else:
            values.add(int(value))
    return values


def trim_output(tokens, stop_ids, fixed_output_length):
    if fixed_output_length:
        return tokens
    for position, token in enumerate(tokens):
        if token in stop_ids:
            return tokens[:position]
    return tokens


def generation_kwargs(model, tokenizer, inputs, gen_length, fixed_output_length):
    eos_value = getattr(model.generation_config, "eos_token_id", None)
    if eos_value is None:
        eos_value = tokenizer.eos_token_id
    if eos_value is not None and not isinstance(eos_value, (list, tuple)):
        eos_value = [eos_value]
    kwargs = {
        **{k: v for k, v in inputs.items()},
        "max_new_tokens": gen_length,
        "num_beams": 1,
        "do_sample": False,
        "eos_token_id": eos_value,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if fixed_output_length:
        kwargs["min_new_tokens"] = gen_length
    lower = str(model.name_or_path if hasattr(model, "name_or_path") else "").lower()
    if "glm" in lower or "chatglm" in lower:
        kwargs["return_last_logit"] = True
    elif "qwen2" in lower:
        kwargs["num_logits_to_keep"] = 1
    return kwargs


def reset_peak_memory_stats(devices=None):
    """Reset peak counters on the participating GPUs (all visible if not given)."""
    import torch

    if devices is None:
        devices = [torch.device(f"cuda:{idx}") for idx in range(torch.cuda.device_count())]
    for device in devices:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def collect_peak_memory_mib(devices=None):
    """Report per-GPU peak allocation/reservation and their maximum in MiB."""
    import torch

    if devices is None:
        devices = [torch.device(f"cuda:{idx}") for idx in range(torch.cuda.device_count())]
    allocated, reserved = {}, {}
    for device in devices:
        torch.cuda.synchronize(device)
        key = str(device)
        allocated[key] = torch.cuda.max_memory_allocated(device) / 1024**2
        reserved[key] = torch.cuda.max_memory_reserved(device) / 1024**2
    return {
        "peak_alloc": max(allocated.values(), default=0.0),
        "peak_reserved": max(reserved.values(), default=0.0),
        "peak_alloc_by_device": allocated,
        "peak_reserved_by_device": reserved,
    }


def release_cache(model):
    import torch

    torch.cuda.empty_cache()


def collect_efficiency_metrics(model, timer):
    timing = getattr(get_base_module(model), "_timing", None)
    if timing is not None and timing.get("ttft") is not None:
        ttft = timing["ttft"]
        decode_steps = timing["decode_steps"]
        decode_time = timing["decode_time"]
        latency = timing["latency"]
        tpot = timing["tpot"]
    else:
        values = timer.metrics()
        ttft = values["ttft"]
        decode_steps = values["decode_steps"]
        decode_time = values["decode_time"]
        latency = values["latency"]
        tpot = values["tpot"]
    return {
        "ttft": ttft,
        "TPOT": tpot * 1000 if tpot is not None and decode_steps > 0 else None,
        "decode_steps": decode_steps,
        "decode_latency": decode_time,
        "latency": latency,
    }


def run_batch(model, tokenizer, samples, max_new_tokens, args, measured=True):
    """Execute one batch; measured calls collect only the selected metric."""
    import torch

    inputs = tokenize_batch(tokenizer, samples, args.max_len)
    target_device = next(model.parameters()).device
    inputs = {key: value.to(target_device) for key, value in inputs.items()}
    input_lengths = inputs["attention_mask"].sum(dim=-1).tolist()
    print(f"Batch size: {len(samples)}, input lengths: {input_lengths}, "
          f"padded length: {inputs['input_ids'].shape[1]}")
    padded_length = inputs["input_ids"].shape[1]
    gen_length = args.fixed_output_length or max_new_tokens
    base_module = get_base_module(model)
    kwargs = generation_kwargs(model, tokenizer, inputs, gen_length, args.fixed_output_length)
    devices = model_cuda_devices(model)

    if measured and args.measurement == "vram":
        reset_peak_memory_stats(devices)
    timer = EfficiencyTimer(base_module) if measured and args.measurement == "efficency" else None
    monitor = DecodeMemoryMonitor(base_module) if measured and args.measurement == "vram" else None
    context = monitor if monitor is not None else (timer if timer is not None else nullcontext())
    try:
        with context, torch.inference_mode():
            output = model.generate(**kwargs)
        if not measured:
            metrics = {}
        elif args.measurement == "vram":
            # Collect before releasing the batch cache and GPU buffers.
            metrics = monitor.merge_batch_peaks(collect_peak_memory_mib(devices))
            metrics.update(monitor.metrics())
        else:
            synchronize_devices(devices)
            metrics = collect_efficiency_metrics(model, timer)
            synchronize_devices(devices)
    finally:
        release_cache(model)

    if output.shape[0] != len(samples):
        raise RuntimeError(f"Expected {len(samples)} generated rows, got {output.shape[0]}")
    generated = output[:, padded_length:].tolist()
    stop_ids = stop_token_ids(model, tokenizer)
    outputs = [trim_output(row, stop_ids, args.fixed_output_length) for row in generated]
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    stats = {
        "measurement": args.measurement, "model": args.model, "method": args.method,
        "indices": [sample["index"] for sample in samples],
        "batch_size": len(samples), "configured_batch_size": args.batch_size,
        "input_tokens": input_lengths, "padded_input_tokens": padded_length,
        "output_tokens": list(map(len, outputs)),
        "max_capacity_prompts": args.max_capacity_prompts, "head_choice": args.head_choice,
        "beta": args.beta, "temp": args.temp, "window_size": args.window_size,
        "kernel_size": args.kernel_size, "pooling": args.pooling, "floor": args.floor,
        "context_length": args.context_length, "fixed_output_length": args.fixed_output_length,
        "max_len": args.max_len, "warmup": args.warmup,
    }
    if measured and args.measurement == "efficency":
        tokens_out = list(map(len, outputs))
        metrics["throughput"] = decode_throughput(len(samples), metrics["decode_steps"],
                                                  metrics["decode_latency"])
        metrics["prefill_tokens_per_s"] = (sum(input_lengths) / metrics["ttft"]
                                           if metrics["ttft"] else None)
        metrics["decode_tokens_per_s"] = (sum(c - 1 for c in tokens_out) / metrics["decode_latency"]
                                          if metrics["decode_steps"] and metrics["decode_latency"] else None)
        metrics["output_tokens_per_s"] = (sum(tokens_out) / metrics["latency"]
                                          if metrics["latency"] else None)
    stats.update(metrics)
    return predictions, stats


def write_jsonl_atomic(path, records):
    """Publish a complete result file without leaving a partially written file."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", encoding="utf-8", delete=False) as handle:
            temporary = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def replace_jsonl_records(path, records, index_field):
    """Replace overlapping sample IDs atomically when updating legacy VRAM data."""
    path = Path(path)

    def indices(record):
        values = record[index_field] if index_field == "indices" else [record[index_field]]
        return {str(index) for index in values}

    updated_indices = {index for record in records for index in indices(record)}
    previous = load_data(path) if path.exists() else []
    result = [record for record in previous if indices(record).isdisjoint(updated_indices)] + records
    write_jsonl_atomic(path, result)


def get_pred(model, tokenizer, samples, max_new_tokens, out_path, args):
    predictions, stats = run_batch(model, tokenizer, samples, max_new_tokens, args)
    metrics_path = args.save_dir / f"{args.task}.{args.measurement}.jsonl"
    prediction_records = []
    for sample, prediction, input_length, output_length in zip(
        samples, predictions, stats["input_tokens"], stats["output_tokens"],
    ):
        record = {
            "pred": prediction, "index": sample["index"],
            "batch_size": len(samples), "input_tokens": input_length, "output_tokens": output_length,
        }
        if args.benchmark == "LongBench":
            record.update(answers=sample["answers"], all_classes=sample.get("all_classes"),
                          length=sample.get("length", input_length))
        else:
            record.update(outputs=sample["outputs"], input=sample["input"],
                          others=sample.get("others", {}), truncation=sample.get("truncation", -1),
                          length=sample.get("length", -1))
        prediction_records.append(record)
    if args.measurement == "vram":
        replace_jsonl_records(metrics_path, [stats], "indices")
        replace_jsonl_records(out_path, prediction_records, "index")
    else:
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stats, ensure_ascii=False) + "\n")
        with Path(out_path).open("a", encoding="utf-8") as handle:
            for record in prediction_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return stats


def write_summary(args):
    metrics_path = args.save_dir / f"{args.task}.{args.measurement}.jsonl"
    records = (load_efficiency_records(metrics_path) if args.measurement == "efficency"
               else load_data(metrics_path) if metrics_path.exists() else [])
    groups = {}
    for record in records:
        groups.setdefault(str(record["batch_size"]), []).append(record)
    summary = {"measurement": args.measurement, "num_batches": len(records),
               "num_samples": sum(record["batch_size"] for record in records), "by_batch_size": {}}
    table = [f"{args.measurement} | task={args.task} | model={args.model}"]
    csv_rows = []

    def fmt_number(value, digits):
        return "N/A" if value is None else f"{value:.{digits}f}"

    for batch_size, group in groups.items():
        values = {"num_batches": len(group), "num_samples": sum(record["batch_size"] for record in group)}
        row = {"batch": batch_size, "batches": len(group), "samples": values["num_samples"]}
        if args.measurement == "efficency":
            tpots = [record["TPOT"] for record in group if record["TPOT"] is not None]
            throughput_records = [record for record in group if record.get("throughput") is not None]
            decode_records = [record for record in throughput_records
                              if record.get("decode_steps", 0) > 0 and record.get("decode_latency", 0) > 0]
            decode_tokens = sum(record["batch_size"] * record["decode_steps"] for record in decode_records)
            decode_time = sum(record["decode_latency"] for record in decode_records)
            weighted_steps = sum(record["decode_steps"] for record in decode_records)
            prefill_records = [record for record in group if record.get("ttft", 0)]
            prefill_tokens = sum(sum(record["input_tokens"]) for record in prefill_records)
            prefill_time = sum(record["ttft"] for record in prefill_records)
            decode_out = sum(sum(c - 1 for c in record["output_tokens"]) for record in decode_records)
            values.update(mean_ttft_s=sum(record["ttft"] for record in group) / len(group),
                          mean_tpot_ms=sum(tpots) / len(tpots) if tpots else None,
                          weighted_tpot_ms=decode_time * 1000 / weighted_steps if weighted_steps else None,
                          num_throughput_batches=len(throughput_records),
                          mean_throughput_tokens_per_s=(sum(record["throughput"] for record in throughput_records)
                                                        / len(throughput_records) if throughput_records else None),
                          throughput_tokens_per_s=(decode_tokens / decode_time if decode_time > 0
                                                   else 0.0 if throughput_records else None),
                          prefill_tokens_per_s=(prefill_tokens / prefill_time if prefill_time else None),
                          decode_tokens_per_s=(decode_out / decode_time if decode_time else None))
            if len(table) == 1:
                table.append("Batch  Batches  Samples  TTFT(s)  TPOT(ms)  weightedTPOT(ms)  "
                             "throughput(token/s)  prefill(token/s)  decode(token/s)")
            table.append(f"{batch_size:>5}  {len(group):>7}  {values['num_samples']:>7}  "
                         f"{fmt_number(values['mean_ttft_s'], 4)}  {fmt_number(values['mean_tpot_ms'], 3)}  "
                         f"{fmt_number(values['weighted_tpot_ms'], 3)}  "
                         f"{fmt_number(values['throughput_tokens_per_s'], 2)}  "
                         f"{fmt_number(values['prefill_tokens_per_s'], 2)}  "
                         f"{fmt_number(values['decode_tokens_per_s'], 2)}")
            row.update(ttft_s=values["mean_ttft_s"], tpot_ms=values["mean_tpot_ms"],
                       weightedTPOT_ms=values["weighted_tpot_ms"],
                       throughput_tokens_per_s=values["throughput_tokens_per_s"],
                       prefill_tokens_per_s=values["prefill_tokens_per_s"],
                       decode_tokens_per_s=values["decode_tokens_per_s"])
        else:
            values.update(peak_alloc_mib=max(record["peak_alloc"] for record in group),
                          peak_reserved_mib=max(record["peak_reserved"] for record in group))
            for field in ("peak_alloc_by_device", "peak_reserved_by_device"):
                devices = {device for record in group for device in record[field]}
                values[field] = {device: max(record[field].get(device, 0.0) for record in group)
                                 for device in sorted(devices)}
            decode_records = [record for record in group if DECODE_MEMORY_FIELDS.issubset(record)
                              and record["decode_num_steps"] > 0]
            steps = sum(record["decode_num_steps"] for record in decode_records)
            devices = {device for record in decode_records for device in record["decode_avg_alloc_by_device"]}
            averages = {
                device: sum(record["decode_avg_alloc_by_device"][device] * record["decode_num_steps"]
                            for record in decode_records if device in record["decode_avg_alloc_by_device"])
                / sum(record["decode_num_steps"] for record in decode_records
                      if device in record["decode_avg_alloc_by_device"])
                for device in sorted(devices)
            }
            peaks = {device: max(record["decode_peak_alloc_by_device"][device] for record in decode_records
                                 if device in record["decode_peak_alloc_by_device"])
                     for device in sorted(devices)}
            values.update(decode_num_steps=steps, num_decode_batches=len(decode_records),
                          decode_avg_alloc_mib=max(averages.values(), default=None),
                          decode_peak_alloc_mib=max(peaks.values(), default=None),
                          decode_avg_alloc_by_device=averages, decode_peak_alloc_by_device=peaks)
            sorted_devices = sorted(values["peak_alloc_by_device"])
            if len(table) == 1:
                table.append("Batch  Batches  Samples  peak_alloc(mib)  peak_reserved(mib)  per-device")
            table.append(f"{batch_size:>5}  {len(group):>7}  {values['num_samples']:>7}  "
                         f"{fmt_number(values['peak_alloc_mib'], 2)}  "
                         f"{fmt_number(values['peak_reserved_mib'], 2)}")
            row.update(peak_alloc_mib=values["peak_alloc_mib"], peak_reserved_mib=values["peak_reserved_mib"])
            for device in sorted_devices:
                table.append(f"  {device}: alloc={fmt_number(values['peak_alloc_by_device'][device], 2)}, "
                             f"reserved={fmt_number(values['peak_reserved_by_device'][device], 2)}")
                row[f"{device}_peak_alloc_mib"] = values["peak_alloc_by_device"][device]
                row[f"{device}_peak_reserved_mib"] = values["peak_reserved_by_device"][device]
        summary["by_batch_size"][batch_size] = values
        csv_rows.append(row)
    base = args.save_dir / f"{args.task}.{args.measurement}.summary"
    base.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    base.with_suffix(".summary.txt").write_text("\n".join(table) + "\n", encoding="utf-8")
    if csv_rows:
        with base.with_suffix(".summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for r in csv_rows for key in r)))
            writer.writeheader()
            writer.writerows(csv_rows)
    print("\n".join(table))
    print(f"Measurement summary JSON: {base.with_suffix('.summary.json')}")
    return base.with_suffix(".summary.json")


def main(argv=None, measurement="efficency"):
    args = parse_args(argv, measurement=measurement)
    data, max_new_tokens, out_path = load_dataset(args)
    if not data:
        print(f"[{args.task}] All selected examples already predicted, skipping model loading.")
        write_summary(args)
        return
    import numpy as np
    import torch
    from tqdm import tqdm

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args, dtype)
    warmup_batch = data[:args.batch_size]
    for iteration in range(args.warmup):
        print(f"Warmup {iteration + 1}/{args.warmup} (excluded from results)")
        run_batch(model, tokenizer, warmup_batch, max_new_tokens, args, measured=False)
    for samples in tqdm(iter_batches(data, args.batch_size),
                        total=(len(data) + args.batch_size - 1) // args.batch_size,
                        desc=f"Predicting {args.task}"):
        get_pred(model, tokenizer, samples, max_new_tokens, out_path, args)
    print(f"[{args.task}] Predictions saved to {out_path}")
    write_summary(args)


if __name__ == "__main__":
    main()
