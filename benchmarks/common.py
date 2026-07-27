from __future__ import annotations

import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

# These are Hugging Face identifiers, not machine-specific paths. Local absolute
# paths, when needed, are stored only in local_paths.json beside this file.
MODEL_PATHS = {
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-2.5-7b-1m": "Qwen/Qwen2.5-7B-Instruct-1M",
    "glm-4-9b-1m": "zai-org/glm-4-9b-chat-1m",
    "ds-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}
MODELS = tuple(MODEL_PATHS)
MODEL_CONTEXT_LENGTHS = {
    "llama-3.1-8b": 131072,
    "qwen-2.5-7b": 131072,
    "qwen-2.5-7b-1m": 1000000,
    "glm-4-9b-1m": 1000000,
    "ds-qwen-1.5b": 32768,
}
METHOD_CONFIG_DIR = ROOT / "method_configs"
METHODS = ("full_attention", "topk", "topk32", "topp", "topp32", "pqcache", "clusterkv",
           "h2o", "keyformer", "snapkv", "streaming","quest","sparq","xattention","flexprefill", "minference", "retroinfer", "magicpig",
           "headkv", "adakv", "cakekv", "duo-attention", "pyramidkv")
PQ_METHODS = frozenset({"topk", "topk32", "topp", "topp32", "pqcache"})
FIXED_BUDGET_METHODS = PQ_METHODS | {"full_attention"}


def method_defaults(method: str) -> dict[str, object]:
    """Read the submission-visible defaults for one method."""
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Available: {', '.join(METHODS)}")
    config_path = METHOD_CONFIG_DIR / f"{method}.yaml"
    try:
        values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Missing method defaults: {config_path}") from error
    if not isinstance(values, dict):
        raise ValueError(f"Method defaults must be a YAML mapping: {config_path}")
    return values


def model_path(model: str) -> str:
    local_paths = Path(__file__).with_name("local_paths.json")
    overrides = json.loads(local_paths.read_text(encoding="utf-8")) if local_paths.exists() else {}
    if model in overrides:
        return overrides[model]
    try:
        return MODEL_PATHS[model]
    except KeyError as error:
        raise ValueError(f"Unknown model '{model}'. Available: {', '.join(MODEL_PATHS)}") from error


def model_context_length(model: str) -> int:
    return MODEL_CONTEXT_LENGTHS[model]


def model_family(model: str) -> str:
    return "glm4" if model.startswith("glm-") else "qwen" if "qwen" in model else "llama"


def parse_set(values: list[str], method: str | None = None) -> dict[str, object]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--set must use KEY=VALUE")
        key, raw = value.split("=", 1)
        try:
            parsed[key] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[key] = raw
    if method in FIXED_BUDGET_METHODS and "fixbudget" in parsed:
        raise ValueError(f"fixbudget is always true for {method}")
    return parsed


def run_name(_method: str, budget: int | None, overrides: dict[str, object] | None = None) -> str:
    values = []
    if budget is not None:
        values.append(f"budget-{budget}")
    for key, value in sorted((overrides or {}).items()):
        safe_key = re.sub(r"[^A-Za-z0-9]+", "-", key).strip("-")
        safe_value = str(value).replace("/", "-").replace(" ", "-")
        values.append(f"{safe_key}-{safe_value}")
    return "__".join(values) or "default"


def output_path(output_root: str, model: str, method: str, benchmark: str, run: str) -> Path:
    return Path(output_root) / model / method / benchmark / f"{run}.jsonl"


def middle_truncate(tokenizer, prompt: str, max_context_length: int, add_special_tokens: bool = True) -> str:
    token_ids = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]
    if len(token_ids) <= max_context_length:
        return prompt
    first = max_context_length // 2
    return tokenizer.decode(token_ids[:first], skip_special_tokens=True) + tokenizer.decode(
        token_ids[-(max_context_length - first):], skip_special_tokens=True
    )


def build_chat_prompt(tokenizer, prompt: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )
