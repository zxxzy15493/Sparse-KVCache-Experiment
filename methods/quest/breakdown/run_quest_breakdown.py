import argparse
import gc
import importlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
if str(METHOD_ROOT) not in sys.path:
  sys.path.insert(0, str(METHOD_ROOT))

from breakdown.quest_breakdown_core import QuestBreakdownProfiler, write_csv

def parse_args():
  parser = argparse.ArgumentParser(
    description="Component-level timing for LLaMA/Qwen evaluation Quest attention."
  )
  parser.add_argument("--model", "--model_path", dest="model", type=str, required=True)
  parser.add_argument(
    "--dataset",
    "--dataset_path",
    dest="dataset",
    type=str,
    default="benchmarks/myinput.txt",
    help="Plain text prompt file. The script reads the whole file.",
  )
  parser.add_argument("--output_dir", type=str, default="./breakdown_core_attn_results")
  parser.add_argument(
    "--output_folder_name",
    "--output_subdir",
    dest="output_folder_name",
    type=str,
    default=None,
    help=(
      "Override the auto-generated result folder under output_dir/model. "
      "Defaults to <quest_module>-budget<token_budget>-chunk<chunk_size>."
    ),
  )
  parser.add_argument("--input_lengths", "--seqlens", dest="input_lengths", type=str, default="4096,65536")
  parser.add_argument("--num_runs", "--iteration", dest="num_runs", type=int, default=4)
  parser.add_argument("--max_new_tokens", type=int, default=512)
  parser.add_argument("--token_budget", type=int, default=1024)
  parser.add_argument("--chunk_size", type=int, default=16)
  parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
  parser.add_argument(
    "--quest_module",
    type=str,
    default="evaluation.quest_attention_retrieve",
    help="Use evaluation.quest_attention_retrieve for LLaMA, or evaluation.quest_qwen_attention_kernel for Qwen.",
  )
  parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
  parser.add_argument("--chat_template", action="store_true")
  parser.add_argument("--task", type=str, default=None)
  parser.add_argument("--seed", type=int, default=42)
  return parser.parse_args()


def seed_everything(seed: int) -> None:
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


def load_text(path: str) -> str:
  dataset_path = Path(path)
  if not dataset_path.is_absolute() and not dataset_path.exists():
    dataset_path = REPO_ROOT / dataset_path
  for encoding in ("utf-8", "utf-8-sig", "gb18030"):
    try:
      with open(dataset_path, "r", encoding=encoding) as f:
        return f.read().strip()
    except UnicodeDecodeError:
      continue
  with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
    return f.read().strip()


def maybe_chat_prompt(tokenizer, prompt: str, enabled: bool) -> str:
  if not enabled:
    return prompt
  try:
    return tokenizer.apply_chat_template(
      [{"role": "user", "content": prompt}],
      add_generation_prompt=True,
      tokenize=False,
    )
  except Exception:
    return prompt


def enable_tuple_cache_if_needed(model_name: str) -> None:
  lower_name = model_name.lower()
  if "llama" in lower_name or "longchat" in lower_name:
    try:
      from evaluation.llama import enable_tuple_kv_cache_for_llama

      enable_tuple_kv_cache_for_llama()
    except Exception as exc:
      print(f"[WARN] failed to enable llama tuple cache: {exc}")


def load_model_and_tokenizer(args, dtype: torch.dtype):
  enable_tuple_cache_if_needed(args.model)
  tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

  kwargs = {
    "trust_remote_code": True,
    "torch_dtype": dtype,
    "device_map": "auto",
  }
  if args.attn_implementation:
    kwargs["attn_implementation"] = args.attn_implementation

  model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
  return model, tokenizer


def enable_quest_eval_attention(model, eval_module, args):
  quest_args = SimpleNamespace(
    token_budget=args.token_budget,
    chunk_size=args.chunk_size,
    save_path=None,
    task=args.task,
  )
  for name in ("enable_quest_qwen_attention_eval", "enable_quest_attention_eval"):
    fn = getattr(eval_module, name, None)
    if fn is not None:
      return fn(model, quest_args)
  raise RuntimeError(
    f"{eval_module.__name__} has no LLaMA/Qwen Quest enable function."
  )


def build_output_paths(args, input_len: int):
  model_dir = os.path.basename(args.model.rstrip("/"))
  dataset_name = Path(args.dataset).stem
  module_tag = args.quest_module.rsplit(".", 1)[-1]
  default_folder = f"{module_tag}-budget{args.token_budget}-chunk{args.chunk_size}"
  folder_name = (args.output_folder_name or "").strip() or default_folder
  if Path(folder_name).name != folder_name or folder_name in {".", ".."}:
    raise ValueError("--output_folder_name must be a single folder name, not a path.")
  out_dir = Path(args.output_dir) / model_dir / folder_name
  out_dir.mkdir(parents=True, exist_ok=True)
  len_tag = f"{input_len // 1024}k" if input_len % 1024 == 0 else str(input_len)
  jsonl_path = out_dir / f"{dataset_name}_{len_tag}.jsonl"
  csv_path = out_dir / f"{dataset_name}_{len_tag}_breakdown.csv"
  return jsonl_path, csv_path


@torch.inference_mode()
def run_one_profile(model, input_ids, max_new_tokens: int, profiler: QuestBreakdownProfiler):
  generated = []

  profiler.reset_run()
  outputs = profiler.measure_forward(
    "prefill",
    model,
    input_ids=input_ids,
    past_key_values=None,
    use_cache=True,
  )
  past_key_values = outputs.past_key_values
  next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
  generated.append(next_token)

  current_input_ids = next_token
  for _ in range(max(0, max_new_tokens - 1)):
    outputs = profiler.measure_forward(
      "decode",
      model,
      input_ids=current_input_ids,
      past_key_values=past_key_values,
      use_cache=True,
    )
    past_key_values = outputs.past_key_values
    current_input_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated.append(current_input_ids)

  return torch.cat(generated, dim=-1)


def main():
  args = parse_args()
  assert torch.cuda.is_available(), "Quest evaluation breakdown requires CUDA."
  seed_everything(args.seed)

  dtype = getattr(torch, args.dtype)
  eval_module = importlib.import_module(args.quest_module)
  profiler = QuestBreakdownProfiler()
  profiler.patch_eval_module(eval_module)

  print(f"Loading model from {args.model} ...")
  model, tokenizer = load_model_and_tokenizer(args, dtype)
  enable_quest_eval_attention(model, eval_module, args)
  profiler.patch_model_modules(model)

  raw_text = load_text(args.dataset)
  prompt = maybe_chat_prompt(tokenizer, raw_text, args.chat_template)
  input_ids_full = tokenizer(prompt, return_tensors="pt").input_ids
  target_lengths = [int(x) for x in args.input_lengths.split(",") if x.strip()]

  for input_len in target_lengths:
    if input_len > input_ids_full.shape[1]:
      print(
        f"[WARN] dataset has {input_ids_full.shape[1]} tokens, "
        f"skip requested length {input_len}."
      )
      continue

    jsonl_path, csv_path = build_output_paths(args, input_len)
    rows = []
    print(f"[RUNNING] input_len={input_len}, runs={args.num_runs}")

    for run_idx in range(1, args.num_runs + 1):
      torch.cuda.empty_cache()
      input_ids = input_ids_full[:, :input_len].to(model.device)
      generated_ids = run_one_profile(
        model, input_ids, args.max_new_tokens, profiler
      )
      pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

      row = profiler.summary_row(
        run_idx=run_idx,
        input_len=input_len,
        generated_tokens=int(generated_ids.shape[-1]),
        quest_module=args.quest_module,
        token_budget=args.token_budget,
        chunk_size=args.chunk_size,
      )
      rows.append(row)

      with open(jsonl_path, "a", encoding="utf-8") as f:
        json.dump(
          {
            "run_idx": run_idx,
            "input_len": input_len,
            "generated_tokens": int(generated_ids.shape[-1]),
            "quest_module": args.quest_module,
            "token_budget": args.token_budget,
            "chunk_size": args.chunk_size,
            "pred": pred,
          },
          f,
          ensure_ascii=False,
        )
        f.write("\n")
      write_csv(str(csv_path), rows)

      print(
        f" run={run_idx} "
        f"prefill={row['prefill_total_time']:.6f}s "
        f"prefill_attn={row['prefill_attn_time']:.6f}s "
        f"prefill_ffn={row['prefill_ffn_time']:.6f}s "
        f"decode_avg={row['decode_total_avg_time']:.6f}s "
        f"decode_attn_avg={row['decode_attn_avg_time']:.6f}s "
        f"decode_ffn_avg={row['decode_ffn_avg_time']:.6f}s"
      )

      gc.collect()
      torch.cuda.empty_cache()

    print(f"[DONE] predictions: {jsonl_path}")
    print(f"[DONE] breakdown:  {csv_path}")


if __name__ == "__main__":
  main()
