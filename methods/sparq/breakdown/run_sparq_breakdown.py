import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
if str(METHOD_ROOT) not in sys.path:
  sys.path.insert(0, str(METHOD_ROOT))

from llminference.myexperiments import Sparsity, SparsityMethods
from breakdown.sparq_breakdown_core import SparqBreakdownProfiler, write_csv


def parse_args():
  parser = argparse.ArgumentParser(description="SparQ component-level timing breakdown.")
  parser.add_argument("--model", "--model_path", dest="model", type=str, required=True)
  parser.add_argument("--model_type", type=str, default="auto", choices=["auto", "llama", "qwen"])
  parser.add_argument("--dataset", "--dataset_path", dest="dataset", type=str, default="benchmarks/myinput.txt")
  parser.add_argument("--output_dir", type=str, default="./breakdown_core_attn_results")
  parser.add_argument(
    "--output_folder_name",
    "--output_subdir",
    dest="output_folder_name",
    type=str,
    default=None,
    help=(
      "Override the auto-generated result folder under output_dir/model. "
      "Defaults to <model_type>-ann-k<k>-local<local_k>-r<rank>."
    ),
  )
  parser.add_argument("--input_lengths", "--seqlens", dest="input_lengths", type=str, default="8192")
  parser.add_argument("--num_runs", "--iteration", dest="num_runs", type=int, default=4)
  parser.add_argument("--max_new_tokens", type=int, default=32)
  parser.add_argument("--k", type=int, default=1024)
  parser.add_argument("--local_k", type=int, default=32)
  parser.add_argument("--rank", type=int, default=16)
  parser.add_argument("--score", type=str, default="sparse_q", choices=["sparse_q", "low_rank"])
  parser.add_argument("--reallocate_to_mean_value", action="store_true", default=True)
  parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
  parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
  parser.add_argument("--chat_template", action="store_true")
  parser.add_argument("--seed", type=int, default=42)
  return parser.parse_args()


def seed_everything(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


def infer_model_type(model_path: str) -> str:
  return "qwen" if "qwen" in model_path.lower() else "llama"


def load_text(path: str) -> str:
  dataset_path = Path(path)
  if not dataset_path.is_absolute() and not dataset_path.exists():
    dataset_path = REPO_ROOT / dataset_path
  for encoding in ("utf-8", "utf-8-sig", "gb18030"):
    try:
      with open(dataset_path, "r", encoding=encoding) as f:
        return f.read()
    except UnicodeDecodeError:
      continue
  with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
    return f.read()


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


def load_model_and_tokenizer(args, dtype: torch.dtype):
  tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

  model_kwargs: Dict[str, Any] = {
    "trust_remote_code": True,
    "torch_dtype": dtype,
    "device_map": "auto",
  }
  if args.attn_implementation:
    model_kwargs["attn_implementation"] = args.attn_implementation

  model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
  sparq = Sparsity(
    name="ann",
    k=args.k,
    local_k=args.local_k,
    rank=args.rank,
    score=args.score,
    reallocate_to_mean_value=args.reallocate_to_mean_value,
  )
  model = SparsityMethods.apply(sparq, model).eval()
  return model, tokenizer


def build_output_paths(args, model_type: str, input_len: int):
  model_dir = os.path.basename(args.model.rstrip("/"))
  dataset_name = Path(args.dataset).stem
  default_folder = f"{model_type}-ann-k{args.k}-local{args.local_k}-r{args.rank}"
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
def run_one_profile(model, input_ids, max_new_tokens: int, profiler: SparqBreakdownProfiler):
  generated = []
  past_key_values = None
  current_input_ids = input_ids

  for step in range(max_new_tokens):
    phase = "prefill" if step == 0 else "decode"
    outputs = profiler.measure_forward(
      phase,
      model,
      input_ids=current_input_ids,
      past_key_values=past_key_values,
      use_cache=True,
      num_logits_to_keep=1,
    )
    past_key_values = outputs.past_key_values
    current_input_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated.append(current_input_ids)

  return torch.cat(generated, dim=-1) if generated else torch.empty((1, 0), device=input_ids.device, dtype=input_ids.dtype)


def main() -> None:
  args = parse_args()
  assert torch.cuda.is_available(), "SparQ breakdown requires CUDA."
  seed_everything(args.seed)

  dtype = getattr(torch, args.dtype)
  model_type = infer_model_type(args.model) if args.model_type == "auto" else args.model_type
  target_lengths = [int(x) for x in args.input_lengths.split(",") if x.strip()]

  profiler = SparqBreakdownProfiler()
  profiler.patch_sparq()

  print(f"Loading model from {args.model} ...")
  model, tokenizer = load_model_and_tokenizer(args, dtype)
  profiler.patch_model_modules(model)

  raw_text = load_text(args.dataset)
  prompt = maybe_chat_prompt(tokenizer, raw_text, args.chat_template)
  input_ids_full = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids

  for input_len in target_lengths:
    if input_len > input_ids_full.shape[1]:
      print(f"[WARN] dataset has {input_ids_full.shape[1]} tokens, skip requested length {input_len}.")
      continue

    jsonl_path, csv_path = build_output_paths(args, model_type, input_len)
    rows = []
    print(f"[RUNNING] model_type={model_type}, input_len={input_len}, runs={args.num_runs}")

    for run_idx in range(1, args.num_runs + 1):
      gc.collect()
      torch.cuda.empty_cache()
      profiler.reset_run()

      input_ids = input_ids_full[:, :input_len].to(model.device)
      generated_ids = run_one_profile(model, input_ids, args.max_new_tokens, profiler)
      pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

      row = profiler.summary_row(
        run_idx=run_idx,
        input_len=input_len,
        generated_tokens=int(generated_ids.shape[-1]),
        model_type=model_type,
        k=args.k,
        local_k=args.local_k,
        rank=args.rank,
        score=args.score,
      )
      rows.append(row)

      with open(jsonl_path, "a", encoding="utf-8") as f:
        json.dump(
          {
            "run_idx": run_idx,
            "model_type": model_type,
            "input_len": input_len,
            "generated_tokens": int(generated_ids.shape[-1]),
            "k": args.k,
            "local_k": args.local_k,
            "rank": args.rank,
            "score": args.score,
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
        f"decode_ffn_avg={row['decode_ffn_avg_time']:.6f}s "
        f"decode_retrieve={row['decode_retrieve_time']:.6f}s"
      )

    print(f"[DONE] jsonl={jsonl_path}")
    print(f"[DONE] csv={csv_path}")


if __name__ == "__main__":
  main()
