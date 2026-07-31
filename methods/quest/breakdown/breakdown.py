import argparse
import csv
import gc
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


PROJECT_ROOT = Path("..")
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from quest.global_time import (
  get_decode_token_count,
  get_summary_times_ms,
  init_timer,
  reset_timer,
  set_recording_state,
)
from quest.models.myllama import LlamaForCausalLM
from quest.models.qwen2 import Qwen2ForCausalLM


STAGE_ORDER = (
  "prefill_pre_ffn",
  "prefill_post_ffn",
  "prefill_others",
  "prefill_build_index",
  "prefill_attn",
  "decode_pre_ffn",
  "decode_post_ffn",
  "decode_others",
  "decode_write_cache",
  "decode_retrieve",
  "decode_attn",
  "total_wall",
)


def parse_int_list(value: str) -> List[int]:
  return [int(part.strip()) for part in value.replace(" ", ",").split(",") if part.strip()]


def parse_args():
  parser = argparse.ArgumentParser(description="Quest myllama component breakdown benchmark.")
  parser.add_argument("--model_name", type=str, default="Llama")
  parser.add_argument("--model_type", type=str, default="llama", choices=["llama", "qwen2"])
  parser.add_argument("--model_path", type=str, required=True)
  parser.add_argument("--dataset_path", type=str, required=True)
  parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
  parser.add_argument("--input_lengths", "--seqlens", type=str, default="4096,8192,16384,32768,65536,131072")
  parser.add_argument("--max_new_tokens", type=int, default=32)
  parser.add_argument("--page_size", type=int, default=16)
  parser.add_argument("--token_budget", type=int, default=1024)
  parser.add_argument("--iteration", type=int, default=3, help="Measured rounds per input length.")
  parser.add_argument("--warmup_iteration", type=int, default=2, help="Warmup rounds per input length.")
  parser.add_argument("--output_dir", type=Path, default=Path("../mybreakdown_results"))
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


def load_prompt_from_file(file_path: str) -> str:
  assert os.path.exists(file_path), f"input file does not exist: {file_path}"
  for encoding in ("utf-8", "utf-8-sig", "gb18030"):
    try:
      with open(file_path, "r", encoding=encoding) as f:
        return f.read()
    except UnicodeDecodeError:
      continue
  with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    return f.read()


def build_chat_prompt(tokenizer, prompt: str) -> str:
  try:
    return tokenizer.apply_chat_template(
      [{"role": "user", "content": prompt}],
      add_generation_prompt=True,
      tokenize=False,
    )
  except Exception:
    return prompt


def load_model(model_path: str, dtype: torch.dtype, device: torch.device, model_type: str):
  torch.set_default_dtype(dtype)
  model_cls = LlamaForCausalLM if model_type == "llama" else Qwen2ForCausalLM
  with device:
    model = model_cls.from_pretrained(
      model_path,
      device_map=device,
      torch_dtype=dtype,
    )
  return model.eval()


def build_input_ids(tokenizer, prompt: str, max_input_len: int, device: torch.device) -> torch.Tensor:
  input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids
  if input_ids.shape[1] < max_input_len:
    repeat_times = (max_input_len + input_ids.shape[1] - 1) // input_ids.shape[1]
    input_ids = input_ids.repeat(1, repeat_times)
  return input_ids[:, :max_input_len].to(device)


@torch.inference_mode()
def run_generation(model, input_ids: torch.Tensor, max_new_tokens: int):
  generated = []
  current_input_ids = input_ids
  past_key_values = None

  torch.cuda.synchronize()
  start_ts = time.perf_counter()

  for token_idx in range(max_new_tokens):
    outputs = model(
      input_ids=current_input_ids,
      past_key_values=past_key_values,
      use_cache=True,
      num_logits_to_keep=1,
    )
    past_key_values = outputs.past_key_values
    current_input_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated.append(current_input_ids)

  torch.cuda.synchronize()
  total_s = time.perf_counter() - start_ts
  return torch.cat(generated, dim=-1), total_s


def phase_for_stage(stage_name: str) -> str:
  if stage_name.startswith("prefill_"):
    return "prefill"
  if stage_name.startswith("decode_"):
    return "decode"
  return "overall"


def aggregation_for_stage(stage_name: str) -> str:
  if stage_name.startswith("decode_"):
    return "decode_per_token_avg"
  if stage_name.startswith("prefill_"):
    return "prefill_total"
  return "wall_total"


def stage_rows_from_run(
  *,
  args,
  input_len: int,
  run_idx: int,
  total_s: float,
  generated_tokens: int,
  decode_token_count: int,
  times_ms: Dict[str, float],
) -> List[dict]:
  rows = []
  now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  times = dict(times_ms)
  times["total_wall"] = total_s * 1000.0

  for stage_name in STAGE_ORDER:
    value_ms = float(times.get(stage_name, 0.0))
    rows.append(
      {
        "time": now,
        "model": args.model_name,
        "model_path": args.model_path,
        "model_type": args.model_type,
        "input_len": input_len,
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": generated_tokens,
        "decode_token_count": decode_token_count,
        "token_budget": args.token_budget,
        "page_size": args.page_size,
        "run_idx": run_idx,
        "phase": phase_for_stage(stage_name),
        "stage": stage_name,
        "aggregation": aggregation_for_stage(stage_name),
        "time_ms": value_ms,
        "time_s": value_ms / 1000.0,
      }
    )
  return rows


def average_rows(rows: Iterable[dict]) -> List[dict]:
  grouped = {}
  counts = {}
  examples = {}
  for row in rows:
    key = (row["input_len"], row["stage"])
    grouped[key] = grouped.get(key, 0.0) + float(row["time_ms"])
    counts[key] = counts.get(key, 0) + 1
    examples.setdefault(key, row)

  out = []
  for key in sorted(grouped.keys()):
    example = dict(examples[key])
    avg_ms = grouped[key] / counts[key]
    example["run_idx"] = "avg"
    example["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    example["time_ms"] = avg_ms
    example["time_s"] = avg_ms / 1000.0
    example["measure_rounds"] = counts[key]
    out.append(example)
  return out


def write_csv(path: Path, rows: List[dict]) -> None:
  if not rows:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = list(rows[0].keys())
  with open(path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def main():
  args = parse_args()
  assert torch.cuda.is_available(), "Quest breakdown requires CUDA."
  seed_everything(args.seed)

  input_lengths = parse_int_list(args.input_lengths)
  device = torch.device("cuda:0")
  dtype = getattr(torch, args.dtype)
  max_seq_len = max(input_lengths) + args.max_new_tokens + 512

  prompt_text = load_prompt_from_file(args.dataset_path)
  args.output_dir.mkdir(parents=True, exist_ok=True)

  print(f"Loading model from {args.model_path} ...", flush=True)
  model = load_model(args.model_path, dtype, device, args.model_type)
  tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

  quest_dtype = torch.float16 if dtype == torch.bfloat16 else dtype
  if quest_dtype != dtype:
    print(f"Quest kernels use {quest_dtype} while model weights stay in {dtype}.", flush=True)

  model.quest_init(
    page_size=args.page_size,
    max_seq_len=max_seq_len,
    token_budget=args.token_budget,
    dtype=quest_dtype,
    device=device,
  )
  init_timer(model.config.num_hidden_layers)

  prompt = build_chat_prompt(tokenizer, prompt_text) if args.chat_template else prompt_text
  input_ids_all = build_input_ids(tokenizer, prompt, max(input_lengths), device)
  print(f"Prepared input_ids shape: {tuple(input_ids_all.shape)}", flush=True)

  suffix = f"{args.model_name}-budget{args.token_budget}-chunk_size{args.page_size}-out{args.max_new_tokens}"
  run_jsonl_path = args.output_dir / f"{suffix}_runs.jsonl"
  run_csv_path = args.output_dir / f"{suffix}_runs.csv"
  summary_csv_path = args.output_dir / f"{suffix}_summary.csv"
  if run_jsonl_path.exists():
    run_jsonl_path.unlink()

  all_run_rows = []
  total_rounds = args.warmup_iteration + args.iteration

  for input_len in input_lengths:
    print(f"\n[RUNNING] input_len={input_len}, warmup={args.warmup_iteration}, measure={args.iteration}", flush=True)
    input_ids = input_ids_all[:, :input_len]

    for round_idx in tqdm(range(total_rounds), desc=f"seqlen={input_len}"):
      is_warmup = round_idx < args.warmup_iteration
      reset_timer()
      set_recording_state(not is_warmup)

      generated_ids, total_s = run_generation(model, input_ids, args.max_new_tokens)

      if not is_warmup:
        torch.cuda.synchronize()
        times_ms = get_summary_times_ms(average_decode_by_token=True)
        decode_token_count = get_decode_token_count()
        run_idx = round_idx - args.warmup_iteration + 1
        rows = stage_rows_from_run(
          args=args,
          input_len=input_len,
          run_idx=run_idx,
          total_s=total_s,
          generated_tokens=int(generated_ids.shape[-1]),
          decode_token_count=decode_token_count,
          times_ms=times_ms,
        )
        all_run_rows.extend(rows)

        pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        with open(run_jsonl_path, "a", encoding="utf-8") as f:
          json.dump(
            {
              "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "model": args.model_name,
              "input_len": input_len,
              "run_idx": run_idx,
              "generated_tokens": int(generated_ids.shape[-1]),
              "decode_token_count": decode_token_count,
              "total_s": total_s,
              "times_ms": times_ms,
              #"prediction": pred,
            },
            f,
            ensure_ascii=False,
          )
          f.write("\n")

        print(
          f" run={run_idx} total={total_s:.6f}s "
          f"prefill_others={times_ms.get('prefill_others', 0.0):.4f}ms "
          f"prefill_attn={times_ms.get('prefill_attn', 0.0):.4f}ms "
          f"decode_others={times_ms.get('decode_others', 0.0):.4f}ms "
          f"decode_retrieve={times_ms.get('decode_retrieve', 0.0):.4f}ms "
          f"decode_attn={times_ms.get('decode_attn', 0.0):.4f}ms",
          flush=True,
        )

      set_recording_state(False)
      model.quest_clear()
      gc.collect()
      torch.cuda.empty_cache()

  summary_rows = average_rows(all_run_rows)
  write_csv(run_csv_path, all_run_rows)
  write_csv(summary_csv_path, summary_rows)

  print("\n[DONE]", flush=True)
  print(f"Run jsonl : {run_jsonl_path}", flush=True)
  print(f"Run csv  : {run_csv_path}", flush=True)
  print(f"Summary  : {summary_csv_path}", flush=True)


if __name__ == "__main__":
  main()
