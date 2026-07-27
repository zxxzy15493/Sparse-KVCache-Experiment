import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
if str(METHOD_ROOT) not in sys.path:
  sys.path.insert(0, str(METHOD_ROOT))

from quest import LlamaForCausalLM, Qwen2ForCausalLM
from breakdown_kernel.quest_kernel_breakdown_core import (
  QuestKernelBreakdownProfiler,
  write_csv,
)


def parse_args():
  parser = argparse.ArgumentParser(description="Quest kernel component-level timing breakdown.")
  parser.add_argument("--model", "--model_path", dest="model", type=str, required=True)
  parser.add_argument(
    "--model_type",
    type=str,
    default="auto",
    choices=["auto", "llama", "qwen"],
    help="Quest kernel model wrapper. auto infers from model path.",
  )
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
      "Defaults to <model_type>-budget<token_budget>-page<page_size>."
    ),
  )
  parser.add_argument("--input_lengths", "--seqlens", dest="input_lengths", type=str, default="4096,65536")
  parser.add_argument("--num_runs", "--iteration", dest="num_runs", type=int, default=4)
  parser.add_argument("--max_new_tokens", type=int, default=512)
  parser.add_argument("--page_size", type=int, default=16)
  parser.add_argument("--token_budget", type=int, default=1024)
  parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
  parser.add_argument("--chat_template", action="store_true")
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


def infer_model_type(model_path: str) -> str:
  if "qwen" in model_path.lower():
    return "qwen"
  return "llama"


def load_model_and_tokenizer(args, dtype: torch.dtype):
  model_type = infer_model_type(args.model) if args.model_type == "auto" else args.model_type
  model_cls = Qwen2ForCausalLM if model_type == "qwen" else LlamaForCausalLM

  model = model_cls.from_pretrained(
    args.model,
    device_map="cuda:0",
    torch_dtype=dtype,
  ).eval()

  tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
  return model, tokenizer, model_type


def build_output_paths(args, model_type: str, input_len: int):
  model_dir = os.path.basename(args.model.rstrip("/"))
  dataset_name = Path(args.dataset).stem
  default_folder = f"{model_type}-budget{args.token_budget}-page{args.page_size}"
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
def run_one_profile(model, input_ids, max_new_tokens: int, profiler: QuestKernelBreakdownProfiler):
  generated = []

  profiler.reset_run()
  outputs = profiler.measure_forward(
    "prefill",
    model,
    input_ids=input_ids,
    use_cache=True,
    num_logits_to_keep=1,
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
      num_logits_to_keep=1,
    )
    past_key_values = outputs.past_key_values
    current_input_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated.append(current_input_ids)

  return torch.cat(generated, dim=-1)


def main():
  args = parse_args()
  assert torch.cuda.is_available(), "Quest kernel breakdown requires CUDA."
  seed_everything(args.seed)

  dtype = getattr(torch, args.dtype)
  target_lengths = [int(x) for x in args.input_lengths.split(",") if x.strip()]
  max_seq_len = max(target_lengths) + args.max_new_tokens + 512
  device = torch.device("cuda:0")

  profiler = QuestKernelBreakdownProfiler()
  profiler.patch_quest_ops()
  profiler.patch_controller()

  print(f"Loading model from {args.model} ...")
  model, tokenizer, model_type = load_model_and_tokenizer(args, dtype)
  quest_dtype = torch.float16 if dtype == torch.bfloat16 else dtype
  if quest_dtype != dtype:
    print(f"Quest kernels use {quest_dtype} while model weights stay in {dtype}.")

  model.quest_init(
    page_size=args.page_size,
    max_seq_len=max_seq_len,
    token_budget=args.token_budget,
    dtype=quest_dtype,
    device=device,
  )
  profiler.patch_model_modules(model)

  raw_text = load_text(args.dataset)
  prompt = maybe_chat_prompt(tokenizer, raw_text, args.chat_template)
  input_ids_full = tokenizer(prompt, return_tensors="pt").input_ids

  for input_len in target_lengths:
    if input_len > input_ids_full.shape[1]:
      print(
        f"[WARN] dataset has {input_ids_full.shape[1]} tokens, "
        f"skip requested length {input_len}."
      )
      continue

    jsonl_path, csv_path = build_output_paths(args, model_type, input_len)
    rows = []
    print(f"[RUNNING] model_type={model_type}, input_len={input_len}, runs={args.num_runs}")

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
      )
      rows.append(row)

      with open(jsonl_path, "a", encoding="utf-8") as f:
        json.dump(
          {
            "run_idx": run_idx,
            "model_type": model_type,
            "input_len": input_len,
            "generated_tokens": int(generated_ids.shape[-1]),
            "token_budget": args.token_budget,
            "page_size": args.page_size,
            "pred": pred,
          },
          f,
          ensure_ascii=False,
        )
        f.write("\n")

      print(
        f" run={run_idx} "
        f"prefill={row['prefill_total_time']:.6f}s "
        f"prefill_attn={row['prefill_attn_time']:.6f}s "
        f"prefill_ffn={row['prefill_ffn_time']:.6f}s "
        f"decode_avg={row['decode_total_avg_time']:.6f}s "
        f"decode_attn_avg={row['decode_attn_avg_time']:.6f}s "
        f"decode_ffn_avg={row['decode_ffn_avg_time']:.6f}s"
      )

      model.quest_clear()
      gc.collect()
      torch.cuda.empty_cache()

    write_csv(str(csv_path), rows)
    print(f"[DONE] predictions: {jsonl_path}")
    print(f"[DONE] breakdown:  {csv_path}")


if __name__ == "__main__":
  main()
