
import os
import json
import csv
import argparse
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from llminference.myexperiments import Sparsity,SparsityMethods


ALGO_ROOT = Path(".")
def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, required=True, default="meta-llama/Llama-3.1-8B-Instruct", help="HF repo name or local model directory")
  parser.add_argument("--output_dir", type=Path, default="pred", help="Output directory for predictions, auto-creates subdir by model name")
  parser.add_argument("--model_name", type=str, default="", help="Model name for chat template and output directory name")
  parser.add_argument("--name", type=str, required=True, help="ANN selection method")
  parser.add_argument("--k", type=int, required=True, help=" Number of tokens to select")
  parser.add_argument("--local_k", type=int, required=True, help="Number of neighbor tokens")
  parser.add_argument("--score", type=str, required=True, help=" Score")
  parser.add_argument("--rank", type=int, required=True, help="Rank dimension")
  parser.add_argument("--reallocate_to_mean_value", type=bool, required=True, help="Whether to reallocate to mean value")
  parser.add_argument("--type", type=str, required=True, help="Metric type to compute")

  parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length")

  parser.add_argument("--task", type=str, default="file_prompt", help="Task name for logging")
  parser.add_argument("--dataset_path", type=str, required=True, help="Path to input text file used as prompt")
  parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate; TTFT mode forces this to 1")
  parser.add_argument("--csv_path", type=Path, default=Path("../vram_results.csv"), help="Peak VRAM CSV results file")
  

  return parser.parse_args()


def get_model_type(model_name: str, model_path: str) -> str:
  name = f"{model_name} {model_path}".lower()
  if "qwen" in name:
    return "Qwen"
  if "llama" in name:
    return "Llama"
  return model_name or Path(model_path).name


def append_vram_csv(args, algorithm: str, model_type: str, input_length: int, output_length: int, peak_alloc: int, peak_reserved: int):
  csv_path = Path(args.csv_path)
  csv_path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = [
    "algorithm",
    "model_type",
    "model_name",
    "budget",
    "input_length",
    "output_length",
    "peak_alloc_mib",
    "peak_reserved_mib",
  ]
  write_header = not csv_path.exists() or csv_path.stat().st_size == 0
  with open(csv_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
      writer.writeheader()
    writer.writerow({
      "algorithm": algorithm,
      "model_type": model_type,
      "model_name": args.model_name or args.model_path,
      "budget": args.k,
      "input_length": input_length,
      "output_length": output_length,
      "peak_alloc_mib": f"{peak_alloc / 1024**2:.2f}",
      "peak_reserved_mib": f"{peak_reserved / 1024**2:.2f}",
    })


# ---------- Utility functions ----------

def seed_everything(seed: int = 42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def load_prompt_from_file(file_path: str) -> str:
  assert os.path.exists(file_path), f"Input file not found: {file_path}"

  encodings_to_try = ["utf-8", "utf-8-sig", "gb18030"]
  for enc in encodings_to_try:
    try:
      with open(file_path, "r", encoding=enc) as f:
        return f.read()
    except UnicodeDecodeError:
      continue

  with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    return f.read()



def build_chat(tokenizer, prompt):
  prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                      add_generation_prompt=True, tokenize=False) 
  return prompt


def load_model_and_tokenizer(model_path: str, device: torch.device,args,out_path):
  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto" , 
    attn_implementation="flash_attention_2",
  )
  if args.type=="recall":
    sparq = Sparsity(name=args.name, k=args.k, local_k=args.local_k, rank=args.rank, score=args.score,
         reallocate_to_mean_value=args.reallocate_to_mean_value,type="recall",recall_save_path=out_path)
  elif args.type=="topkrate":
    sparq = Sparsity(name=args.name, k=args.k, local_k=args.local_k, rank=args.rank, score=args.score,
         reallocate_to_mean_value=args.reallocate_to_mean_value,type="topkrate",recall_save_path=out_path)
  else:  #not recall mode, skip extra params
    sparq = Sparsity(name=args.name, k=args.k, local_k=args.local_k, rank=args.rank, score=args.score,
         reallocate_to_mean_value=args.reallocate_to_mean_value)
  model = SparsityMethods.apply(sparq, model)

  if not torch.cuda.is_available():
    model.to(device)
  model.eval()
  return model, tokenizer




def VRAM(
  args,
  model,
  tokenizer,
  data,
  task_name: str,
  model_name: str,
  max_new_tokens: int,
  prompt_format: str,
  seqlen:int,
  device: torch.device,
  out_path: str,
):
  for _ in range(1):
    torch.cuda.memory.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history()
    for json_obj in tqdm(data, desc=f"Task={task_name}"):
      prompt = prompt_format.format(**json_obj)

      input_ids_full = tokenizer(prompt, return_tensors="pt").input_ids

      input_ids = input_ids_full[:, :seqlen].to(model.device)
      
      decode_latency = []
      past_key_values = None
      current_input_ids = input_ids

      import time

      with torch.no_grad():
        for i in range(max_new_tokens):
          torch.cuda.synchronize()
          start_ts = time.perf_counter()
          
          forward_kwargs = {
            "input_ids": current_input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
          }

          forward_kwargs["num_logits_to_keep"] = 1
          
          outputs = model(**forward_kwargs)
          # )


          past_key_values = outputs.past_key_values
          next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
          
          torch.cuda.synchronize()
          decode_latency.append(time.perf_counter() - start_ts)
          current_input_ids = next_token_id
      peak_alloc = torch.cuda.memory.max_memory_allocated()  # peak tensor allocation
      peak_reserved = torch.cuda.memory.max_memory_reserved() # peak allocator reservation
      print(f"seqlen = {seqlen}")
      print(f"budget = {args.k}")
      print(f"peak_alloc  = {peak_alloc / 1024**2:.2f} MiB")
      #print(f"peak_reserved = {peak_reserved / 1024**2:.2f} MiB")  
      append_vram_csv(
        args=args,
        algorithm="sparq",
        model_type=get_model_type(args.model_name, args.model_path),
        input_length=seqlen,
        output_length=max_new_tokens,
        peak_alloc=peak_alloc,
        peak_reserved=peak_reserved,
      )



      del outputs
      del past_key_values
      del input_ids
      del input_ids_full
      del current_input_ids
      del next_token_id

      import gc
      
      gc.collect()
      torch.cuda.empty_cache()
      torch.cuda.ipc_collect()








# ---------- Main inference logic ----------

def latency(
  args,
  model,
  tokenizer,
  data,
  task_name: str,
  model_name: str,
  max_new_tokens: int,
  prompt_format: str,
  seqlen:int,
  device: torch.device,
  out_path: str,
):
  for _ in range(4):
    for json_obj in tqdm(data, desc=f"Task={task_name}"):
      prompt = prompt_format.format(**json_obj)

      input_ids_full = tokenizer(prompt, return_tensors="pt").input_ids

      input_ids = input_ids_full[:, :seqlen].to(model.device)
      
      decode_latency = []
      past_key_values = None
      current_input_ids = input_ids

      import time

      with torch.no_grad():
        for i in range(max_new_tokens):
          torch.cuda.synchronize()
          start_ts = time.perf_counter()
          
          forward_kwargs = {
            "input_ids": current_input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
          }

          forward_kwargs["num_logits_to_keep"] = 1
          
          outputs = model(**forward_kwargs)
          # )


          past_key_values = outputs.past_key_values
          next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
          
          torch.cuda.synchronize()
          decode_latency.append(time.perf_counter() - start_ts)
          current_input_ids = next_token_id

      del outputs
      del past_key_values
      del input_ids
      del input_ids_full
      del current_input_ids
      del next_token_id

      import gc
      
      gc.collect()
      torch.cuda.empty_cache()
      torch.cuda.ipc_collect()


      ttft = decode_latency[0]
      tpot = np.mean(decode_latency[1:]) if len(decode_latency) > 1 else 0
      latency = sum(decode_latency)
      decode_time = sum(decode_latency[1:]) if len(decode_latency) > 1 else 0
      save_data = {
        "in_len": seqlen,
        "budget":args.k,
        "ttft": ttft,
        "tpot(ms)": tpot*1000,
        "latency": latency,
        "decode_time":decode_time,
        "decode_latency": decode_latency
      }
      output_path=out_path
      with open(output_path, "a", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False)
        f.write("\n")


def recall(
  model,
  tokenizer,
  data,
  task_name: str,
  model_name: str,
  max_new_tokens: int,
  prompt_format: str,
  seqlen:int,
  device: torch.device,
  out_path: str,
):

  for json_obj in tqdm(data, desc=f"Task={task_name}"):
    prompt = prompt_format.format(**json_obj)




    inputs = tokenizer(
      prompt,
      truncation=True,
      max_length=seqlen,
      return_tensors="pt"
    ).to(device)
    context_len = inputs.input_ids.shape[-1]
    
    assert torch.cuda.is_available()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()

    output_ids = model.generate(
      **inputs,
      max_new_tokens=max_new_tokens,
      num_beams=1,
      do_sample=False,
      temperature=1.0,
      top_p=1.0,      # no top-p truncation (1.0 = keep all)
      top_k=0,
      pad_token_id=tokenizer.eos_token_id,
    )[0]

    end_ev.record()
    torch.cuda.synchronize()
    ms = start_ev.elapsed_time(end_ev) # ms
  
    gen_ids = output_ids[context_len:]
    pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    new_token_num = len(gen_ids)

  


def main():
  args = parse_args()
  seed_everything(42)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model_name_for_prompt = args.model_name if args.model_name else args.model_path

  task = args.task
  prompt_format = "{prompt}"
  max_new_tokens = args.max_new_tokens

  seqlen = args.seqlen
  if args.type=="TTFT":
    max_new_tokens=1

  if "Llama" in args.model_name:
    modelName="Llama"
  elif "Qwen" in args.model_name:
    modelName="Qwen"
  
  prompt_file_stem = Path(args.dataset_path).stem
  suffix = f"{modelName}-{prompt_file_stem}-{args.name}-k{args.k}-local{args.local_k}-r{args.rank}"
  
  pred_file = args.output_dir/ f'{suffix}.jsonl'
  pred_file.parent.mkdir(parents=True, exist_ok=True)
  out_path = pred_file

  model, tokenizer = load_model_and_tokenizer(args.model_path, device,args,out_path)

  prompt_text = load_prompt_from_file(args.dataset_path)
  data = [{
    "prompt": prompt_text,
    "source_file": args.dataset_path,
  }]


  VRAM(
      args=args,
      model=model,
      tokenizer=tokenizer,
      data=data,
      task_name=task,
      model_name=model_name_for_prompt,
      max_new_tokens=max_new_tokens,
      prompt_format=prompt_format,
      seqlen=seqlen,
      device=device,
      out_path=out_path,
    )





  #   )
  #   )


if __name__ == "__main__":
  main()
