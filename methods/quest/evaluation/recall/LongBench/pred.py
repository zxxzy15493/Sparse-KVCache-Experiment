

import os
from datasets import load_dataset
import torch
import json
from transformers import (
  AutoTokenizer,
  AutoConfig,
  LlamaTokenizer,
  LlamaForCausalLM,
  AutoModelForCausalLM,
)
from tqdm import tqdm
import numpy as np
import random
import argparse
import sys
from pathlib import Path

QUEST_ROOT = Path(".")
PROJECT_ROOT = Path("../..")
for path in (PROJECT_ROOT, QUEST_ROOT):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))
from evaluation.llama import enable_tuple_kv_cache_for_llama 

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, required=True, default="", help="HF repo name or local model directory")
  parser.add_argument("--task", type=str, required=True, help="LongBench task name, e.g. hotpotqa, qasper, gov_report")
  parser.add_argument("--config_path", type=str, required=True, help="Directory with dataset2prompt.json, dataset2maxlen.json, etc.")
  parser.add_argument("--dataset_path", type=str, default=None, help="Local LongBench data directory")
  parser.add_argument("--output_dir", type=str, default="pred", help="Output directory for predictions, auto-creates subdir by model name")
  parser.add_argument("--model_name", type=str, default="", help="Model name for chat template and output directory name")
  parser.add_argument("--e", action="store_true", help="Whether to evaluate LongBench-E (use *_e.jsonl)")
  parser.add_argument("--token_budget", type=int, default=None)
  parser.add_argument("--chunk_size", type=int, default=None)
  parser.add_argument("--quest", action="store_true", help="Enable Quest Attention")
  parser.add_argument("--save_path",type=str, help="")

  return parser.parse_args()




def seed_everything(seed: int = 42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def build_chat(tokenizer, prompt, model_name):
  prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                      add_generation_prompt=True, tokenize=False)
  return prompt 



def load_model_and_tokenizer(model_path: str, model_name,device: torch.device,args):
    
  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    attn_implementation="flash_attention_2"
  )
  model = model.eval()


  if "llama" in args.model_name.lower():
    from methods.quest.evaluation.quest_attention import enable_quest_attention_eval
    print("llama model")
  elif "qwen" in args.model_name.lower():
    from methods.quest.evaluation.quest_qwen_attention import enable_quest_attention_eval
    print("qwen model")
  elif "glm" in args.model_name.lower():
    from evaluation.quest_glm_attention import enable_quest_attention_eval
    print("glm model")


  enable_quest_attention_eval(model, args)


  return model, tokenizer




def run_longbench_pred(
  model,
  tokenizer,
  data,
  task_name: str,
  model_name: str,
  max_length_ctx: int,
  max_new_tokens: int,
  prompt_format: str,
  device: torch.device,
  out_path: str,
):

  if os.path.exists(out_path):
    os.remove(out_path)

  for json_obj in tqdm(data, desc=f"Task={task_name}"):

    prompt = prompt_format.format(**json_obj)


    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt")
    input_ids = tokenized.input_ids[0]


    if len(input_ids) > max_length_ctx:
      half = max_length_ctx // 2
      kept_ids = torch.cat([input_ids[:half], input_ids[-half:]], dim=0)
      prompt = tokenizer.decode(kept_ids, skip_special_tokens=True)


    if task_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
      prompt = build_chat(tokenizer, prompt, model_name)


    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    context_len = inputs.input_ids.shape[-1]
    

    if task_name == "samsum":

      
      output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens, 
        num_beams=1,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_length=context_len + 1,
        eos_token_id=[
          tokenizer.eos_token_id,
          tokenizer.encode("\n", add_special_tokens=False)[-1],
        ],
        pad_token_id=tokenizer.eos_token_id,
      )[0]
    else:
      output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=1,

        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        pad_token_id=tokenizer.eos_token_id,
      )[0]


    gen_ids = output_ids[context_len:]
    pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    record = {
      "pred": pred,
      "answers": json_obj["answers"],
      "all_classes": json_obj["all_classes"],
      "length": json_obj["length"],
    }



def main():
  args = parse_args()
  seed_everything(42)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model_name_for_prompt = args.model_name if args.model_name else args.model_path


  dataset2prompt = json.load(open(os.path.join(args.config_path, "dataset2prompt.json"), "r"))
  dataset2maxlen = json.load(open(os.path.join(args.config_path, "dataset2maxlen.json"), "r"))


  model2maxlen_path = os.path.join(args.config_path, "model2maxlen.json")
  if os.path.exists(model2maxlen_path):
    model2maxlen = json.load(open(model2maxlen_path, "r"))
    max_length_ctx = model2maxlen.get(model_name_for_prompt, 16384)
  else:
    max_length_ctx = 16384
  task = args.task
  assert task in dataset2prompt, f"{task} not found in dataset2prompt.json"

  prompt_format = dataset2prompt[task]
  max_new_tokens = dataset2maxlen[task]


  model, tokenizer = load_model_and_tokenizer(args.model_path,args.model_name, device,args)


  if args.dataset_path:

    if args.e:
      data_file = os.path.join(args.dataset_path, f"{task}_e.jsonl")
    else:
      data_file = os.path.join(args.dataset_path, f"{task}.jsonl")
    assert os.path.exists(data_file), f"Data file not found: {data_file}"

    dataset_dict = load_dataset("json", data_files={"test": data_file})
    data = dataset_dict["test"]


  model_tag = os.path.basename(model_name_for_prompt.rstrip("/"))
  out_root = os.path.join(args.output_dir + ("_e" if args.e else ""))
  os.makedirs(out_root, exist_ok=True)
  model_dir = os.path.join(out_root, model_tag)
  os.makedirs(model_dir, exist_ok=True)
  suffix = f"budget{args.token_budget}"

  out_path = os.path.join(model_dir, f"{task}-{suffix}.jsonl")


  run_longbench_pred(
    model=model,
    tokenizer=tokenizer,
    data=data,
    task_name=task,
    model_name=model_name_for_prompt,
    max_length_ctx=max_length_ctx,
    max_new_tokens=max_new_tokens,
    prompt_format=prompt_format,
    device=device,
    out_path=out_path,
  )


if __name__ == "__main__":
  main()
