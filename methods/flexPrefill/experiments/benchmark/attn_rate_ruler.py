

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
import sys
from tqdm.auto import tqdm
import yaml
import importlib
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from pathlib import Path
from flex_prefill import patch_model
from utils import (
  seed_everything,
  get_args,
  str_to_dict,
)

RULER_DIR = Path("experiments/benchmark/recall/ruler")
SCRIPT_DIR = Path(__file__).resolve().parent
FLEX_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = SCRIPT_DIR.parents[3]
RULER_DATA_ROOT = REPO_ROOT / "benchmarks" / "Ruler_recall"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

accelerator = Accelerator()


SEQ_LENGTHS = ["65536"]




TASKS = [
  "niah_single_3",
  "vt",
  "fwe",
]

TASK_TO_MAX_NEW_TOKNES = {
  "niah_single_1": 256,
  "niah_single_2": 256,
  "niah_single_3": 256,
  "niah_multikey_1": 256,
  "niah_multikey_2": 256,
  "niah_multikey_3": 256,
  "niah_multivalue": 256,
  "niah_multiquery": 256,
  "vt": 256,
  "cwe": 256,
  "fwe": 256,
  "qa_1": 256,
  "qa_2": 256,
}


class ListDataset(Dataset):
  def __init__(self, data_list):
    self.data_list = data_list

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, idx):
    return self.data_list[idx]



def read_manifest(file_path):
  file_path = str(file_path)
  data = []
  if not os.path.exists(file_path):
    return data
  with open(file_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        data.append(json.loads(line))
      except json.JSONDecodeError as e:
        raise ValueError(
          f"JSON decode error in {file_path} at line {line_num}: {e}"
        )
  return data

def get_dataloader(data_list):
  data_loader = DataLoader(ListDataset(data_list), batch_size=1, shuffle=False)
  return data_loader


def get_tasks(task_str: str):
  if task_str == "ruler":
    tasks = []
    for t in TASKS:
      for s in SEQ_LENGTHS:
        tasks.append((t, s))
    return tasks
  elif task_str.startswith("ruler"):
    tasks = []
    length = task_str.split(",")[-1]
    for t in TASKS:
      tasks.append((t, length))
    return tasks
  else:
    task, length = task_str.split(",")
    return [(task, length)]


def main():
  args = get_args()
  seed_everything(args.seed)
  model_name = args.model.strip("/").split("/")[-1]

  tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    _attn_implementation="flash_attention_2",
    trust_remote_code=True,
  )
  model.config.task=args.task

  flex_prefill_config = {
    "block_size": 128,
    "flex_prefill_gamma": args.p,
    "flex_prefill_tau": 0.1,
    "flex_prefill_min_budget": 512,
    "flex_prefill_max_budget": None,
  }

  patch_model(model, "flex_prefill", flex_prefill_config)


  model.config.type = "recall"





  curr_folder = os.path.dirname(os.path.abspath(__file__))
  try:
    sys.path.append(str(RULER_DIR))
    module = importlib.import_module("data.synthetic.constants")
  except ImportError:
    print("Module data.synthetic.constants not found.")
    return

  tasks_base = module.TASKS
  with open(
    RULER_DIR / "synthetic.yaml",
    "r",
  ) as f:
    tasks_customized = yaml.safe_load(f)

  dataloaders = []
  all_tasks = get_tasks(args.task)
  for task, length in all_tasks:
    if task not in tasks_customized:
      raise ValueError(f"{task} is not found in config_tasks.yaml")
    config = tasks_customized.get(task)
    config.update(tasks_base[config["task"]])

    if "llama" in model_name.lower():
      save_dir = os.path.join(
        FLEX_ROOT / "efficiency" / "attn_rate-results" / "ruler" / "llama",
        str(args.p))
      model.config.save_path = os.path.join(save_dir, f"{task}.jsonl")
      
      # )
      task_file = os.path.join(
        str(RULER_DATA_ROOT / "llama-3.1-8b" / "synthetic"),
        length,
        "data",
        task,
        "validation.jsonl",
      )
      
    elif "qwen" in model_name.lower():
      save_dir = os.path.join(
        FLEX_ROOT / "efficiency" / "attn_rate-results" / "ruler" / "qwen",
        str(args.p),)
      model.config.save_path = os.path.join(save_dir, f"{task}.jsonl")
      task_file = os.path.join(
        str(RULER_DATA_ROOT / "qwen-2.5-7b-1m" / "synthetic"),
        length,
        "data",
        task,
        "validation.jsonl",
      )
    else:
      raise ValueError(f"Unknown model type in model_name: {model_name}")


    data = read_manifest(task_file)

    dataloaders.append(get_dataloader(data))

  for loader, (task, length) in zip(dataloaders, all_tasks):
    loader = accelerator.prepare_data_loader(loader)
    model.config.task=task
  
    def get_output(index, input, outputs, others, truncation, length_val):
      try:
        if args.chat:
          try:
            input_ids = tokenizer.apply_chat_template(
              [{"role": "user", "content": input}],
              add_generation_prompt=True,
              return_tensors="pt",
            ).to(model.device)
          except Exception:
            encoded = tokenizer(
              input,
              return_tensors="pt",
            )
            input_ids = encoded.input_ids.to(model.device)
            attention_mask = encoded.attention_mask.to(model.device)
        else:
          encoded = tokenizer(
            input,
            return_tensors="pt",
          )
          input_ids = encoded.input_ids.to(model.device)
          attention_mask = encoded.attention_mask.to(model.device)

        do_sample = False if args.top_p <= 0 else True
        generation_config = dict(
          do_sample=do_sample,
          max_new_tokens=TASK_TO_MAX_NEW_TOKNES[task],
          pad_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
          generation_config["top_p"] = args.top_p
          generation_config["temperature"] = args.temperature


        if "attention_mask" in locals():


          output = model.generate(
            input_ids, attention_mask=attention_mask, **generation_config
          )
        else:
          output = model.generate(input_ids, **generation_config)






        generated_text = tokenizer.decode(
          output[0][input_ids.shape[1] :], skip_special_tokens=True
        )

        if generated_text.startswith(input):
          generated_text = generated_text[len(input) :]
        if generated_text.find("</s>") > 0:
          generated_text = generated_text[: generated_text.find("</s>")]
        pred = {"text": [generated_text]}


        if len(pred["text"]) > 0:
          return {
            "index": int(index),
            "pred": pred["text"][0],
            "input": input,
            "outputs": outputs,
            "others": others,
            "truncation": truncation,
            "length": length_val,
          }
        else:
          return None
      except Exception as e:

        print(
          f"[WARN] get_output failed at task={task}, length={length}, index={index}, err={e}"
        )
        return None

    if len(loader) == 0:
      print(f"[INFO] No samples to run for task={task}, length={length}")
      continue


    pbar = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)


    for idx, data_point in enumerate(loader):

      if data_point is None:
        print(
          f"[WARN] data_point is None at task={task}, length={length}, idx={idx}"
        )
        continue

      if not isinstance(data_point, dict):
        print(
          f"[WARN] Unexpected data_point type at task={task}, length={length}, idx={idx}, type={type(data_point)}"
        )
        print("data_point =", data_point)
        continue

      if "index" not in data_point:
        print(
          f"[WARN] 'index' not in data_point at task={task}, length={length}, idx={idx}"
        )
        print("data_point =", data_point)
        continue

      if data_point["index"] is None or len(data_point["index"]) == 0:
        print(
          f"[WARN] data_point['index'] is None/empty at task={task}, length={length}, idx={idx}"
        )
        print("data_point =", data_point)
        continue

      invalid_sample = False
      for key in ["input", "outputs"]:
        if key not in data_point or data_point[key] is None or len(data_point[key]) == 0:
          print(
            f"[WARN] key '{key}' missing/empty in data_point at task={task}, length={length}, idx={idx}"
          )
          print("data_point =", data_point)
          invalid_sample = True
          break
      if invalid_sample:
        continue

      outputs = [
        item[0] if isinstance(item, (list, tuple)) and len(item) > 0 else item
        for item in data_point["outputs"]
      ]

      model.config.sample_id = f"{task}_{data_point['index'][0]}"

      output = get_output(
        data_point["index"][0],
        data_point["input"][0],
        outputs,
        data_point.get("others", [{}])[0],
        data_point.get("truncation", [-1])[0],
        int(data_point.get("length", [-1])[0]),
      )

      pbar.set_description(desc=f"task {task}, len {length}")
      pbar.update(1)

    accelerator.wait_for_everyone()

  all_length = set([length for _, length in all_tasks])

  #     )


if __name__ == "__main__":
  main()
