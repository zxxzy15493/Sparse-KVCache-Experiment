#
#
#

"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
  "index" int,
  "input": str,
  "outputs": [str],
}

prediction jsonl: 
{
  "index" int,
  "input": str,
  "outputs": [str],
  "pred": str,
}
"""

import argparse
import json
import yaml
import os
import sys
import threading
import importlib
import math
import time
import torch
from tqdm import tqdm
from pathlib import Path
import traceback
from xattn.src.budget_load_llama_recall import FastPrefillConfig

SERVER_TYPES = (
  'trtllm',
  'vllm',
  'sglang',
  'openai',
  'gemini',
  'hf',
  'mamba',
)


class ServerAction(argparse.Action):
  def __call__(self, parser, namespace, values, option_string=None):
    namespace.server_type = values


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path, required=True, help='path to load the dataset jsonl files')
parser.add_argument("--save_dir", type=Path, required=True, help='path to save the prediction jsonl files')
parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
parser.add_argument("--task", type=str, required=True, help='Options: tasks in benchmark')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--chunk_idx", type=int, default=0, help='index of current split chunk')
parser.add_argument("--chunk_amount", type=int, default=1, help='size of split chunk')

parser.add_argument("--server_type", default='nemo', action=ServerAction, choices=SERVER_TYPES)
parser.add_argument("--server_host", type=str, default='127.0.0.1')
parser.add_argument("--server_port", type=str, default='5000')
parser.add_argument("--ssh_server", type=str)
parser.add_argument("--ssh_key_path", type=str)
parser.add_argument("--model_name_or_path", type=str, default='gpt-3.5-turbo', 
          help='supported models from OpenAI or HF (provide a key or a local path to the checkpoint)')

parser.add_argument("--save_path", type=str)

parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=32)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--stop_words", type=str, default='')
parser.add_argument("--sliding_window_size", type=int)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=1)

parser.add_argument("--threshold", type=float, default=None, help="Threshold for grouping.")
parser.add_argument("--print_detail", action='store_true', default=False, help="Print detailed information. Default is False.")
parser.add_argument("--stride", type=int, default=16, help="Small block size") 
parser.add_argument("--metric", type=str, default="xattn", help="")

parser.add_argument("--p", type=float, default=0.9, help="")


args = parser.parse_args()
args.stop_words = list(filter(None, args.stop_words.split(',')))
if args.server_type == 'hf' or args.server_type == 'gemini':
  args.threads = 1

fastprefillconfig = FastPrefillConfig(
  threshold=args.threshold,
  print_detail=args.print_detail,
  stride = args.stride,
  p = args.p,
  task=args.task,
  metric=args.metric,
  save_path=args.save_path,
)



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

def get_llm(tokens_to_generate):
  if args.server_type == 'trtllm':
    from client_wrappers import TRTLLMClient
    llm = TRTLLMClient(
      server_host=args.server_host,
      server_port=args.server_port,
      ssh_server=args.ssh_server,
      ssh_key_path=args.ssh_key_path,
      temperature=args.temperature,
      top_k=args.top_k,
      top_p=args.top_p,
      random_seed=args.random_seed,
      stop=args.stop_words,
      tokens_to_generate=tokens_to_generate,
      max_attention_window_size=args.sliding_window_size,
    )

  elif args.server_type == 'vllm':
    from client_wrappers import VLLMClient
    llm = VLLMClient(
      server_host=args.server_host,
      server_port=args.server_port,
      ssh_server=args.ssh_server,
      ssh_key_path=args.ssh_key_path,
      temperature=args.temperature,
      top_k=args.top_k,
      top_p=args.top_p,
      random_seed=args.random_seed,
      stop=args.stop_words,
      tokens_to_generate=tokens_to_generate,
    )

  elif args.server_type == 'sglang':
    from client_wrappers import SGLClient
    llm = SGLClient(
      server_host=args.server_host,
      server_port=args.server_port,
      ssh_server=args.ssh_server,
      ssh_key_path=args.ssh_key_path,
      temperature=args.temperature,
      top_k=args.top_k,
      top_p=args.top_p,
      random_seed=args.random_seed,
      stop=args.stop_words,
      tokens_to_generate=tokens_to_generate,
    )
    
  elif args.server_type == 'openai':
    from client_wrappers import OpenAIClient
    llm = OpenAIClient(
      model_name=args.model_name_or_path,
      temperature=args.temperature,
      top_k=args.top_k,
      top_p=args.top_p,
      random_seed=args.random_seed,
      stop=args.stop_words,
      tokens_to_generate=tokens_to_generate,
    )

  elif args.server_type == 'gemini':
    from client_wrappers import GeminiClient
    llm = GeminiClient(
      model_name=args.model_name_or_path,
      temperature=args.temperature,
      top_k=args.top_k,
      top_p=args.top_p,
      random_seed=args.random_seed,
      stop=args.stop_words,
      tokens_to_generate=tokens_to_generate,
    )
    
  elif args.server_type == 'hf':
    from model_wrappers import HuggingFaceModel
    llm = HuggingFaceModel(
      name_or_path=args.model_name_or_path,
      fastprefillconfig=fastprefillconfig,
      do_sample=args.temperature > 0,
      repetition_penalty=1,
      temperature=args.temperature,
      top_k=args.top_k,
      top_p=args.top_p,
      stop=args.stop_words,
      max_new_tokens=tokens_to_generate,
    )
  
  elif args.server_type == 'mamba':
    from model_wrappers import MambaModel
    llm = MambaModel(
      name_or_path=args.model_name_or_path,
      repetition_penalty=1,
      temperature=args.temperature,
      top_k=args.top_k,
      top_p=args.top_p,
      stop=args.stop_words,
      max_new_tokens=tokens_to_generate,
    )
    
  else:
    raise RuntimeError(f'Unsupported server type {args.server_type}')

  return llm


def main():
  start_time = time.time()
  
  curr_folder = os.path.dirname(os.path.abspath(__file__))
  
  try:
    sys.path.append(os.path.dirname(curr_folder))
    module = importlib.import_module(f"data.{args.benchmark}.constants")
  except ImportError:
    print(f"Module data.{args.benchmark}.constants not found.")

  tasks_base = module.TASKS
  with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
    tasks_customized = yaml.safe_load(f)

  if args.task not in tasks_customized:
    raise ValueError(f'{args.task} is not found in config_tasks.yaml')
    
  config = tasks_customized.get(args.task)
  config.update(tasks_base[config['task']])

  task_file = args.data_dir / args.task / f'{args.subset}.jsonl'
  
  print(f'Predict {args.task} \nfrom {task_file}')
  data = read_manifest(task_file)

  llm = get_llm(config['tokens_to_generate'])

  def get_output(idx_list, index_list, input_list, outputs_list, others_list, truncation_list, length_list):
    nonlocal llm

    while True:
      try:
        with torch.no_grad():
          pred_list = llm.process_batch(prompts=input_list)
          break
      except Exception as e:
        traceback.print_exc()

    zipped_iter = zip(pred_list, idx_list, index_list, input_list,
             outputs_list, others_list, truncation_list, length_list)

    for pred, idx, index, input, outputs, others, truncation, length in zipped_iter:
      if isinstance(pred['text'], str):
        pred_text = pred['text']
      elif len(pred['text']) > 0:
        pred_text = pred['text'][0]
      else:
        pred_text = ''

      outputs_parallel[idx] = {
        'index': index,
        'pred': pred_text,
        'input': input,
        'outputs': outputs,
        'others': others,
        'truncation': truncation,
        'length': length,
      }

  threads = []
  outputs_parallel = [{} for _ in range(len(data))]

  batched_data = []
  batch = []
  for idx, data_point in enumerate(data):
    data_point['idx'] = idx

    if len(batch) >= args.batch_size:
      batched_data.append(batch)
      batch = []

    batch.append(data_point)

  if len(batch):
    batched_data.append(batch)

  for batch_idx, batch in tqdm(enumerate(batched_data), total=len(batched_data)):
    idx_list = [data_point['idx'] for data_point in batch]

    thread = threading.Thread(
      target=get_output,
      kwargs=dict(
        idx_list=idx_list,
        index_list=[data_point['index'] for data_point in batch],
        input_list=[data_point['input'] for data_point in batch],
        outputs_list=[data_point['outputs'] for data_point in batch],
        others_list=[data_point.get('others', {}) for data_point in batch],
        truncation_list=[data_point.get('truncation', -1) for data_point in batch],
        length_list=[data_point.get('length', -1) for data_point in batch],
      ),
    )
    thread.start()
    threads.append(thread)

    is_last_batch = (batch_idx == len(batched_data) - 1)

    if (len(threads) == args.threads) or is_last_batch:
      for thread in threads:
        thread.join()
      threads = []

  print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")


if __name__ == '__main__':
  main()
