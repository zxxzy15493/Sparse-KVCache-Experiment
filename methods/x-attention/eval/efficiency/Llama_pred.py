


import os
import json
import argparse
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset
from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
  GenerationConfig,
)
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
  repeat_kv,
  apply_rotary_pos_emb,
  nn,
)
import math
from xattn.src.Xattention import Xattention_prefill
from flash_attn import flash_attn_func
import types





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






def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, required=True, default="meta-llama/Llama-3.1-8B-Instruct", help="HF repo name or local model directory")
  
  



  
  
  parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length")
  parser.add_argument("--task", type=str, default="file_prompt", help="Task name for logging")
  parser.add_argument("--dataset_path", type=str, required=True, help="Path to input text file used as prompt")
  parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate; TTFT mode forces this to 1")
  
  parser.add_argument("--output_dir", type=Path, default="pred", help="Output directory for predictions, auto-creates subdir by model name")
  parser.add_argument("--model_name", type=str, default="", help="Model name for chat template and output directory name")
  parser.add_argument("--p", type=float, default=0.9, help="")

  
  


  parser.add_argument("--stride", type=int, help="Stride for anti-diagonal computation")

  parser.add_argument("--model",type=str,default=None,)
  parser.add_argument("--method", type=str, required=True, help="Whether to reallocate to mean value")
  parser.add_argument("--type", type=str, required=True, help="Metric type to compute")


  return parser.parse_args()




def seed_everything(seed: int = 42):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def new_attention_forward(
  self,
  hidden_states: torch.Tensor,
  attention_mask: Optional[torch.Tensor] = None,
  position_ids: Optional[torch.LongTensor] = None,
  past_key_value: Optional[Cache] = None,
  output_attentions: bool = False,
  use_cache: bool = False,
  cache_position: Optional[torch.LongTensor] = None,
  position_embeddings: Optional[
    Tuple[torch.Tensor, torch.Tensor]
  ] = None, # will become mandatory in v4.46
  **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
  bsz, q_len, _ = hidden_states.size()

  query_states = self.q_proj(hidden_states)
  key_states = self.k_proj(hidden_states)
  value_states = self.v_proj(hidden_states)

  query_states = query_states.view(
    bsz, q_len, self.num_heads, self.head_dim
  ).transpose(1, 2)
  key_states = key_states.view(
    bsz, q_len, self.num_key_value_heads, self.head_dim
  ).transpose(1, 2)
  value_states = value_states.view(
    bsz, q_len, self.num_key_value_heads, self.head_dim
  ).transpose(1, 2)

  if position_embeddings is None:
    cos, sin = self.rotary_emb(value_states, position_ids)
  else:
    cos, sin = position_embeddings
  query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

  if past_key_value is not None:
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_value.update(
      key_states, value_states, self.layer_idx, cache_kwargs
    )
    
  key_states = repeat_kv(key_states, self.num_key_value_groups)
  value_states = repeat_kv(value_states, self.num_key_value_groups)
  
  if key_states.shape[2] == query_states.shape[2]:
    if self.method == "xattn":
      self.threshold = self.threshold.to(key_states.device)
      threshold = self.threshold
      stride=self.xattn_stride
      layer_id = int(getattr(self, "layer_idx", -1))
      if "Llama" in self.model_name:
        modelName="Llama"
      elif "Qwen" in self.model_name:
        modelName="Qwen"

      attn_output = Xattention_prefill(
        query_states,
        key_states,
        value_states,
        type=self.type,
        model_name= modelName,
        layer_id=layer_id,
        norm=1,
        stride=stride,
        threshold=threshold,
        use_triton=True,
        keep_sink=True,
        keep_recent=True,
      )
    #   )
    elif self.method == "full":
      attn_output = flash_attn_func(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        causal=True,
      ).transpose(1, 2)
  else:
    ########################################################################################################################
    attn_weights = torch.matmul(
      query_states, key_states.transpose(2, 3)
    ) / math.sqrt(self.head_dim)

    if attention_mask is not None: # no matter the length, we just slice it
      causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
      attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(
      attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    attn_weights = nn.functional.dropout(
      attn_weights, p=self.attention_dropout, training=self.training
    )
    attn_output = torch.matmul(attn_weights, value_states)

  if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
    raise ValueError(
      f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
      f" {attn_output.size()}"
    )

  attn_output = attn_output.transpose(1, 2).contiguous()

  attn_output = attn_output.reshape(bsz, q_len, -1)

  attn_output = self.o_proj(attn_output)

  if not output_attentions:
    attn_weights = None

  return attn_output, attn_weights, past_key_value


def build_chat_prompt(prompt: str, model_name: str) -> str:
  """
  Determine chat prompt template based on model name
  Adjust as needed for your model
  """
  name = model_name.lower()


  if "llama-2" in name or "llama2" in name:
    return f"[INST] {prompt} [/INST]"


  if "xgen" in name:
    header = (
      "A chat between a curious human and an artificial intelligence assistant. "
      "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
    )
    return header + f"### Human: {prompt}\n### Assistant:"


  if "internlm" in name:
    return f"<|User|>:{prompt}<eoh>\n<|Bot|>:"


  return prompt


def load_model_and_tokenizer(model_path: str, device: torch.device,args,out_path):
  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,use_fast=False)
  model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto" , 
    attn_implementation="flash_attention_2",
  )



  model.eval()
  return model, tokenizer




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

      max_model_len = getattr(model.config, "max_position_embeddings", None)
      effective_seqlen = min(seqlen, max_model_len) if max_model_len else seqlen
      if effective_seqlen != seqlen:
        print(
          f"[WARN] requested seqlen={seqlen} exceeds model max_position_embeddings="
          f"{max_model_len}; using {effective_seqlen}."
        )

      input_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=effective_seqlen,
        return_tensors="pt",
      ).input_ids.to(model.device)
      context_len = input_ids.shape[-1]
      
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
      ttft = decode_latency[0]
      tpot = np.mean(decode_latency[1:]) if len(decode_latency) > 1 else 0
      latency = sum(decode_latency)
      decode_time = sum(decode_latency[1:]) if len(decode_latency) > 1 else 0

      del outputs
      del past_key_values
      del input_ids
      del current_input_ids
      del next_token_id

      import gc
      
      gc.collect()
      torch.cuda.empty_cache()
      torch.cuda.ipc_collect()

      save_data = {
        "in_len": context_len,
        "requested_in_len": seqlen,
        "budget":args.p,
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





def ttft(
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
      max_new_tokens=1,
      num_beams=1,

      do_sample=False,
      temperature=1.0,
      top_p=1.0,
      top_k=0,
      pad_token_id=tokenizer.eos_token_id,
    )[0]

    end_ev.record()
    torch.cuda.synchronize()
    ms = start_ev.elapsed_time(end_ev)

    gen_ids = output_ids[context_len:]
    pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    record = {
      "ttft":ms/1000.0,
      "pred": pred,
      
    }

    with open(out_path, "a", encoding="utf-8") as f:
      json.dump(record, f, ensure_ascii=False)
      f.write("\n")

def throughput(
  model,
  tokenizer,
  data,
  task_name: str,
  model_name: str,
  seqlen:int,
  max_new_tokens: int,
  prompt_format: str,
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
      top_p=1.0,
      top_k=0,
      pad_token_id=tokenizer.eos_token_id,
    )[0]

    end_ev.record()
    torch.cuda.synchronize()
    ms = start_ev.elapsed_time(end_ev)
  

    gen_ids = output_ids[context_len:]
    pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    new_token_num = len(gen_ids)

    throughput = new_token_num / (ms / 1000.0)
    record = {
      "throughput":throughput,
      "token_num":new_token_num,
      "total_time":ms/1000.0,
      "pred": pred,
      
    }

    with open(out_path, "a", encoding="utf-8") as f:
      json.dump(record, f, ensure_ascii=False)
      f.write("\n")


def recall(
  model,
  tokenizer,
  data,
  task_name: str,
  model_name: str,
  seqlen: int,
  max_new_tokens: int,
  prompt_format: str,
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
      top_p=1.0,
      top_k=0,
      pad_token_id=tokenizer.eos_token_id,
    )[0]

    end_ev.record()
    torch.cuda.synchronize()
    ms = start_ev.elapsed_time(end_ev)
  

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



  if "Llama" in args.model_name:
    modelName="Llama"
  elif "Qwen" in args.model_name:
    modelName="Qwen"
  
  suffix = f"{modelName}-stride{args.stride}"

  pred_file = args.output_dir/ f'{suffix}.jsonl'
  pred_file.parent.mkdir(parents=True, exist_ok=True)
  out_path = pred_file


  model, tokenizer = load_model_and_tokenizer(args.model_path, device,args,out_path)


  if args.p == 0.9:
    from llama0_90_ratio import max
  elif args.p == 0.8:
    from llama0_8_ratio import max
  elif args.p == 0.85:
    from llama0_85_ratio import max
  elif args.p == 0.95:
    from llama0_95_ratio import max

  def build_head_threshold(layer_threshold, head_num: int) -> torch.Tensor:
    if isinstance(layer_threshold, torch.Tensor):
      values = layer_threshold.detach().cpu().flatten().tolist()
    elif isinstance(layer_threshold, (list, tuple)):
      values = list(layer_threshold)
    else:
      values = [float(layer_threshold)]

    if len(values) >= head_num:
      values = values[:head_num]
    else:
      avg = float(sum(values) / len(values))
      values = values + [avg] * (head_num - len(values))

    return torch.tensor(values, dtype=torch.float32)


  for name, module in model.named_modules():
    if name.split(".")[-1] == "self_attn":
      layer_idx = int(name.split(".")[2])
      module.method = args.method
      module.xattn_stride = args.stride
      module.type=args.type
      module.model_name=args.model_name

      if args.method == "xattn":
        module.threshold = build_head_threshold(max[layer_idx], module.num_heads)
      module.forward = types.MethodType(new_attention_forward, module)


  prompt_text = load_prompt_from_file(args.dataset_path)
  data = [{
    "prompt": prompt_text,
    "source_file": args.dataset_path,
  }]



  latency(
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
  #   )



if __name__ == "__main__":
  main()
