import os
from datasets import load_dataset
import torch
import json
from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
  GenerationConfig,
)
from tqdm import tqdm
import numpy as np
import random
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
  repeat_kv,
  apply_rotary_pos_emb,
  nn,
)
from pathlib import Path
import math
from xattn.src.Xattention_recall import Xattention_prefill
from flash_attn import flash_attn_func
import types
from ratio import max_ratio, max


ALGO_ROOT = Path(".")


def parse_args(args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
  parser.add_argument("--model_name", type=str, default=None)
  parser.add_argument("--task", type=str, help="task name", required=True)
  parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
  parser.add_argument("--method", type=str, default="xattn", choices=["xattn", "full"])
  parser.add_argument("--stride", type=int, default=16)
  parser.add_argument("--budget", type=float, default=0.9)
  parser.add_argument("--type", type=str, default="")
  parser.add_argument("--save_path", type=str, default=" ")
  parser.add_argument("--longbench_dir", type=str, default="../../benchmarks/Longbench_recall")
  parser.add_argument("--max_context_len", type=int, default=131072)
  parser.add_argument("--max_new_tokens", type=int, default=128)
  parser.add_argument("--seed", type=int, default=42)
  return parser.parse_args(args)


def build_chat(tokenizer, prompt, model_name):
  prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                      add_generation_prompt=True, tokenize=False)
  return prompt


def post_process(response, model_name):
  if "xgen" in model_name:
    response = response.strip().replace("Assistant:", "")
  elif "internlm" in model_name:
    response = response.split("<eoa>")[0]
  elif "llama-3" in model_name.lower():
    response = (
      response.split(".assistant")[0]
      .split("\n\nQuestion")[0]
      .split("</s>")[0]
      .strip()
    )
  elif "Llama-2-7B-32K-Instruct" in model_name:
    response = (
      response.split("(Document")[0]
      .split("\n\nQuestion")[0]
      .split("\n\nAnswer")[0]
      .split("(Passage")[0]
      .strip()
    )
  return response


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
  ] = None,
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
      if "Llama" in self.model_name:
        modelName = "Llama"
      elif "Qwen" in self.model_name:
        modelName = "Qwen"
      layer_id = int(getattr(self, "layer_idx", -1))
      attn_output = Xattention_prefill(
        query_states,
        key_states,
        value_states,
        norm=1,
        stride=self.xattn_stride,
        type=self.xtype,
        task=self.task,
        save_path=self.save_path,
        model_name=modelName,
        layer_id=layer_id,
        threshold=threshold,
        use_triton=True,
        keep_sink=True,
        keep_recent=True,
      )
    elif self.method == "full":
      attn_output = flash_attn_func(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        causal=True,
      ).transpose(1, 2)
  else:
    attn_weights = torch.matmul(
      query_states, key_states.transpose(2, 3)
    ) / math.sqrt(self.head_dim)

    if attention_mask is not None:
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


def get_pred(
  model,
  tokenizer,
  eos_token_ids,
  data,
  max_length,
  max_gen,
  prompt_format,
  dataset,
  model_name,
  out_path,
):
  preds = []
  pbar = tqdm(data)
  for idx, json_obj in enumerate(pbar):
    prompt = prompt_format.format(**json_obj)
    tokenized_prompt = tokenizer(
      prompt, truncation=False, return_tensors="pt"
    ).input_ids[0]
    if len(tokenized_prompt) > max_length:
      half = int(max_length / 2)
      prompt = tokenizer.decode(
        tokenized_prompt[:half], skip_special_tokens=True
      ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    if dataset not in [
      "trec",
      "triviaqa",
      "samsum",
      "lsht",
      "lcc",
      "repobench-p",
    ]:
      prompt = build_chat(tokenizer, prompt, model_name)

    input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
    pbar.set_description(f"Generating for {idx}, len = {input.input_ids.shape[-1]}")
    with torch.no_grad():
      output = model(
        input_ids=input.input_ids,
        past_key_values=None,
        use_cache=True,
        num_logits_to_keep=1,
      )
      past_key_values = output.past_key_values
      pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
      generated_content = [pred_token_idx.item()]
      for _ in range(max_gen - 1):
        outputs = model(
          input_ids=pred_token_idx,
          past_key_values=past_key_values,
          use_cache=True,
          num_logits_to_keep=1,
        )

        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_content += [pred_token_idx.item()]
        if pred_token_idx.item() in eos_token_ids:
          break

    pred = tokenizer.decode(generated_content, skip_special_tokens=True)
    pred = post_process(pred, model_name)

    preds.append(
      {
        "pred": pred,
        "answers": json_obj["answers"],
        "all_classes": json_obj["all_classes"],
        "length": json_obj["length"],
      }
    )
    record = {
      "pred": pred,
      "answers": json_obj["answers"],
      "all_classes": json_obj["all_classes"],
      "length": json_obj["length"],
    }
  return preds


def seed_everything(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_path):
  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
  model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
    attn_implementation="flash_attention_2",
  )

  generation_config = GenerationConfig.from_pretrained(model_path)
  eos_token_ids = generation_config.eos_token_id
  if not isinstance(eos_token_ids, list):
    eos_token_ids = [eos_token_ids]

  model = model.eval()
  return model, tokenizer, eos_token_ids


if __name__ == "__main__":
  args = parse_args()
  seed_everything(args.seed)

  if args.model_name is None:
    args.model_name = os.path.basename(args.model_path.rstrip("/"))

  model, tokenizer, eos_token_ids = load_model_and_tokenizer(args.model_path)

  if args.budget == 0.9:
    threshold_data = max
  elif args.budget == 0.8:
    from llama0_8_ratio import max as max_08
    threshold_data = max_08
  elif args.budget == 0.85:
    from llama0_85_ratio import max as max_085
    threshold_data = max_085
  elif args.budget == 0.95:
    from llama0_95_ratio import max as max_095
    threshold_data = max_095
  else:
    threshold_data = max

  for name, module in model.named_modules():
    if name.split(".")[-1] == "self_attn":
      layer_idx = int(name.split(".")[2])
      module.method = args.method
      module.model_name = args.model_name
      module.xattn_stride = args.stride
      module.xtype = args.type
      module.task = args.task
      module.save_path = args.save_path
      if args.method == "xattn":
        module.threshold = torch.tensor(threshold_data[layer_idx])
      module.forward = types.MethodType(new_attention_forward, module)

  max_length = args.max_context_len

  dataset2prompt = json.load(open(ALGO_ROOT / "eval/recall/LongBench/config/dataset2prompt.json", "r"))
  dataset2maxlen = json.load(open(ALGO_ROOT / "eval/recall/LongBench/config/dataset2maxlen.json", "r"))

  if args.e:
    datasets = [
      "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
      "gov_report", "multi_news", "trec", "triviaqa",
      "samsum", "passage_count", "passage_retrieval_en",
      "lcc", "repobench-p",
    ]
  else:
    datasets = [args.task]

  for dataset in datasets:
    local_file = os.path.join(args.longbench_dir, f"{dataset}.jsonl")
    if not os.path.exists(local_file):
      raise FileNotFoundError(f"Local LongBench file not found: {local_file}")

    data = load_dataset("json", data_files={"test": local_file})["test"]

    prompt_format = dataset2prompt[dataset]
    max_gen = dataset2maxlen[dataset]
    preds = get_pred(
      model, tokenizer, eos_token_ids, data,
      max_length, max_gen, prompt_format, dataset,
      args.model_name, None,
    )
