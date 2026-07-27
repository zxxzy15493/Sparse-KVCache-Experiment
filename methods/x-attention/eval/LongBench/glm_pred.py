import os
import json
import math
import random
import argparse
import types
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from xattn.src.Xattention import Xattention_prefill
from xattn.src.Flexprefill import Flexprefill_prefill
from xattn.src.Minference import Minference_prefill


try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


#  glm  threshold 
#  qwen_ratio.py glm_ratio.py  max 
try:
    from glm_ratio import max as glm_max
except ImportError:
    glm_max = None

OUT_PATH = None

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--task", type=str, help="task name", required=True)
    parser.add_argument("--method", type=str, default="full")
    parser.add_argument(
        "--longbench_dir",
        type=str,
        default="",
        help="Path to local LongBench data directory",
    )
    return parser.parse_args(args)


# chat prompt  qwen 
# GLM  apply_chat_template
def build_chat(tokenizer, prompt, model_name):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
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


def get_glm_threshold(layer_idx: int):
    if glm_max is None:
        raise ValueError(
            "args.method='xattn'  glm_ratio.py max threshold"
        )

    layer_value = glm_max[layer_idx]
    if isinstance(layer_value, torch.Tensor):
        return layer_value.detach().cpu().float()
    return torch.tensor(layer_value, dtype=torch.float32)


@torch.no_grad()
def glm_core_attention_forward(self, query_layer, key_layer, value_layer, attention_mask):
    # ChatGLM :
    # query_layer/key_layer/value_layer: [batch, num_heads, q_len_or_k_len, head_dim]
    bsz, num_heads, q_len, head_dim = query_layer.shape
    k_len = key_layer.shape[2]

    is_prefill = q_len == k_len

    if is_prefill:
        if self.method == "xattn":
            threshold = self.threshold
            if isinstance(threshold, torch.Tensor):
                threshold = threshold.to(key_layer.device, dtype=key_layer.dtype)
            attn_output = Xattention_prefill(
                query_layer,
                key_layer,
                value_layer,
                norm=1,
                stride=16,
                threshold=threshold,
                use_triton=True,
                keep_sink=True,
                keep_recent=True,
            )
        elif self.method == "flex":
            attn_output = Flexprefill_prefill(
                query_layer.transpose(1, 2),
                key_layer.transpose(1, 2),
                value_layer.transpose(1, 2),
                gamma=0.9,
                tau=0.1,
            ).transpose(1, 2)
        elif self.method == "minference":
            attn_output = Minference_prefill(query_layer, key_layer, value_layer)
        elif self.method == "full":
            if FLASH_ATTN_AVAILABLE:
                attn_output = flash_attn_func(
                    query_layer.transpose(1, 2),
                    key_layer.transpose(1, 2),
                    value_layer.transpose(1, 2),
                    causal=True,
                ).transpose(1, 2)
            else:
                attn_output = F.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=None,
                    is_causal=True,
                )
        else:
            raise ValueError(f"Unknown method: {self.method}")
    else:
        # decode  dense attention
        #  kv cache / rope SelfAttention.forward 
        attn_weights = torch.matmul(query_layer, key_layer.transpose(2, 3)) / math.sqrt(head_dim)

        # ChatGLM  attention_mask  bool maskTrue  mask
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask, float("-inf"))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_layer.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout.p, training=self.training)
        attn_output = torch.matmul(attn_weights, value_layer)

    expected_shape = (bsz, num_heads, q_len, head_dim)
    if attn_output.size() != expected_shape:
        raise ValueError(
            f"`attn_output` should be of size {expected_shape}, but is {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size_per_partition)
    return attn_output

@torch.no_grad()
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
):
    preds = []
    pbar = tqdm(data)

    for idx, json_obj in enumerate(pbar):
        prompt = prompt_format.format(**json_obj)

        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                tokenized_prompt[-half:], skip_special_tokens=True
            )

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)

        inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
        pbar.set_description(f"Generating for {idx}, len = {inputs.input_ids.shape[-1]}")

        with torch.no_grad():
            #  ChatGLM Qwen/Llama  past_key_values
            #  position_ids 
            model_inputs = model.prepare_inputs_for_generation(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                past_key_values=None,
                use_cache=True,
                is_first_forward=True,
            )
            outputs = model(**model_inputs)
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_content = [pred_token_idx.item()]

            all_input_ids = torch.cat([inputs.input_ids, pred_token_idx], dim=-1)
            all_attention_mask = torch.cat(
                [inputs.attention_mask, inputs.attention_mask.new_ones((inputs.attention_mask.shape[0], 1))],
                dim=-1,
            )

            for _ in range(max_gen - 1):
                model_inputs = model.prepare_inputs_for_generation(
                    input_ids=all_input_ids,
                    attention_mask=all_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    is_first_forward=False,
                )
                outputs = model(**model_inputs)
                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_content.append(pred_token_idx.item())

                if pred_token_idx.item() in eos_token_ids:
                    break

                all_input_ids = torch.cat([all_input_ids, pred_token_idx], dim=-1)
                all_attention_mask = torch.cat(
                    [all_attention_mask, all_attention_mask.new_ones((all_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        pred = post_process(pred, model_name)

        record = {
            "pred": pred,
            "answers": json_obj["answers"],
            "all_classes": json_obj["all_classes"],
            "length": json_obj["length"],
        }
        preds.append(record)

        Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
    return preds

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name):
    #  GLM-4-9B-Chat-1Mremote code 
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    generation_config = GenerationConfig.from_pretrained(path, trust_remote_code=True)
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]
    model = model.eval()
    return model, tokenizer, eos_token_ids

def patch_glm_attention(model, method):
    patched = 0
    for name, module in model.named_modules():
        if name.endswith("core_attention"):
            module.method = method
            if method == "xattn":
                # ChatGLM  layer_number  1 ratio  0 
                layer_idx = int(module.layer_number) - 1
                module.threshold = get_glm_threshold(layer_idx)
            module.forward = types.MethodType(glm_core_attention_forward, module)
            patched += 1

    if patched == 0:
        raise RuntimeError(" core_attention  ChatGLM-4")
    print(f"[info] patched {patched} core_attention modules")


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    model2path = json.load(open("eval/LongBench/config/model2path.json", "r"))
    model2maxlen = json.load(open("eval/LongBench/config/model2maxlen.json", "r"))
    dataset2prompt = json.load(open("eval/LongBench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("eval/LongBench/config/dataset2maxlen.json", "r"))

    model_name = args.model
    model, tokenizer, eos_token_ids = load_model_and_tokenizer(model2path[model_name], model_name)

    patch_glm_attention(model, args.method)

    max_length = model2maxlen[model_name]
    if args.e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        datasets = [args.task]

    if not os.path.exists("eval/LongBench/pred"):
        os.makedirs("eval/LongBench/pred")
    if not os.path.exists("eval/LongBench/pred_e"):
        os.makedirs("eval/LongBench/pred_e")

    for dataset in datasets:
        local_file = os.path.join(args.longbench_dir, f"{dataset}.jsonl")
        if not os.path.exists(local_file):
            raise FileNotFoundError(f"Local LongBench file not found: {local_file}")

        data = load_dataset("json", data_files={"test": local_file})["test"]

        model_out_dir = f"eval/LongBench/pred/{model_name}"
        os.makedirs(model_out_dir, exist_ok=True)

        if args.method == "full":
            OUT_PATH = f"{model_out_dir}/{dataset}-full.jsonl"
        elif args.method == "xattn":
            OUT_PATH = f"{model_out_dir}/{dataset}-xattn-stride=16.jsonl"
        elif args.method == "flex":
            OUT_PATH = f"{model_out_dir}/{dataset}-flex.jsonl"
        elif args.method == "minference":
            OUT_PATH = f"{model_out_dir}/{dataset}-minference.jsonl"
        else:
            OUT_PATH = f"{model_out_dir}/{dataset}-{args.method}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        _ = get_pred(
            model,
            tokenizer,
            eos_token_ids,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            model_name,
        )
