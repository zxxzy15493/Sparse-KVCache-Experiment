import os
import re
import json
import argparse
import random
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import (
    repeat_kv,
    apply_rotary_pos_emb,
)
import torch.nn as nn
import math
from xattn.src.Xattention import Xattention_prefill
from flash_attn import flash_attn_func
import types
from qwen0_90_ratio import max as qwen_max

# ── prompt templates ──────────────────────────────────────────────
template_rag = open("prompts/0shot_rag.txt", encoding="utf-8").read()
template_no_context = open("prompts/0shot_no_context.txt", encoding="utf-8").read()
template_0shot = open("prompts/0shot.txt", encoding="utf-8").read()
template_0shot_cot = open("prompts/0shot_cot.txt", encoding="utf-8").read()
template_0shot_cot_ans = open("prompts/0shot_cot_ans.txt", encoding="utf-8").read()


# ── utils ─────────────────────────────────────────────────────────
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_answer(response: str) -> Optional[str]:
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", response)
    if match:
        return match.group(1)
    return None


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


# ── xattention forward (Qwen) ─────────────────────────────────────
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
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

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

    if key_states.shape[2] == query_states.shape[2]:  # prefill
        if self.method == "xattn":
            if isinstance(self.threshold, torch.Tensor):
                self.threshold = self.threshold.to(key_states.device, dtype=key_states.dtype)
            threshold = self.threshold
            stride = self.xattn_stride
            layer_id = int(getattr(self, "layer_idx", -1))
            attn_output = Xattention_prefill(
                query_states,
                key_states,
                value_states,
                type=self.xtype,
                model_name="Qwen",
                layer_id=layer_id,
                norm=1,
                stride=stride,
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
    else:  # decode: full attention
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


# ── model loading ─────────────────────────────────────────────────
def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    else:
        torch_dtype = torch.float32

    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if args.attn_impl != "auto":
        model_kwargs["attn_implementation"] = args.attn_impl
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    else:
        if torch.cuda.is_available():
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    attn_implementation="flash_attention_2",
                    **model_kwargs,
                )
            except Exception as e:
                print(f"[Warn] flash_attention_2 load failed: {e}")
                print("[Warn] Fallback to default attention implementation.")
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )

    model.eval()
    return model, tokenizer


# ── prompt building ───────────────────────────────────────────────
def build_chat_prompt(tokenizer, prompt: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        return prompt


def truncate_token_ids(input_ids: List[int], max_len: int) -> List[int]:
    if len(input_ids) <= max_len:
        return input_ids
    keep_front = max_len // 2
    keep_back = max_len - keep_front
    return input_ids[:keep_front] + input_ids[-keep_back:]


def encode_prompt(tokenizer, prompt: str, max_context_len: int, device: torch.device):
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = truncate_token_ids(input_ids, max_context_len)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def build_base_prompt(item: Dict[str, Any], args) -> str:
    context = item["context"]

    if args.rag > 0:
        template = template_rag
        if "retrieved_context" not in item:
            raise ValueError(
                "RAG mode requires field 'retrieved_context' in each sample, "
                "but it was not found."
            )
        retrieved = item["retrieved_context"][:args.rag]
        retrieved = sorted(retrieved, key=lambda x: x["c_idx"])
        context = "\n\n".join(
            [f"Retrieved chunk {idx + 1}: {x['content']}" for idx, x in enumerate(retrieved)]
        )
    elif args.no_context:
        template = template_no_context
    elif args.cot:
        template = template_0shot_cot
    else:
        template = template_0shot

    prompt = (
        template.replace("$DOC$", context.strip())
        .replace("$Q$", item["question"].strip())
        .replace("$C_A$", item["choice_A"].strip())
        .replace("$C_B$", item["choice_B"].strip())
        .replace("$C_C$", item["choice_C"].strip())
        .replace("$C_D$", item["choice_D"].strip())
    )
    return prompt


# ── generation ────────────────────────────────────────────────────
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_context_len: int,
    max_new_tokens: int,
) -> str:
    device = get_model_device(model)
    chat_prompt = build_chat_prompt(tokenizer, prompt)
    inputs = encode_prompt(tokenizer, chat_prompt, max_context_len, device)
    prompt_len = inputs["input_ids"].shape[-1]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            top_p=1.0,
            top_k=0,
            temperature=1.0,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]

    gen_ids = output_ids[prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text


# ── data loading ──────────────────────────────────────────────────
def load_data(args):
    if args.data_file:
        dataset = load_dataset("json", data_files=args.data_file, split="train")
    else:
        dataset = load_dataset(args.dataset_path, split=args.split)

    data_all = []
    for item in dataset:
        sample = {
            "_id": item["_id"],
            "domain": item["domain"],
            "sub_domain": item["sub_domain"],
            "difficulty": item["difficulty"],
            "length": item["length"],
            "question": item["question"],
            "choice_A": item["choice_A"],
            "choice_B": item["choice_B"],
            "choice_C": item["choice_C"],
            "choice_D": item["choice_D"],
            "answer": item["answer"],
            "context": item["context"],
        }
        if "retrieved_context" in item:
            sample["retrieved_context"] = item["retrieved_context"]
        data_all.append(sample)
    return data_all


# ── output ────────────────────────────────────────────────────────
def build_output_record(item, response, pred, judge, response_cot=None):
    record = {
        "pred": pred,
        "answer": item["answer"],
        "response": response,
    }
    if response_cot is not None:
        record["response_cot"] = response_cot
    record["domain"] = item["domain"]
    record["sub_domain"] = item["sub_domain"]
    record["difficulty"] = item["difficulty"]
    record["length"] = item["length"]
    record["question"] = item["question"]
    record["choice_A"] = item["choice_A"]
    record["choice_B"] = item["choice_B"]
    record["choice_C"] = item["choice_C"]
    record["choice_D"] = item["choice_D"]
    record["_id"] = item["_id"]
    record["context"] = item["context"][:1000]
    record["judge"] = judge
    return record


# ── main prediction loop ──────────────────────────────────────────
def run_longbenchv2_pred(model, tokenizer, data, args, fout):
    for item in tqdm(data, desc="Running prediction"):
        try:
            prompt = build_base_prompt(item, args)

            if args.cot:
                cot_response = generate_text(
                    model=model, tokenizer=tokenizer, prompt=prompt,
                    max_context_len=args.max_context_len,
                    max_new_tokens=args.cot_max_new_tokens,
                )
                final_prompt = (
                    template_0shot_cot_ans
                    .replace("$DOC$", item["context"].strip())
                    .replace("$Q$", item["question"].strip())
                    .replace("$C_A$", item["choice_A"].strip())
                    .replace("$C_B$", item["choice_B"].strip())
                    .replace("$C_C$", item["choice_C"].strip())
                    .replace("$C_D$", item["choice_D"].strip())
                    .replace("$COT$", cot_response)
                )
                response = generate_text(
                    model=model, tokenizer=tokenizer, prompt=final_prompt,
                    max_context_len=args.max_context_len,
                    max_new_tokens=args.max_new_tokens,
                )
                pred = extract_answer(response)
                judge = pred == item["answer"]
                output_record = build_output_record(
                    item=item, response=response, pred=pred, judge=judge,
                    response_cot=cot_response,
                )
            else:
                response = generate_text(
                    model=model, tokenizer=tokenizer, prompt=prompt,
                    max_context_len=args.max_context_len,
                    max_new_tokens=args.max_new_tokens,
                )
                pred = extract_answer(response)
                judge = pred == item["answer"]
                output_record = build_output_record(
                    item=item, response=response, pred=pred, judge=judge,
                )

            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            fout.flush()

        except Exception as e:
            print(f"[Error] sample_id={item.get('_id', 'unknown')} failed: {e}")


# ── entry point ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--model_path", type=str, default="Q/Qwen2.5-7B-Instruct-1M")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct-1M")

    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--data_file", type=str, default="../../../benchmarks/longbenchv2/filtered_longbench_v2_64k-192k.jsonl")
    parser.add_argument("--split", type=str, default="train")

    parser.add_argument("--max_context_len", type=int, default=204800)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--cot_max_new_tokens", type=int, default=1024)

    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--no_context", action="store_true")
    parser.add_argument("--rag", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2",
                        choices=["auto", "eager", "sdpa", "flash_attention_2"])

    # xattention-specific
    parser.add_argument("--method", type=str, default="xattn", choices=["xattn", "full"])
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--type", type=str, default="",
                        help="XAttention diagnostic type (recall, topkrate, or empty for normal)")
    parser.add_argument("--budget", type=float, default=0.9,
                        help="Threshold budget (0.8, 0.85, 0.9, 0.95)")

    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.model_name is None:
        args.model_name = os.path.basename(args.model_path.rstrip("/"))

    # suffix
    if args.rag > 0:
        suffix = f"_rag_{args.rag}"
    elif args.no_context:
        suffix = "_no_context"
    elif args.cot:
        suffix = "_cot"
    else:
        suffix = ""

    method_suffix = f"_xattn_s{args.stride}" if args.method == "xattn" else ""
    out_file = os.path.join(
        args.save_dir,
        safe_model_name(args.model_name) + method_suffix + suffix + ".jsonl",
    )

    print("=" * 60)
    print(f"Method:       {args.method} (stride={args.stride}, budget={args.budget})")
    print(f"Model:        {args.model_name}")
    print(f"Max ctx len:  {args.max_context_len}")
    print(f"Output:       {out_file}")
    print("=" * 60)

    data_all = load_data(args)

    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}

    data = [item for item in data_all if item["_id"] not in has_data]

    print(f"Total samples   : {len(data_all)}")
    print(f"Remaining samples: {len(data)}")

    if len(data) == 0:
        print("No remaining samples to run.")
        return

    model, tokenizer = load_model_and_tokenizer(args)

    # ── xattention patching (qwen ratio) ──
    # Select ratio file by budget
    if args.budget == 0.9:
        ratio_data = qwen_max

    for name, module in model.named_modules():
        if name.split(".")[-1] == "self_attn":
            layer_idx = int(name.split(".")[2])
            module.method = args.method
            module.xattn_stride = args.stride
            module.xtype = args.type
            module.model_name = args.model_name
            if args.method == "xattn":
                module.threshold = torch.tensor(ratio_data[layer_idx])
            module.forward = types.MethodType(new_attention_forward, module)

    print(f"[Info] Qwen XAttention patched ({args.method}, stride={args.stride}, budget={args.budget})")

    with open(out_file, "a", encoding="utf-8") as fout:
        run_longbenchv2_pred(
            model=model, tokenizer=tokenizer, data=data, args=args, fout=fout,
        )


if __name__ == "__main__":
    main()
