import os
import re
import json
import argparse
import random
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


from llminference.myexperiments import Sparsity,SparsityMethods


template_rag = open("prompts/0shot_rag.txt", encoding="utf-8").read()
template_no_context = open("prompts/0shot_no_context.txt", encoding="utf-8").read()
template_0shot = open("prompts/0shot.txt", encoding="utf-8").read()
template_0shot_cot = open("prompts/0shot_cot.txt", encoding="utf-8").read()
template_0shot_cot_ans = open("prompts/0shot_cot_ans.txt", encoding="utf-8").read()


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


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
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
        #  flash_attention_2
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

    sparq = Sparsity(name="ann", k=4096, local_k=32, rank=16, score="sparse_q",
                 reallocate_to_mean_value=True)
    model = SparsityMethods.apply(sparq, model)

    model.eval()
    return model, tokenizer


def build_chat_prompt(tokenizer, prompt: str) -> str:
    """
     chat template tokenizer  prompt
    """
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
    """
     pred.py “ + ”
    """
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = truncate_token_ids(input_ids, max_context_len)

    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


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


def load_data(args):
    """
    
    1) Hugging Face /  dataset repo: --dataset_path
    2)  json/jsonl : --data_file
    """
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

def build_output_record(item, response, pred, judge, response_cot=None):
    record = {
        "pred": pred,
        "answer": item["answer"],
        "response": response,
    }

    if response_cot is not None:
        record["response_cot"] = response_cot

    #  _id  context 
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

    #  judge
    record["judge"] = judge

    return record
def run_longbenchv2_pred(model, tokenizer, data, args, fout):
    for item in tqdm(data, desc="Running prediction"):
        try:
            prompt = build_base_prompt(item, args)

            if args.cot:
                cot_response = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
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
                    model=model,
                    tokenizer=tokenizer,
                    prompt=final_prompt,
                    max_context_len=args.max_context_len,
                    max_new_tokens=args.max_new_tokens,
                )

                pred = extract_answer(response)
                judge = pred == item["answer"]

                output_record = build_output_record(
                    item=item,
                    response=response,
                    pred=pred,
                    judge=judge,
                    response_cot=cot_response,
                )

            else:
                response = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_context_len=args.max_context_len,
                    max_new_tokens=args.max_new_tokens,
                )

                pred = extract_answer(response)
                judge = pred == item["answer"]

                output_record = build_output_record(
                    item=item,
                    response=response,
                    pred=pred,
                    judge=judge,
                )

            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            fout.flush()

        except Exception as e:
            print(f"[Error] sample_id={item.get('_id', 'unknown')} failed: {e}")


def run_longbenchv2_pred(model, tokenizer, data, args, fout):
    for item in tqdm(data, desc="Running prediction"):
        try:
            prompt = build_base_prompt(item, args)

            if args.cot:
                cot_response = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
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

                final_response = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=final_prompt,
                    max_context_len=args.max_context_len,
                    max_new_tokens=args.max_new_tokens,
                )

                output_item = dict(item)
                output_item["response_cot"] = cot_response
                output_item["response"] = final_response
            else:
                response = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_context_len=args.max_context_len,
                    max_new_tokens=args.max_new_tokens,
                )

                output_item = dict(item)
                output_item["response"] = response

            output_item["pred"] = extract_answer(output_item["response"])
            output_item["judge"] = output_item["pred"] == output_item["answer"]
            output_item["context"] = output_item["context"][:1000]

            # pred = extract_answer(response)
            # judge = pred == item["answer"]

            # output_record = build_output_record(
            #     item=item,
            #     response=response,
            #     pred=pred,
            #     judge=judge,
            # )


            # output_record = output_item["response_cot"]
            fout.write(json.dumps(output_item, ensure_ascii=False) + "\n")
            fout.flush()

        except Exception as e:
            print(f"[Error] sample_id={item.get('_id', 'unknown')} failed: {e}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")

    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--data_file", type=str, default="../../../benchmarks/longbenchv2/filtered_longbench_v2_64k-192k.jsonl")
    parser.add_argument("--split", type=str, default="")

    parser.add_argument("--max_context_len", type=int, default=204800)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--cot_max_new_tokens", type=int, default=1024)

    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--no_context", action="store_true")
    parser.add_argument("--rag", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="flash_attention_2",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
    )

    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.model_name is None:
        args.model_name = os.path.basename(args.model_path.rstrip("/"))

    print(args)

    if args.rag > 0:
        suffix = f"_rag_{args.rag}"
    elif args.no_context:
        suffix = "_no_context"
    elif args.cot:
        suffix = "_cot"
    else:
        suffix = ""

    out_file = os.path.join(
        args.save_dir,       #01. 02
        safe_model_name(args.model_name) + suffix + "_50k.jsonl"
    )

    data_all = load_data(args)

    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}

    data = [item for item in data_all if item["_id"] not in has_data]

    print(f"Total samples   : {len(data_all)}")
    print(f"Remaining samples: {len(data)}")
    print(f"Output file     : {out_file}")

    if len(data) == 0:
        print("No remaining samples to run.")
        return

    model, tokenizer = load_model_and_tokenizer(args)

    with open(out_file, "a", encoding="utf-8") as fout:
        run_longbenchv2_pred(
            model=model,
            tokenizer=tokenizer,
            data=data,
            args=args,
            fout=fout,
        )


if __name__ == "__main__":
    main()
