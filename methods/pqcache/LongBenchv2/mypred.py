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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from vq_method.qwen25_patch import VQQwen2ForCausalLM
from vq_method.retrieval_based.pq_search import initialize_objects, del_objects


template_rag = open("prompts/0shot_rag.txt", encoding="utf-8").read()
template_no_context = open("prompts/0shot_no_context.txt", encoding="utf-8").read()
template_0shot = open("prompts/0shot.txt", encoding="utf-8").read()
template_0shot_cot = open("prompts/0shot_cot.txt", encoding="utf-8").read()
template_0shot_cot_ans = open("prompts/0shot_cot_ans.txt", encoding="utf-8").read()
TOPP_SAVE_TOPK = os.environ.get("TOPP_SAVE_TOPK") 

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

    if args.enable_vq_cache:
        # PQ-compressed model loading path
        config = AutoConfig.from_pretrained(args.model_path)
        config.compress_ratio = args.compress_ratio
        config.fixbudget = args.fixbudget
        config.budget = args.budget
        config.important_ratio = args.important_ratio
        config.sink_size = args.sink_size
        config.recent_size = args.recent_size
        config.compressor = args.compressor
        config.threshold = args.threshold
        config.n_subvec_per_head = args.n_subvec_per_head
        config.n_subbits = args.n_subbits
        config.topr = args.topr
        config.pp_size = args.pp_size
        config.gqa = (args.gqa == "True")
        config.max_iter = args.max_iter
        config.device = torch.device("cuda:0")
        config.mean_v_trick = (args.sparq_mean_v_trick == "True")
        config.recent_ratio = args.recent_ratio
        config.fixthreshold = args.fixthreshold
        config.keyformer_mode = args.keyformer_mode
        config.drop_ratio = args.drop_ratio
        config.preserve_layer = args.preserve_layer
        config.score_func = args.score_func

        if config.compressor == "pq_search":
            config.max_seq_len = 200000
            config.cache_block_size = 128
            config.global_cache_size = 4096
            config.cache_topk = 32
            initialize_objects(config, model="qwen-2.5-7b")

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        model = VQQwen2ForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, config=config)
        model.patch(config)
        model = model.to("cuda:0").eval()
    elif args.attn_impl != "auto":
        model_kwargs["attn_implementation"] = args.attn_impl
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    else:
        # Try flash_attention_2 first, fall back otherwise
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


def build_chat_prompt(tokenizer, prompt: str) -> str:
    """
    Use the chat template when possible; if the tokenizer doesn't support it, return the raw prompt.
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
    Encode first, then apply the "front half + back half" truncation like in pred.py.
    """
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # input_ids = truncate_token_ids(input_ids, max_context_len)

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
    torch.cuda.empty_cache()
    return text


def load_data(args):
    """
    Supports two ways, in order of priority:
    1) Hugging Face / local dataset repo: --dataset_path
    2) A single json/jsonl file: --data_file
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

    # Keep the rest in original order, but put _id before context
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

    # Put judge last
    record["judge"] = judge

    return record


def run_longbenchv2_pred(model, tokenizer, data, args, fout):
    _id = 0
    for item in tqdm(data, desc="Running prediction"):
        if  TOPP_SAVE_TOPK is not None:
            tmp = TOPP_SAVE_TOPK

            with open(f"record/{tmp}.txt", "a") as f:
                f.write(f"data:{_id}\n")
        try:
            prompt = build_base_prompt(item, args)

            if args.cot:
                # args.cot_max_new_tokens = 10
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

            # Custom field order: response, pred, answer, judge first; question, context last
            ordered_keys = ["response", "pred", "answer", "judge", "response_cot", "_id", "domain", "sub_domain", "difficulty", "length", "choice_A", "choice_B", "choice_C", "choice_D", "question", "context"]
            ordered_item = {k: output_item[k] for k in ordered_keys if k in output_item}
            # Add any other remaining fields
            for k, v in output_item.items():
                if k not in ordered_item:
                    ordered_item[k] = v

            fout.write(json.dumps(ordered_item, ensure_ascii=False) + "\n")
            fout.flush()

        except Exception as e:
            print(f"[Error] sample_id={item.get('_id', 'unknown')} failed: {e}")
        _id += 1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct-1M")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct-1M")

    parser.add_argument("--dataset_path", type=str, default="../../../benchmarks/longbenchv2/")
    parser.add_argument("--data_file", type=str, default="../../../benchmarks/longbenchv2/filtered_longbench_v2_64k-192k.jsonl")
    parser.add_argument("--split", type=str, default="train")

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

    # PQ compression parameters
    parser.add_argument("--enable_vq_cache", action="store_true",
                        help="Enable VQ cache compression")
    parser.add_argument("--compressor", type=str, default="pq_search",
                        help="Compression method")
    parser.add_argument("--compress_ratio", type=float, default=0.1,
                        help="Compression ratio (lower = more compression)")
    parser.add_argument("--fixbudget", action="store_true", default=True,
                        help="Use fixed budget instead of ratio-based calculation")
    parser.add_argument("--no-fixbudget", dest="fixbudget", action="store_false",
                        help="Disable fixed budget mode")
    parser.add_argument("--budget", type=int, default=1024,
                        help="Fixed budget size (used when --fixbudget is set)")
    parser.add_argument("--important_ratio", type=float, default=0.5,
                        help="Important token ratio (for pq_search)")
    parser.add_argument("--recent_ratio", type=float, default=0.5,
                        help="Recent token ratio (for pq_search)")
    parser.add_argument("--sink_size", type=int, default=16,
                        help="Sink tokens (always kept)")
    parser.add_argument("--recent_size", type=int, default=32,
                        help="Recent cache size")
    parser.add_argument("--n_subvec_per_head", type=int, default=2,
                        help="Number of subvectors per head for PQ")
    parser.add_argument("--n_subbits", type=int, default=6,
                        help="Number of bits per subvector for PQ")
    parser.add_argument("--topr", type=int, default=32,
                        help="Top-r tokens for SPARQ")
    parser.add_argument("--gqa", type=str, default="True",
                        help="Use Grouped-Query Attention")
    parser.add_argument("--sparq_mean_v_trick", type=str, default="False",
                        help="Use mean v trick for SPARQ")
    parser.add_argument("--max_iter", type=int, default=0,
                        help="K-means iterations (0=auto)")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Threshold for compression")
    parser.add_argument("--score_func", type=str, default="sum",
                        help="Score function (sum, max)")
    parser.add_argument("--fixthreshold", type=float, default=-1,
                        help="If > 0, use topp attention threshold")
    parser.add_argument("--keyformer_mode", type=int, default=0,
                        help="Keyformer mode")
    parser.add_argument("--drop_ratio", type=float, default=0,
                        help="Drop ratio for H2O")
    parser.add_argument("--preserve_layer", type=int, default=0,
                        help="Number of layers to preserve")
    parser.add_argument('--pp_size', type=int, choices=[1,2,4,8])
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.model_name is None:
        args.model_name = os.path.basename(args.model_path.rstrip("/"))

    print(args)
    if  TOPP_SAVE_TOPK is not None:
            tmp =  TOPP_SAVE_TOPK

            os.makedirs("record", exist_ok=True)

            with open(f"record/{tmp}.txt", "a") as f:
                f.write("begin")
    # if args.rag > 0:
    #     suffix = f"_rag_{args.rag}"
    # elif args.no_context:
    #     suffix = "_no_context"
    # elif args.cot:
    #     suffix = "_cot"
    # elif args.enable_vq_cache:
    #     suffix = f"_pq_{args.compressor}_c{int(args.compress_ratio*100)}_subvec{args.n_subvec_per_head}_bits{args.n_subbits}"
    # else:
    #     suffix = ""
    suffix_parts = []

    if args.rag > 0:
        suffix_parts.append(f"rag_{args.rag}")

    if args.no_context:
        suffix_parts.append("no_context")

    if args.cot:
        suffix_parts.append("cot")

    if args.enable_vq_cache:
        suffix_parts.append(
            f"pq_{args.compressor}_c{int(args.compress_ratio * 100)}"
            f"_subvec{args.n_subvec_per_head}_bits{args.n_subbits}"
        )

    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

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

    # Clean up PQ objects
    if args.enable_vq_cache and args.compressor == "pq_search":
        del_objects()


if __name__ == "__main__":
    main()
