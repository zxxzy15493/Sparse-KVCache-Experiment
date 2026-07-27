import os
import sys
import torch
import json
from tqdm import tqdm
import numpy as np
import random
from pathlib import Path
import argparse
import re
import time


from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import load_data
from accuracy.patch import enable_attention_eval, get_config_output_affix
from accuracy.cluster_attention import cluster_reset


# -----------------------------------------------------------------------------
# Few-shot demonstration prompts (kept identical to pred_full.py)
# -----------------------------------------------------------------------------
def get_examples():
    examples = {}
    examples["gsm8k-cot"] = [
        (
            "question: There are 15 trees in the grove. Grove workers will plant trees in thegrove today. After they are done, there will be 21 trees. How many trees didthe grove workers plant today?",
            "target: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 15 = 6. The answer is 6.",
        ),
        (
            "question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "target: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
        ),
        (
            "question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
            "target: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."
        ),
        (
            "question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12lollipops. How many lollipops did Jason give to Denny?",
            "target: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."
        ),
        (
            "question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
            "target: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."
        ),
        (
            "question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
            "target: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."
        ),
        (
            "question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
            "target: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
        ),
        (
            "question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
            "target: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. The answer is 8.",
        ),
    ]
    return examples


EXAMPLES = get_examples()


def load_prompt(prompt_name, num_shots):
    if not num_shots:
        return []
    return EXAMPLES[prompt_name][:num_shots]


def construct_prompt(example, args):
    demos = load_prompt(args.cot_type, args.num_shots)
    demo_prompt = "".join(
        [q + "\n" + a for q, a in demos]
    )
    return demo_prompt + "\nQuestion: " + example["question"] + "\n"


# -----------------------------------------------------------------------------
# Argparse
# -----------------------------------------------------------------------------
def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # Generation / IO
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--num_shots", type=int, default=8, help="number of shots for few-shot prompting")
    parser.add_argument("--cot_type", type=str, default="gsm8k-cot", help="type of chain-of-thought prompting")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=3000,
        help="maximum number of new tokens to generate",
    )
    parser.add_argument("--do_sample", action="store_true", help="use sampling decoding")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        help="HuggingFace attn_implementation (flash_attention_2 / sdpa / eager)")
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Optional comma-separated list of dataset indices to evaluate (otherwise use the built-in subset).",
    )

    # ---------------------------------------------------------------
    # ClusterKV / Quest sparse-attention switches.
    # Mirrors accuracy/patch.py:parse_common_args so that --cluster
    # plugs into the same enable_attention_eval pipeline used by
    # accuracy/LongBench/recall_pred.py.
    # ---------------------------------------------------------------
    parser.add_argument("--quest", action="store_true", help="Enable Quest Attention")
    parser.add_argument("--cluster", action="store_true", help="Enable ClusterKV Attention")

    parser.add_argument("--token_budget", "--budget", dest="token_budget", type=int, default=1024,
                        help="Total per-head token budget (includes sink tokens)")
    parser.add_argument("--chunk_size", type=int, default=16)

    parser.add_argument("--sink", type=int, default=16,
                        help="Number of leading 'sink' tokens always kept")
    parser.add_argument("--recent", type=int, default=32,
                        help="Number of most-recent tokens always kept")

    parser.add_argument("--head_sel", type=str, choices=["truc", "pad"], default="truc",
                        help="truncate or pad for different heads to make same selection budget")
    parser.add_argument("--balance", action="store_true", help="Use Balanced KMeans")
    parser.add_argument("--nlist", type=int, default=40, help="Number of clusters")
    parser.add_argument("--fit_iter", "--iter", dest="fit_iter", type=int, default=10,
                        help="Max steps for clustering")
    parser.add_argument("--gqa_policy", type=str, choices=["qavg"], default=None)
    parser.add_argument(
        "--dist_t", type=str,
        choices=["cosine", "inner_product", "l2", "l1", "euclidean",
                 "chebyshev", "canberra"],
        default="cosine", help="Distance for clustering",
    )

    parser.add_argument("--cache_steps", type=int, default=0,
                        help="Stat cache hit rate of recent steps")
    parser.add_argument("--topk_stat", action="store_true", help="Stat hit rate of TopK tokens")

    parsed = parser.parse_args(args)
    assert not (parsed.quest and parsed.cluster), "--quest and --cluster cannot be enabled at the same time"
    return parsed


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
def load_dataset(pred_dir, out_filename):
    data_file = '../../../benchmarks/gsm8k/data/gsm8k_test.jsonl'
    datas = load_data(data_file)
    for i, data in enumerate(datas):
        data.setdefault('index', i)

    out_path = Path(pred_dir) / out_filename

    if os.path.exists(out_path):
        # Skip already-predicted samples so re-runs only fill the gaps.
        pred_index = [sample["index"] for sample in load_data(out_path)]
        data = [sample for sample in datas if sample["index"] not in pred_index]
    else:
        data = datas

    return data


def parse_indices(indices_str):
    if indices_str is None:
        return None
    out = []
    for part in indices_str.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


# -----------------------------------------------------------------------------
# Output filename
# -----------------------------------------------------------------------------
def get_out_filename(args):
    """
    Layout:
        gsm8k.jsonl                                # full attention
        gsm8k-<config>.jsonl                       # cluster / quest
    """
    config_affix = get_config_output_affix(args)
    if config_affix:
        # Encode the run-specific 'recent' window in the filename so different
        # values don't silently overwrite each other.
        config_affix += f"r{args.recent}"
    return f"gsm8k{config_affix}.jsonl"


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------
def get_pred(llm, message, data, tokenizer, out_path, args):
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, (list, tuple)):
            tokenizer.pad_token_id = tokenizer.eos_token_id[0]
        else:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt = [{"role": "user", "content": message}]
    prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    print(prompt)

    inputs = tokenizer([prompt], return_tensors="pt", padding=True)
    seq_len = inputs.input_ids.shape[1]
    print(f"\nInput id length is : {inputs.input_ids.shape}\n")

    # Each request starts with a clean cluster cache, mirroring recall_pred.py.
    if args.cluster:
        cluster_reset(llm)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = llm.generate(
        **inputs.to(llm.device),
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    gen_len = len(out[0]) - seq_len
    print(f"\nOutput length is : {gen_len}, gen_time={t1 - t0:.2f}s\n")

    output = tokenizer.batch_decode(out[:, seq_len:], skip_special_tokens=True)

    torch.cuda.empty_cache()
    pred = output[0]

    pattern_final_answer = r"#### (\d{1,3}(?:,\d{3})*(?:\.?\d+)?)"
    final_answer = re.search(pattern_final_answer, data['answer'])
    if final_answer:
        final_answer = final_answer.group(1)

    with open(out_path, "a", encoding="utf-8") as f:
        json.dump(
            {
                "index": data.get("index"),
                "match_result": "null",
                "final_answer": final_answer,
                "pred": pred,
            },
            f,
            ensure_ascii=False,
        )
        f.write('\n')


# -----------------------------------------------------------------------------
# Model loading + attention patch
# -----------------------------------------------------------------------------
def _patch_model_name_for_eval(model_str):
    """
    enable_attention_eval branches on substrings in the model_name string
    ("llama", "wen2", "glm4", "intern"). HuggingFace ids like
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' contain 'Qwen' but not 'wen2',
    so they fail the Qwen2 dispatch check. Normalize the string here.
    """
    lowered = model_str.lower()
    if "qwen" in lowered and "wen2" not in lowered:
        lowered = lowered.replace("qwen", "qwen2", 1)
    return lowered


def load_model_and_tokenizer(args, dtype):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    kwargs = dict(
        torch_dtype=dtype,
        device_map="cuda:0",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=True,
    )
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    llm = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    llm = llm.eval()

    if args.quest or args.cluster:
        enable_attention_eval(_patch_model_name_for_eval(args.model), llm, args)

    return tokenizer, llm


if __name__ == "__main__":
    args = parse_args()

    dtype = torch.bfloat16
    pred_dir = args.save_dir
    pred_dir.mkdir(parents=True, exist_ok=True)

    out_filename = get_out_filename(args)
    gsm8k_datas = load_dataset(pred_dir, out_filename)

    tokenizer, llm = load_model_and_tokenizer(args, dtype)

    if args.cluster:
        print(
            f"[cluster] token_budget={args.token_budget} sink={args.sink} "
            f"recent={args.recent} nlist={args.nlist} fit_iter={args.fit_iter}"
        )

    pred_indices = parse_indices(args.indices)
    # If --indices is explicitly given, use it as a filter; otherwise run all.
    pred_indices = set(pred_indices) if pred_indices is not None else None

    out_path = Path(pred_dir) / out_filename

    for data_sample in tqdm(gsm8k_datas):
        if pred_indices is not None and data_sample['index'] not in pred_indices:
            continue
        message = construct_prompt(data_sample, args)
        get_pred(
            llm,
            message,
            data_sample,
            tokenizer,
            out_path,
            args,
        )
