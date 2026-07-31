"""
GSM8K prediction script with DuoAttention compressed attention.

Usage:
    python pred_duo.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --save_dir ./results/DuoAttention/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --attn_load_dir ../attn_patterns/deepseek-r1-distill-qwen-1.5b \
        --sparsity 0.5 \
        --sink_size 128 \
        --recent_size 256

    Then evaluate:
    python evaluate.py --input ./results/DuoAttention/.../gsm8k.jsonl --output ./results/DuoAttention/.../gsm8k_eval.jsonl
"""

import os
import sys
import torch
import json
from tqdm import tqdm
import numpy as np
import random
from pathlib import Path
import argparse
from utils import load_data
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


# ---------------------------------------------------------------------------
# DuoAttention imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from duo_attn.patch import enable_duo_attention_eval
from duo_attn.utils import load_attn_pattern, sparsify_attention_heads
from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache


# ---------------------------------------------------------------------------
# Few-shot demonstration prompts  (identical to pred_cake.py / pred_full.py)
# ---------------------------------------------------------------------------
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
    demos = load_prompt('gsm8k-cot', 8)
    demo_prompt = "".join(
        [q + "\n" + a for q, a in demos]
    )
    return demo_prompt + "\nQuestion: " + example["question"] + "\n"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # Model / IO
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--num_shots", type=int, default=8)
    parser.add_argument("--cot_type", type=str, default="gsm8k-cot")
    parser.add_argument("--max_new_tokens", type=int, default=10000)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    # DuoAttention parameters
    parser.add_argument("--attn_load_dir", type=str, required=True,
                        help="Directory containing full_attention_heads{_latest}.tsv and config.json")
    parser.add_argument("--sparsity", type=float, default=0.5,
                        help="Attention sparsity ratio")
    parser.add_argument("--sink_size", type=int, default=None,
                        help="Sink token count (auto from config.json if not set)")
    parser.add_argument("--recent_size", type=int, default=None,
                        help="Recent/window token count (auto from config.json if not set)")

    return parser.parse_args(args)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_dataset(pred_dir):
    _repo_root = Path(__file__).resolve().parents[4]
    data_file = str(_repo_root / 'benchmarks' / 'gsm8k' / 'data' / 'gsm8k_test.jsonl')
    datas = load_data(data_file)
    for i, data in enumerate(datas):
        data.setdefault('index', i)
    out_path = Path(pred_dir) / "gsm8k.jsonl"

    if os.path.exists(out_path):
        pred_index = [sample["index"] for sample in load_data(out_path)]
        data = [sample for sample in datas if sample["index"] not in pred_index]
    else:
        data = datas

    return data


# ---------------------------------------------------------------------------
# DuoAttention model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_name, args):
    dtype = torch.bfloat16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # DuoAttention requires eager attention implementation
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    model = model.eval()

    # Load attention pattern
    attn_dir = args.attn_load_dir
    # Handle both full_attention_heads.tsv and full_attention_heads_latest.tsv
    tsv_path = os.path.join(attn_dir, "full_attention_heads.tsv")
    if not os.path.exists(tsv_path):
        tsv_path = os.path.join(attn_dir, "full_attention_heads_latest.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"No full_attention_heads.tsv or _latest.tsv found in {attn_dir}")

    # Load config for sink_size / recent_size if not provided by user
    config_path = os.path.join(attn_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            train_config = json.load(f)
        sink_size = args.sink_size if args.sink_size is not None else train_config.get("sink_size", 64)
        recent_size = args.recent_size if args.recent_size is not None else train_config.get("recent_size", 256)
    else:
        sink_size = args.sink_size or 64
        recent_size = args.recent_size or 256

    print(f"Loading attention pattern from {attn_dir} (sparsity={args.sparsity})")
    print(f"  sink_size={sink_size}, recent_size={recent_size}")

    full_attention_heads = np.loadtxt(tsv_path, dtype=float, delimiter="\t")
    full_attention_heads = np.clip(full_attention_heads, 0, 1)

    full_attention_heads, true_sparsity = sparsify_attention_heads(
        full_attention_heads, threshold=None, sparsity=args.sparsity
    )
    print(f"True sparsity: {true_sparsity}")

    # Enable tuple KV cache first (required for duo_attn eval)
    enable_tuple_kv_cache(model)

    # Enable DuoAttention eval
    # DeepSeek-R1-Distill-Qwen-1.5B is Qwen2-based
    enable_duo_attention_eval(
        model,
        full_attention_heads,
        sink_size,
        recent_size,
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@torch.inference_mode()
def get_pred(llm, message, data, tokenizer, out_path, args):

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif tokenizer.pad_token_id is None and len(tokenizer.eos_tokens) > 0:
        tokenizer.pad_token_id = tokenizer.eos_token_id[0]

    prompt = [{"role": "user", "content": message}]
    prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    print(prompt)

    inputs = tokenizer([prompt], return_tensors="pt", padding=True)
    seq_len = inputs.input_ids.shape[1]
    print(f"\nInput id length is : {inputs.input_ids.shape}\n")

    out = llm.generate(
        **inputs.to(llm.device),
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(f"\nOutput length is : {len(out[0]) - seq_len}\n")
    output = tokenizer.batch_decode(out[:, seq_len:], skip_special_tokens=True)

    torch.cuda.empty_cache()
    out_path = Path(out_path) / "gsm8k.jsonl"
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
            ensure_ascii=False
        )
        f.write('\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    seed_everything(42)

    pred_dir = args.save_dir
    os.makedirs(pred_dir, exist_ok=True)

    gsm8k_datas = load_dataset(pred_dir)

    print(f"Loading model: {args.model} with DuoAttention (sparsity={args.sparsity})")
    llm, tokenizer = load_model_and_tokenizer(args.model, args)
    print("Model loaded.")

    for data_sample in tqdm(gsm8k_datas):
        message = construct_prompt(data_sample, args)
        get_pred(
            llm,
            message,
            data_sample,
            tokenizer,
            pred_dir,
            args,
        )

    print("All gsm8k evaluation done.")