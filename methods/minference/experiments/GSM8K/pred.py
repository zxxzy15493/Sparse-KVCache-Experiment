import os
import sys
import json
import re
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from minference import MInference
from utils import load_data
from examples import get_examples

model2path = json.load(open("./config/model2path.json", "r"))
model2maxlen = json.load(open("./config/model2maxlen.json", "r"))

EXAMPLES = get_examples()


def load_prompt(prompt_name, num_shots):
    if not num_shots:
        return []
    return EXAMPLES[prompt_name][:num_shots]


def construct_prompt(example, args):
    demos = load_prompt('gsm8k-cot', 8)
    demo_prompt = "".join(
        [
            q + "\n" + a
            for q, a in demos
        ]
    )
    return demo_prompt + "\nQuestion: " + example["question"] + "\n"


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_shots", type=int, default=8)
    parser.add_argument("--cot_type", type=str, default="gsm8k-cot")
    parser.add_argument("--max_new_tokens", type=int, default=3000)
    return parser.parse_args(args)


def load_dataset(dataset, pred_dir):
    data_file = f"./data/{dataset}.jsonl"
    datas = load_data(data_file)
    for i, data in enumerate(datas):
        data.setdefault("index", i)

    out_path = Path(pred_dir) / "gsm8k.jsonl"

    if os.path.exists(out_path):
        pred_index = [sample["index"] for sample in load_data(out_path)]
        data = [sample for sample in datas if sample["index"] not in pred_index]
    else:
        data = datas

    return data


def get_pred(model, tokenizer, message, data, max_new_tokens, out_path):
    prompt = [{"role": "user", "content": message }]
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    seq_len = input_ids.shape[1]
    print(f"Input token length: {seq_len}")

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=1.0,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )

    pred = tokenizer.decode(out[0, seq_len:], skip_special_tokens=True)
    print(f"Output token length: {len(out[0]) - seq_len}")

    # Extract final answer
    pattern = r"<answer>(.*?)</answer>"
    match_answer = re.search(pattern, pred)
    if match_answer:
        match_answer = match_answer.group(1)

    pattern_final_answer = r"#### (\d{1,3}(?:,\d{3})*(?:\.?\d+)?)"
    final_answer = re.search(pattern_final_answer, data["answer"])
    if final_answer:
        final_answer = final_answer.group(1)

    with open(out_path, "a", encoding="utf-8") as f:
        json.dump(
            {
                "index": data.get("index"),
                "match_result": match_answer,
                "final_answer": final_answer,
                "pred": pred,
                "answer": data["answer"],
                "question": data["question"],
            },
            f,
            ensure_ascii=False,
        )
        f.write("\n")

    torch.cuda.empty_cache()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model(model_path, max_len, dtype, device, max_new_tokens):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    minference_patch = MInference(model_path)
    model = minference_patch(model)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    dataset = "gsm8k_test"
    model_name = args.model_name
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    model_path = model2path[model_name]
    max_new_tokens = args.max_new_tokens

    pred_dir = args.save_dir
    out_path = Path(pred_dir) / "gsm8k.jsonl"

    gsm8k_datas = load_dataset(dataset, pred_dir)
    print(f"Number of samples to predict: {len(gsm8k_datas)}")

    model, tokenizer = load_model(
        model_path, args.max_len, dtype, args.device, max_new_tokens
    )
    print(f"Model loaded on {model.device}")

    for data_sample in tqdm(gsm8k_datas):
        full_prompt = construct_prompt(data_sample, tokenizer, args)
        get_pred(
            model,
            tokenizer,
            full_prompt,
            data_sample,
            max_new_tokens,
            out_path,
        )
