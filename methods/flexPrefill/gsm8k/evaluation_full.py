from email.mime import text
import os
import sys
import torch
import json
from tqdm import tqdm
import numpy as np
import random
from pathlib import Path
import argparse
#from utils import load_data
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = REPO_ROOT / "benchmarks" / "gsm8k" / "data" / "gsm8k_test.jsonl"

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import json
from flex_prefill import patch_model    # 

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
        
    # demos = load_prompt(args.cot_type, args.num_shots)
    demos = load_prompt('gsm8k-cot', 8)
    demo_prompt = "".join(
        [
            q + "\n" + a
            for q, a in demos
        ]
    )
    return demo_prompt + "\nQuestion: " + example["question"] + "\n"

def dump_jsonl(fname, data):
    with open(fname, "w", encoding="utf8") as fout:
        for line in data:
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")

def iter_jsonl(fname, cnt=None):
    i = 0
    with open(fname, "r") as fin:
        for line in fin:
            if i == cnt:
                break
            yield json.loads(line)
            i += 1           

def load_data(fname):
    return list(iter_jsonl(fname))


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--num_shots", type=int, default=8, help="number of shots for few-shot prompting")
    parser.add_argument("--cot_type", type=str, default="gsm8k-cot", help="type of chain-of-thought prompting")

    # Generation controls (avoid Transformers default max_length=20)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=3000,
        help="maximum number of new tokens to generate",
    )
    parser.add_argument("--do_sample", action="store_true", help="use sampling decoding")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--token_budget", type=int, default=4096)
    parser.add_argument("--chunk_size", type=int, default=16)


    return parser.parse_args(args)

def load_dataset(pred_dir, prompt_cot=None):
    # load data
    datas = load_data(DATA_FILE)
    for i, data in enumerate(datas):
        data.setdefault('index', i)
    out_path = Path(pred_dir) / "gsm8k.jsonl"

    if os.path.exists(out_path): ## pred
        pred_index = [sample["index"] for sample in load_data(out_path)]
        data = [sample for sample in datas if sample["index"] not in pred_index]
    else:
        data = datas

    return data

def get_pred(llm, message, data, tokenizer, out_path, args):  

     # Some chat tokenizers ship without pad_token_id; make padding explicit.
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif tokenizer.pad_token_id is None and len(tokenizer.eos_tokens) > 0:
        tokenizer.pad_token_id = tokenizer.eos_token_id[0]

    prompt = [{"role": "user", "content": message }]
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
    print(f"\nOutput length is : {len(out[0])- seq_len}\n")
    output = tokenizer.batch_decode(out[:, seq_len:], skip_special_tokens=True)
    
    torch.cuda.empty_cache()
    out_path = Path(out_path) / f"gsm8k.jsonl"
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

if __name__ == "__main__":  
    args = parse_args()

    dataset = "gsm8k_test"
    dtype = torch.bfloat16
    pred_dir = args.save_dir
    pred_dir.mkdir(parents=True, exist_ok=True)

    gsm8k_datas = load_dataset(pred_dir)
    # Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="cuda:0")
    #     
    flex_prefill_config = {
        "block_size": 128,
        "flex_prefill_gamma":0.8,
        "flex_prefill_tau": 0.1,
        "flex_prefill_min_budget": 512,
        "flex_prefill_max_budget": None,
    }
    #  dense attention  FlexPrefill sparse attention
    patch_model(llm, "flex_prefill", flex_prefill_config)
    llm.eval()
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
