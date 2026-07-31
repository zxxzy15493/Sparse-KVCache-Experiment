"""
GSM8K prediction script with CakeKV compressed attention.

Usage:
    python pred_cake.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --save_dir ./results/cakekv \
        --cache_size 1024 \
        --window_size 32 \
        --gamma 200.0

    Then evaluate:
    python evaluate.py --input ./results/cakekv/gsm8k.jsonl --output ./results/cakekv/gsm8k_eval.jsonl
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
# CakeKV imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from cakekv.cake.cake_cache import CakeprefillKVCache
from cakekv.cake.utils import CompressConfig


# ---------------------------------------------------------------------------
# Few-shot demonstration prompts  (identical to pred_full.py)
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
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--num_shots", type=int, default=8)
    parser.add_argument("--cot_type", type=str, default="gsm8k-cot")
    parser.add_argument("--max_new_tokens", type=int, default=10000)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    # CakeKV compression parameters
    parser.add_argument("--compress", action="store_true", default=False,
                        help="Enable CakeKV cache compression")
    parser.add_argument("--cascading", action="store_true",
                        help="Use cascading cache management")
    parser.add_argument("--cache_size", type=int, default=1024,
                        help="Per-layer cache budget (tokens)")
    parser.add_argument("--window_size", type=int, default=32,
                        help="Number of recent tokens always kept")
    parser.add_argument("--tau1", type=float, default=None,
                        help="Entropy exponent (auto by default)")
    parser.add_argument("--tau2", type=float, default=None,
                        help="Variance exponent (auto by default)")
    parser.add_argument("--gamma", type=float, default=200.0,
                        help="Variance weight in CakeKV scoring")

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
# CakeKV model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_name, device, compress_config):
    """
    Load a model and optionally monkey-patch its attention with CakeKV.
    Adapted from cakekv/experiments/LongBench/pred_cake.py
    """

    # Determine model type from the name
    model_name_lower = model_name.lower()

    # Apply CakeKV monkey-patch BEFORE loading the model
    if compress_config.compress:
        if "llama" in model_name_lower:
            from cakekv.cake.monkeypatch import replace_flashllama_attn_with_cakeattn
            replace_flashllama_attn_with_cakeattn()
        elif "mistral" in model_name_lower:
            from cakekv.cake.monkeypatch import replace_flashmistral_attn_with_cakeattn
            replace_flashmistral_attn_with_cakeattn()
        elif "qwen2" in model_name_lower or "qwen" in model_name_lower:
            from cakekv.cake.monkeypatch import replace_flashqwen2_attn_with_cakeattn
            replace_flashqwen2_attn_with_cakeattn()
        else:
            raise ValueError(f"Unsupported model type for CakeKV: {model_name}")

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Load model with flash attention (required by CakeKV monkey-patch)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)

    # Configure CakeKV for each layer
    if compress_config.compress:
        layers = config.num_hidden_layers
        for i in range(layers):
            attn = model.model.layers[i].self_attn
            attn.config.key_size = [compress_config.cache_size - compress_config.window_size] * layers
            attn.config.window_size = [compress_config.window_size] * layers
            attn.config.prefill = [True] * layers
            attn.config.decoding_evict = [None] * layers
            attn.config.tau1 = compress_config.hyper[0]
            attn.config.tau2 = compress_config.hyper[1]
            attn.config.gamma = compress_config.hyper[2]
            attn.config.prefill_cake_evict = [CakeprefillKVCache(
                cache_size=compress_config.cache_size,
                window_size=compress_config.window_size,
                k_seq_dim=2,
                v_seq_dim=2,
                num_heads=attn.num_heads,
                num_layers=layers,
                use_cascading=compress_config.cascading,
            )] * layers

    model = model.eval()
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

    # After generation, reset CakeKV per-layer state for next sample
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

    # Reset CakeKV per-layer state for next sample
    if hasattr(llm, "model") and hasattr(llm.model, "layers"):
        for i in range(len(llm.model.layers)):
            attn = llm.model.layers[i].self_attn
            attn.config.prefill = [True] * len(llm.model.layers)
            attn.config.decoding_evict = [None] * len(llm.model.layers)

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

    dataset = "gsm8k_test"
    dtype = torch.bfloat16
    pred_dir = args.save_dir
    os.makedirs(pred_dir, exist_ok=True)

    gsm8k_datas = load_dataset(pred_dir)

    # Build CakeKV compress config
    compress_config = CompressConfig(
        compress=args.compress,
        cascading=args.cascading,
        cache_size=args.cache_size,
        window_size=args.window_size,
    )

    # Try to load tau values from cakekv's config if available
    tau1, tau2 = args.tau1, args.tau2
    if tau1 is None or tau2 is None:
        try:
            model2tau_path = Path(__file__).resolve().parent.parent.parent / "experiments" / "LongBench" / "config" / "model2tau.json"
            if model2tau_path.exists():
                model2tau = json.loads(model2tau_path.read_text())
                model_key = args.model
                if model_key in model2tau and str(args.cache_size) in model2tau[model_key]:
                    tau1 = model2tau[model_key][str(args.cache_size)]["tau1"]
                    tau2 = model2tau[model_key][str(args.cache_size)]["tau2"]
                    print(f"Loaded tau values: tau1={tau1}, tau2={tau2}")
        except Exception as e:
            print(f"Could not load tau config: {e}")

    if tau1 is None:
        tau1 = 1.0
    if tau2 is None:
        tau2 = 1.0

    gamma = args.gamma
    compress_config.hyper = [tau1, tau2, gamma]
    print(f"CakeKV config: {compress_config}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {args.model} with CakeKV (compress={args.compress})")
    llm, tokenizer = load_model_and_tokenizer(args.model, device, compress_config)
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