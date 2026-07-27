"""
GSM8K prediction script with VQ cache support.
Reference: vq_pred.py (root level). Model path/name is passed directly via --model,
no config/model2path.json or config/model2maxlen.json reading.
"""
import os
import sys
import re
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Add parent dir to sys.path so we can import vq_method / h2o_method / utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_data  # gsm8k/utils.py (loaded via sys.path)

from vq_method.llama_patch import VQLlamaForCausalLM
from vq_method.llama31_patch import VQLlama31ForCausalLM
from vq_method.qwen25_patch import VQQwen2ForCausalLM
from vq_method.mistral_patch import VQMistralForCausalLM
try:
    from vq_method.glm_patch import VQGlmForCausalLM
except ModuleNotFoundError:
    VQGlmForCausalLM = None
from h2o_method.h2o_attention import H2OLlamaForCausalLM, H2OLlamaAttention
from vq_method.retrieval_based.pq_search import initialize_objects, del_objects


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


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # Model is now passed directly (no config/model2path.json lookup)
    parser.add_argument("--model", type=str, required=True,
                        help="Model path or HF model name, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Short model family identifier used to pick the VQ patch "
                             "(e.g. qwen-2.5-7b, llama-3.1, llama-3.1-8b-instruct, glm-9b, mistral-7b-Instruct-32k)")
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--num_shots", type=int, default=8, help="number of shots for few-shot prompting")
    parser.add_argument("--cot_type", type=str, default="gsm8k-cot", help="type of chain-of-thought prompting")

    # Generation controls
    parser.add_argument("--max_new_tokens", type=int, default=3000,
                        help="maximum number of new tokens to generate")
    parser.add_argument("--do_sample", action="store_true", help="use sampling decoding")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    # VQ / H2O cache parameters (mirroring vq_pred.py)
    compressor_choices = ["h2o", "original", "no_drop_lb", "pq_search", "sparq_f",
                          "no_drop_lb_32", "topp", "no_drop_lb_topp", "no_drop_lb_topp32"]
    parser.add_argument("--compress_ratio", type=float, default=1)
    parser.add_argument("--fixbudget", action="store_true", default=True,
                        help="Use fixed budget instead of ratio-based calculation")
    parser.add_argument("--no-fixbudget", dest="fixbudget", action="store_false",
                        help="Disable fixed budget mode")
    parser.add_argument("--budget", type=int, default=1024,
                        help="Fixed budget size (used when --fixbudget is set)")
    parser.add_argument("--important_ratio", type=float, default=0)
    parser.add_argument("--recent_ratio", type=float, default=1)
    parser.add_argument("--enable_vq_cache", action='store_true')
    parser.add_argument("--enable_h2o_cache", action='store_true')
    parser.add_argument("--sink-size", type=int, default=16)
    parser.add_argument("--keyformer_mode", type=int, default=0)
    parser.add_argument("--drop_ratio", type=float, default=0)
    parser.add_argument("--preserve_layer", type=int, default=0)
    parser.add_argument("--score_func", type=str, default="sum")
    parser.add_argument("--compressor", type=str, default="off", choices=compressor_choices)
    parser.add_argument("--threshold", type=float, default=1)
    parser.add_argument("--n_subvec_per_head", type=int, default=0)
    parser.add_argument("--n_subbits", type=int, default=0)
    parser.add_argument("--topr", type=int, default=32)
    parser.add_argument("--recent_size", type=int, default=32)
    parser.add_argument("--gqa", type=str, default="True")
    parser.add_argument("--sparq_mean_v_trick", type=str, default="False")
    parser.add_argument("--max_iter", type=int, default=0)
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision")
    parser.add_argument("--pp-size", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument("--test_mode", action='store_true')
    parser.add_argument("--fixthreshold", type=float, default=-1,
                        help="if > 0, use topp attention")

    return parser.parse_args(args)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_dataset(pred_dir):
    # load data
    data_file = '../../../benchmarks/gsm8k/data/gsm8k_test.jsonl'
    datas = load_data(data_file)
    for i, data in enumerate(datas):
        data.setdefault('index', i)
    out_path = Path(pred_dir) / "gsm8k.jsonl"

    if os.path.exists(out_path):  # resume from partial predictions
        pred_index = [sample["index"] for sample in load_data(out_path)]
        data = [sample for sample in datas if sample["index"] not in pred_index]
    else:
        data = datas

    return data


def load_model_and_tokenizer(args, device, pp_size=1):
    """Load model with the appropriate VQ/H2O patch.

    Mirrors vq_pred.load_model_and_tokenizer, but takes the model path directly
    from `args.model` instead of looking it up in config/model2path.json.
    """
    path = args.model
    model_name = args.model_name

    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        model = model.eval().to(device)
    elif "mistral" in model_name:
        config = AutoConfig.from_pretrained(path)
        config.recent_ratio = args.recent_ratio
        config.compress_ratio = args.compress_ratio
        config.fixbudget = args.fixbudget
        config.budget = args.budget
        config.important_ratio = args.important_ratio
        config.pp_size = pp_size
        config.sink_size = args.sink_size
        config.recent_size = args.recent_size
        config.keyformer_mode = (args.keyformer_mode == 1)
        config.drop_ratio = args.drop_ratio
        config.preserve_layer = args.preserve_layer
        config.score_func = args.score_func
        config.compressor = args.compressor
        config.threshold = args.threshold
        config.n_subvec_per_head = args.n_subvec_per_head
        config.n_subbits = args.n_subbits
        config.topr = args.topr
        config.gqa = (args.gqa == "True")
        config.mean_v_trick = (args.sparq_mean_v_trick == "True")
        config.max_iter = args.max_iter
        config.device = torch.device("cuda:0")

        if config.compressor == "pq_search":
            config.max_seq_len = 33000
            config.cache_block_size = 128
            config.global_cache_size = 4096
            config.cache_topk = 32
            initialize_objects(config, model="mistral")

        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, config=config)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = VQMistralForCausalLM.from_pretrained(path, config=config)
        model = model.half().eval()
    elif "llama" in model_name and "2." in model_name:
        config = AutoConfig.from_pretrained(path)
        config.compress_ratio = args.compress_ratio
        config.fixbudget = args.fixbudget
        config.budget = args.budget
        config.important_ratio = args.important_ratio
        config.pp_size = pp_size
        config.sink_size = args.sink_size
        config.recent_size = args.recent_size
        config.keyformer_mode = (args.keyformer_mode == 1)
        config.drop_ratio = args.drop_ratio
        config.preserve_layer = args.preserve_layer
        config.score_func = args.score_func
        config.compressor = args.compressor
        config.threshold = args.threshold
        config.n_subvec_per_head = args.n_subvec_per_head
        config.n_subbits = args.n_subbits
        config.topr = args.topr
        config.gqa = (args.gqa == "True")
        config.max_iter = args.max_iter
        config.device = torch.device("cuda:0")
        config.mean_v_trick = (args.sparq_mean_v_trick == "True")
        config.recent_ratio = args.recent_ratio
        if args.enable_vq_cache:
            config.compress_ratio = args.compress_ratio
            config.important_ratio = args.important_ratio
        elif args.enable_h2o_cache:
            config.hh_ratio = args.important_ratio

        if config.compressor == "pq_search":
            config.max_seq_len = 32768
            config.cache_block_size = 128
            config.global_cache_size = 4096
            config.cache_topk = 32
            initialize_objects(config, model=model_name)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        if args.enable_vq_cache:
            model = VQLlamaForCausalLM.from_pretrained(path, config=config)
        elif args.enable_h2o_cache:
            model = H2OLlamaForCausalLM.from_pretrained(path, config=config)
        model = model.half().eval().to(device)
    elif "llama" in model_name and "3" in model_name:
        config = AutoConfig.from_pretrained(path)
        config.compress_ratio = args.compress_ratio
        config.fixbudget = args.fixbudget
        config.budget = args.budget
        config.important_ratio = args.important_ratio
        config.pp_size = pp_size
        config.sink_size = args.sink_size
        config.recent_size = args.recent_size
        config.keyformer_mode = (args.keyformer_mode == 1)
        config.drop_ratio = args.drop_ratio
        config.preserve_layer = args.preserve_layer
        config.score_func = args.score_func
        config.compressor = args.compressor
        config.threshold = args.threshold
        config.n_subvec_per_head = args.n_subvec_per_head
        config.n_subbits = args.n_subbits
        config.fixthreshold = args.fixthreshold
        config.topr = args.topr
        config.gqa = (args.gqa == "True")
        config.max_iter = args.max_iter
        config.device = torch.device("cuda:0")
        config.mean_v_trick = (args.sparq_mean_v_trick == "True")
        config.recent_ratio = args.recent_ratio
        if args.enable_vq_cache:
            config.compress_ratio = args.compress_ratio
            config.important_ratio = args.important_ratio
        elif args.enable_h2o_cache:
            config.hh_ratio = args.important_ratio

        if config.compressor == "pq_search":
            config.max_seq_len = 130000
            config.cache_block_size = 128
            config.global_cache_size = 4096
            config.cache_topk = 32
            initialize_objects(config, model=model_name)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        if args.enable_vq_cache:
            model = VQLlama31ForCausalLM.from_pretrained(path, config=config, torch_dtype=torch.bfloat16)
        elif args.enable_h2o_cache:
            model = H2OLlamaForCausalLM.from_pretrained(path, config=config)
        model.patch(config)
        model = model.to(device).eval()
    elif "qwen" in model_name:
        config = AutoConfig.from_pretrained(path)
        config.compress_ratio = args.compress_ratio
        config.fixbudget = args.fixbudget
        config.budget = args.budget
        config.important_ratio = args.important_ratio
        config.pp_size = pp_size
        config.sink_size = args.sink_size
        config.recent_size = args.recent_size
        config.keyformer_mode = (args.keyformer_mode == 1)
        config.drop_ratio = args.drop_ratio
        config.preserve_layer = args.preserve_layer
        config.score_func = args.score_func
        config.compressor = args.compressor
        config.threshold = args.threshold
        config.n_subvec_per_head = args.n_subvec_per_head
        config.n_subbits = args.n_subbits
        config.fixthreshold = args.fixthreshold
        config.topr = args.topr
        config.gqa = (args.gqa == "True")
        config.max_iter = args.max_iter
        config.device = torch.device("cuda:0")
        config.mean_v_trick = (args.sparq_mean_v_trick == "True")
        config.recent_ratio = args.recent_ratio
        if args.enable_vq_cache:
            config.compress_ratio = args.compress_ratio
            config.important_ratio = args.important_ratio
        elif args.enable_h2o_cache:
            config.hh_ratio = args.important_ratio

        if config.compressor == "pq_search":
            config.max_seq_len = 130000
            config.cache_block_size = 128
            config.global_cache_size = 4096
            config.cache_topk = 32
            initialize_objects(config, model=model_name)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        if args.enable_vq_cache:
            model = VQQwen2ForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, config=config)
        model.patch(config)
        model = model.to(device).eval()
    elif "glm" in model_name:
        if VQGlmForCausalLM is None:
            raise ImportError("vq_method.glm_patch is required for GLM models")
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        ###
        config.attention_bias = config.add_qkv_bias
        config.attention_dropout = config.attention_dropout

        # head / dim
        config.head_dim = config.kv_channels
        config.hidden_size = config.hidden_size
        config.intermediate_size = config.ffn_hidden_size

        # heads
        config.num_attention_heads = config.num_attention_heads
        config.num_key_value_heads = config.multi_query_group_num

        # num layers
        config.num_hidden_layers = config.num_hidden_layers

        # seq length
        config.max_position_embeddings = config.seq_length

        # rope
        config.rope_theta = float(config.rope_ratio)

        # norm
        config.rms_norm_eps = config.layernorm_epsilon

        # vocab
        config.vocab_size = config.padded_vocab_size
        ###
        config.compress_ratio = args.compress_ratio
        config.fixbudget = args.fixbudget
        config.budget = args.budget
        config.important_ratio = args.important_ratio
        config.pp_size = pp_size
        config.sink_size = args.sink_size
        config.recent_size = args.recent_size
        config.keyformer_mode = (args.keyformer_mode == 1)
        config.drop_ratio = args.drop_ratio
        config.preserve_layer = args.preserve_layer
        config.score_func = args.score_func
        config.compressor = args.compressor
        config.threshold = args.threshold
        config.n_subvec_per_head = args.n_subvec_per_head
        config.n_subbits = args.n_subbits
        config.fixthreshold = args.fixthreshold
        config.topr = args.topr
        config.gqa = (args.gqa == "True")
        config.max_iter = args.max_iter
        config.device = torch.device("cuda:0")
        config.mean_v_trick = (args.sparq_mean_v_trick == "True")
        config.recent_ratio = args.recent_ratio
        if args.enable_vq_cache:
            config.compress_ratio = args.compress_ratio
            config.important_ratio = args.important_ratio
        elif args.enable_h2o_cache:
            config.hh_ratio = args.important_ratio

        if config.compressor == "pq_search":
            config.max_seq_len = 130000
            config.cache_block_size = 128
            config.global_cache_size = 4096
            config.cache_topk = 32
            initialize_objects(config, model=model_name)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if args.enable_vq_cache:
            model = VQGlmForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, config=config, trust_remote_code=True)
        model.patch(config)
        model = model.to(device).eval()
    else:
        # Fallback: plain HF causal LM
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        model = model.eval().to(device)

    return model, tokenizer


def get_pred(llm, message, data, tokenizer, out_path, args):
    # Some chat tokenizers ship without pad_token_id; make padding explicit.
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

    # Clean H2O cache between samples so previous-sample state doesn't leak
    if args.enable_h2o_cache:
        for name, m in llm.named_modules():
            if isinstance(m, H2OLlamaAttention):
                m._clean_cache()

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


if __name__ == "__main__":
    args = parse_args()

    seed_everything(42)

    # exactly one of vq-cache / h2o-cache should be enabled
    assert args.enable_vq_cache + args.enable_h2o_cache == 1, \
        "Enable exactly one of --enable_vq_cache / --enable_h2o_cache"

    pred_dir = args.save_dir
    os.makedirs(pred_dir, exist_ok=True)

    gsm8k_datas = load_dataset(pred_dir)

    device = torch.device("cuda:0")
    print(f"Loading model: {args.model_name} from {args.model}")
    llm, tokenizer = load_model_and_tokenizer(args, device, args.pp_size)
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

    if args.compressor == "pq_search":
        del_objects()

    print("All gsm8k evaluation done.")
