import os, time
from requests.exceptions import ProxyError, SSLError
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
    AutoModelForCausalLM,
)
from transformers.cache_utils import DynamicCache
from tqdm import tqdm
import numpy as np
import random
import argparse
from accuracy.patch import parse_common_args, enable_attention_eval, get_config_output_affix
from accuracy.cluster_attention import cluster_reset


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser = parse_common_args(parser)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        help="one or more dataset names; defaults to the standard LongBench subset",
        default=None,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data_idx", type=int, default=None)
    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    # if "glm4" in model_name or "intern" in model_name or "llama3" in model_name or "wen2" in model_name:
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                            add_generation_prompt=True, tokenize=False)
    return prompt


def post_process(pred, model_name):
    if "glm4" in model_name:
        pred = pred.split("Assistant:")[-1]
    if "wen2" in model_name:
        pred = pred.split("<|im_end|>")[-1].split("<|im_start|>")[-1].strip()
    return pred


def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    model_name,
    out_path=None,
):
    preds = []
    # i = 0
    for _, json_obj in enumerate(tqdm(data)):
        # if i >= 3:
        #     break
        # i += 1
        if args.cluster:
            cluster_reset(model)
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if "glm4" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
        
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
        context_length = input.input_ids.shape[-1]

        with torch.no_grad():
            if dataset == "samsum":
                # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        pred_item = {
            "pred": pred,
            "answers": json_obj["answers"],
            "all_classes": json_obj["all_classes"],
            "length": json_obj["length"],
        }
        preds.append(pred_item)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj.get("all_classes", []),
                "length": json_obj["length"],
            }, f, ensure_ascii=False)
            f.write('\n')
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device):
    if "intern" in model_name or "qwen" in model_name or "glm4" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map="auto", low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2", use_cache=True
        ).to(device)
    elif "llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2", use_cache=True
        )
    else:
        assert False
    model = model.eval()

    if args.quest or args.cluster:
        enable_attention_eval(model_name, model, args)

    return model, tokenizer


def load_model_with_retry(model_path, model_name, device, retries=3, delay=1):
    for attempt in range(retries):
        try:
            model, tokenizer = load_model_and_tokenizer(model_path, model_name, device)
            return model, tokenizer
        except (ProxyError, SSLError) as e:
            print(f"Attempt {attempt + 1} failed due to network error: {e}")
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                raise  # Re-raise the last exception if all retries fail

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    assert not (args.quest and args.cluster)     # cannot be enabled at same time
    if args.dist_t != "cosine":
        assert args.debug

    model2path = json.load(open("../config/model2path.json", "r"))
    model2maxlen = json.load(open("../config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    # define your model
    model, tokenizer = load_model_with_retry(
        model2path[model_name], model_name, device
    )
    max_length = model2maxlen[model_name]
    default_datasets = [
        "narrativeqa", "qasper",
        "2wikimqa", "musique",
        "gov_report", "multi_news",
        "triviaqa", "samsum",
        "passage_count", "passage_retrieval_en",
        "lcc", "repobench-p",
    ]
    datasets = args.task if args.task is not None else default_datasets
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    if not os.path.exists("debug"):
        os.makedirs("debug")
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        try:
            if args.e:
                # data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
                data_file = f"data/{dataset}_e.jsonl"
                data = load_dataset(
                    "json",
                    data_files=data_file,
                    split="train"
                )
                res_dir = "debug" if args.debug or args.data_idx is not None else "pred_e"
                if not os.path.exists(f"{res_dir}/{model_name}"):
                    os.makedirs(f"{res_dir}/{model_name}")
                out_path = f"{res_dir}/{model_name}/{dataset}.jsonl"
                if args.quest:
                    out_path = f"{res_dir}/{model_name}/{dataset}-{args.token_budget}.jsonl"
                else:
                    out_path = f"{res_dir}/{model_name}/{dataset}.jsonl"
            else:
                data = load_dataset("THUDM/LongBench", f"{dataset}", split="test")
                # data_file = f"data/{dataset}.jsonl"
                # data = load_dataset(
                #     "json",
                #     data_files=data_file,
                #     split="train"
                # )
                res_dir = "debug" if args.debug or args.data_idx is not None else "pred"
                if not os.path.exists(f"{res_dir}/{model_name}"):
                    os.makedirs(f"{res_dir}/{model_name}")
                config_affix = get_config_output_affix(args)
                out_path = f"{res_dir}/{model_name}/{dataset}{config_affix}.jsonl"
            prompt_format = dataset2prompt[dataset]
            max_gen = dataset2maxlen[dataset]
            if args.debug:
                data = data.select(range(1))
            elif args.data_idx is not None:
                data = data.select(range(args.data_idx, args.data_idx+1))
            preds = get_pred(
                model,
                tokenizer,
                data,
                max_length,
                max_gen,
                prompt_format,
                dataset,
                model_name,
                out_path,
            )
        except Exception as e:
            print(f"Error processing dataset '{dataset}': {e}")
            print("Continuing to next dataset...")
            continue
