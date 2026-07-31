import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch
from snapkv.monkeypatch.monkeypatch import replace_llama, replace_qwen, replace_glm
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--compress_args_path', type=str, default=None, help="Path to the compress args")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--dataset', type=str, default="LongBench", help="The folder of dataset to evaluate on")
    parser.add_argument('--dataset_name', type=str, default='qasper', help="The name of dataset to evaluate on")
    parser.add_argument('--budget', type=int, default=None)
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt):
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                            add_generation_prompt=True, tokenize=False)  
    return prompt

@torch.inference_mode()
def get_pred_single_gpu(data, max_length, max_gen, 
                        prompt_format, dataset, dataset_name, model_name, 
                        model2path, output_path, 
                        compress=False, 
                        window_sizes = None,
                        max_capacity_prompts = None,
                        kernel_sizes = None,
                        pooling = None):
    # device = torch.device(f'cuda:{rank}')
    # device = model.device
    print(f"Loading model from {model2path[model_name]}...")
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device = "cuda", compress=compress)
    device = model.device
    
    if compress and "glm" in model_name.lower():
        from snapkv.monkeypatch.monkeypatch import replace_glm
        replace_glm(model)
        
    done_count = 0
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_count += 1

    remaining_data = data[done_count:]
    with open(output_path, "w", encoding="utf-8") as fout:
        for json_obj in tqdm(remaining_data):
            ############################################################################################################
            # load compress args
            if compress:
                import torch
                if hasattr(model, "model") and hasattr(model.model, "layers"):
                    model_layers = model.model.layers
                elif hasattr(model, "transformer") and hasattr(model.transformer, "encoder") and hasattr(model.transformer.encoder, "layers"):
                    model_layers = model.transformer.encoder.layers
                else:
                    raise ValueError("Could not find layers in model")
                layers = len(model_layers)
                # check if window_sizes is a list
                if not isinstance(window_sizes, list):
                    window_sizes = [window_sizes] * layers
                if not isinstance(max_capacity_prompts, list):
                    max_capacity_prompts = [max_capacity_prompts] * layers
                if not isinstance(kernel_sizes, list):
                    kernel_sizes = [kernel_sizes] * layers
                for i in range(layers):
                    attn = getattr(model_layers[i], "self_attn", getattr(model_layers[i], "self_attention", None))
                    attn.config.window_size = window_sizes[i]
                    attn.config.max_capacity_prompt = max_capacity_prompts[i]
                    attn.config.kernel_size = kernel_sizes[i]
                    attn.config.pooling = pooling
            ############################################################################################################
            if "ruler" in dataset:  
                prompt = prompt_format.format(**json_obj)
                answers = json_obj.get("outputs", [])
                all_classes = json_obj.get("all_classes", []) 
                length = json_obj.get("length", len(json_obj.get("input", "")))
            elif "LongBench" in dataset:
                prompt = prompt_format.format(**json_obj)
                answers = json_obj["answers"]
                all_classes = json_obj["all_classes"]
                length = json_obj["length"]
            
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            print(f"Dataset: {dataset_name} | Sample Index: {data.index(json_obj)} | Original Length: {len(tokenized_prompt)}")
            if len(tokenized_prompt) > max_length:
                 half = int(max_length/2)
                 prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            if "LongBench" in dataset and dataset_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
                prompt = build_chat(tokenizer, prompt)
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input.input_ids.shape[-1]
            if dataset_name == "samsum": 
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
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
                    min_length=context_length+1,
                )[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True).strip()
            with open(output_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": answers , "all_classes": all_classes, "length": length}, f, ensure_ascii=False)
                f.write('\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device, compress=False):

    tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
            path, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
            )

    model = model.eval()
    return model, tokenizer

def build_output_path(model, dataset_name, budget=None):
    base_output = "./output"
    sub_dir = ""

    if budget is not None:
        output_dir = os.path.join("budget", f"budget{budget}", model, sub_dir)
    else:
        output_dir = os.path.join(base_output, model, sub_dir)
    os.makedirs(output_dir, exist_ok=True)

    return os.path.join(output_dir, f"{dataset_name}.jsonl")
# has changed
if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    # world_size = torch.cuda.device_count()
    # mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    dataset = args.dataset
    dataset_name=args.dataset_name
    dataset_path=args.dataset+'/'+dataset_name+".jsonl"

    dataset = args.dataset
    # for dataset in datasets:
    if args.compress_args_path:
        compress_args = json.load(open(args.compress_args_path, "r"))
        compress = True
        write_model_name = model_name
        replace_llama()
        replace_qwen()
    else:
        compress = False
        compress_args = None
        write_model_name = model_name
    if "LongBench" in dataset:
        subset = f"{dataset_name}_e" if getattr(args, 'extended', False) else dataset_name
        print(f"Loading data from THUDM/LongBench ({subset}) ...")
        data = load_dataset("THUDM/LongBench", subset, split="test")
        prompt_format = dataset2prompt[dataset_name]
        max_gen = dataset2maxlen[dataset_name]
    elif "ruler" in dataset:
        data = load_dataset("json", data_files=dataset_path,split="train")
        prompt_format = dataset2prompt[dataset_name]  
        max_gen = dataset2maxlen[dataset_name]
    data_all = [data_sample for data_sample in data]
    output_path = build_output_path(
        args.model,
        args.dataset_name,
        budget=getattr(args, 'budget', None),
    )
    if compress_args is not None:
        get_pred_single_gpu(data_all, max_length, max_gen, prompt_format, dataset, dataset_name, model_name, model2path, output_path, compress,  **compress_args)
    else:
        get_pred_single_gpu(data_all, max_length, max_gen, prompt_format, dataset, dataset_name, model_name, model2path, output_path, compress)