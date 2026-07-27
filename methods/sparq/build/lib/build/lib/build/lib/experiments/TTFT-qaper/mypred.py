


import os
import json
import argparse
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from llminference.myexperiments import Sparsity,SparsityMethods

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, default="meta-llama/Llama-3.1-8B-Instruct", help="HuggingFace repo ")

    parser.add_argument("--task", type=str, required=True, help="LongBench  hotpotqa / qasper / gov_report ")
    parser.add_argument("--config_path", type=str, required=True, help=" dataset2prompt.jsondataset2maxlen.json ")
    parser.add_argument("--dataset_path", type=str, default=None, help=" LongBench  hotpotqa.jsonl / hotpotqa_e.jsonl ")
    parser.add_argument("--output_dir", type=Path, default="pred", help="")
    parser.add_argument("--model_name", type=str, default="", help=" chat  &  Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--e", action="store_true", help=" LongBench-E *_e.jsonl")
    parser.add_argument("--name", type=str, required=True, help=" ann ")
    parser.add_argument("--k", type=int, required=True, help=" ")
    parser.add_argument("--local_k", type=int, required=True, help="")
    parser.add_argument("--score", type=str, required=True, help=" ")
    parser.add_argument("--rank", type=int, required=True, help="")
    parser.add_argument("--reallocate_to_mean_value", type=bool, required=True, help="")
    parser.add_argument("--type", type=str, required=True, help="")


    return parser.parse_args()




def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_chat_prompt(prompt: str, model_name: str) -> str:
    """
     chat prompt 
    
    """
    name = model_name.lower()


    if "llama-2" in name or "llama2" in name:
        return f"[INST] {prompt} [/INST]"


    if "xgen" in name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        return header + f"### Human: {prompt}\n### Assistant:"


    if "internlm" in name:
        return f"<|User|>:{prompt}<eoh>\n<|Bot|>:"


    return prompt


def load_model_and_tokenizer(model_path: str, device: torch.device,args,out_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto" , 
        attn_implementation="flash_attention_2",
    )

    if args.type=="recall":
        sparq = Sparsity(name=args.name, k=args.k, local_k=args.local_k, rank=args.rank, score=args.score,
                 reallocate_to_mean_value=args.reallocate_to_mean_value,type="recall",recall_save_path=out_path)
    elif args.type=="topkrate":
        sparq = Sparsity(name=args.name, k=args.k, local_k=args.local_k, rank=args.rank, score=args.score,
                 reallocate_to_mean_value=args.reallocate_to_mean_value,type="topkrate",recall_save_path=out_path)
    else:
        sparq = Sparsity(name=args.name, k=args.k, local_k=args.local_k, rank=args.rank, score=args.score,
                 reallocate_to_mean_value=args.reallocate_to_mean_value)
    model = SparsityMethods.apply(sparq, model)

    if not torch.cuda.is_available():
        model.to(device)
    model.eval()
    return model, tokenizer




def ttft(
    model,
    tokenizer,
    data,
    task_name: str,
    model_name: str,
    max_length_ctx: int,
    max_new_tokens: int,
    prompt_format: str,
    device: torch.device,
    out_path: str,
):


    for json_obj in tqdm(data, desc=f"Task={task_name}"):

        prompt = prompt_format.format(**json_obj)


        tokenized = tokenizer(prompt, truncation=False, return_tensors="pt")
        input_ids = tokenized.input_ids[0]


        if len(input_ids) > max_length_ctx:
            half = max_length_ctx // 2
            kept_ids = torch.cat([input_ids[:half], input_ids[-half:]], dim=0)
            prompt = tokenizer.decode(kept_ids, skip_special_tokens=True)


        if task_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat_prompt(prompt, model_name)


        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        context_len = inputs.input_ids.shape[-1]
        
        assert torch.cuda.is_available()
        torch.cuda.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()


        output_ids = model.generate(
            **inputs,
            max_new_tokens=1,
            #attention_mask=enc["attention_mask"],
            num_beams=1,

            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            pad_token_id=tokenizer.eos_token_id,
        )[0]

        end_ev.record()
        torch.cuda.synchronize()
        ms = start_ev.elapsed_time(end_ev)

        gen_ids = output_ids[context_len:]
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        record = {
            "ttft":ms/1000.0,
            "pred": pred,
            "answers": json_obj["answers"],
            "all_classes": json_obj["all_classes"],
            "length": json_obj["length"],
        }

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

def throughput(
    model,
    tokenizer,
    data,
    task_name: str,
    model_name: str,
    max_length_ctx: int,
    max_new_tokens: int,
    prompt_format: str,
    device: torch.device,
    out_path: str,
):


    for json_obj in tqdm(data, desc=f"Task={task_name}"):

        prompt = prompt_format.format(**json_obj)


        tokenized = tokenizer(prompt, truncation=False, return_tensors="pt")
        input_ids = tokenized.input_ids[0]


        if len(input_ids) > max_length_ctx:
            half = max_length_ctx // 2
            kept_ids = torch.cat([input_ids[:half], input_ids[-half:]], dim=0)
            prompt = tokenizer.decode(kept_ids, skip_special_tokens=True)


        if task_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat_prompt(prompt, model_name)


        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        context_len = inputs.input_ids.shape[-1]
        
        assert torch.cuda.is_available()
        torch.cuda.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()


        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            #attention_mask=enc["attention_mask"],
            num_beams=1,

            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            pad_token_id=tokenizer.eos_token_id,
        )[0]

        end_ev.record()
        torch.cuda.synchronize()
        ms = start_ev.elapsed_time(end_ev)
    

        gen_ids = output_ids[context_len:]
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        new_token_num = len(gen_ids)

        throughput = new_token_num / (ms / 1000.0)
        record = {
            "throughput":throughput,
            "token_num":new_token_num,
            "total_time":ms/1000.0,
            "pred": pred,
            "answers": json_obj["answers"],
            "all_classes": json_obj["all_classes"],
            "length": json_obj["length"],
        }

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def recall(
    model,
    tokenizer,
    data,
    task_name: str,
    model_name: str,
    max_length_ctx: int,
    max_new_tokens: int,
    prompt_format: str,
    device: torch.device,
    out_path: str,
):


    for json_obj in tqdm(data, desc=f"Task={task_name}"):

        prompt = prompt_format.format(**json_obj)


        tokenized = tokenizer(prompt, truncation=False, return_tensors="pt")
        input_ids = tokenized.input_ids[0]


        if len(input_ids) > max_length_ctx:
            half = max_length_ctx // 2
            kept_ids = torch.cat([input_ids[:half], input_ids[-half:]], dim=0)
            prompt = tokenizer.decode(kept_ids, skip_special_tokens=True)


        if task_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat_prompt(prompt, model_name)


        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        context_len = inputs.input_ids.shape[-1]
        
        assert torch.cuda.is_available()
        torch.cuda.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()


        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            pad_token_id=tokenizer.eos_token_id,
        )[0]

        end_ev.record()
        torch.cuda.synchronize()
        ms = start_ev.elapsed_time(end_ev)
    

        gen_ids = output_ids[context_len:]
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        new_token_num = len(gen_ids)

    


def main():
    args = parse_args()
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name_for_prompt = args.model_name if args.model_name else args.model_path


    dataset2prompt = json.load(open(os.path.join(args.config_path, "dataset2prompt.json"), "r"))
    dataset2maxlen = json.load(open(os.path.join(args.config_path, "dataset2maxlen.json"), "r"))


    model2maxlen_path = os.path.join(args.config_path, "model2maxlen.json")
    if os.path.exists(model2maxlen_path):
        model2maxlen = json.load(open(model2maxlen_path, "r"))
        max_length_ctx = model2maxlen.get(model_name_for_prompt, 16384)
    else:
        max_length_ctx = 16384

    task = args.task
    assert task in dataset2prompt, f"{task} not found in dataset2prompt.json"

    prompt_format = dataset2prompt[task]
    max_new_tokens = dataset2maxlen[task]


    if args.type=="TTFT":
        max_new_tokens=1


    if "Llama" in args.model_name:
        modelName="Llama"
    elif "Qwen" in args.model_name:
        modelName="Qwen"
    #suffix = f"{args.model_name}-{args.name}-k{args.k}-local{args.local_k}-r{args.rank}"
    suffix = f"{modelName}-{args.name}-k{args.k}-local{args.local_k}-r{args.rank}"
    pred_file = args.output_dir/ f'{suffix}.jsonl'
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    out_path = pred_file


    model, tokenizer = load_model_and_tokenizer(args.model_path, device,args,out_path)


    if args.dataset_path:

        data_file = args.dataset_path
        assert os.path.exists(data_file), f": {data_file}"

        dataset_dict = load_dataset("json", data_files={"test": data_file})
        data = dataset_dict["test"]





    if args.type=="TTFT":
        ttft(
            model=model,
            tokenizer=tokenizer,
            data=data,
            task_name=task,
            model_name=model_name_for_prompt,
            max_length_ctx=max_length_ctx,
            max_new_tokens=max_new_tokens,
            prompt_format=prompt_format,
            device=device,
            out_path=out_path,
        )
    elif args.type=="THROUGHPUT":
        throughput(
            model=model,
            tokenizer=tokenizer,
            data=data,
            task_name=task,
            model_name=model_name_for_prompt,
            max_length_ctx=max_length_ctx,
            max_new_tokens=max_new_tokens,
            prompt_format=prompt_format,
            device=device,
            out_path=out_path,
        )
    elif args.type=="recall" or args.type=="topkrate":
        recall(
            model=model,
            tokenizer=tokenizer,
            data=data,
            task_name=task,
            model_name=model_name_for_prompt,
            max_length_ctx=max_length_ctx,
            max_new_tokens=max_new_tokens,
            prompt_format=prompt_format,
            device=device,
            out_path=out_path,
        )


if __name__ == "__main__":
    main()
