"""
Full prediction script for LongBench evaluation.
Uses standard Transformer model without PQ compression.
"""

import os
import json
import torch
import numpy as np
import random
import argparse
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=[
        "llama-7b", "llama2-7b-chat-4k", "llama2-7b-32K", "mistral-7b-Instruct-32k", "llama-3.1",
        "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k",
        "vicuna-v1.5-7b-16k", "qwen-2.5-7b"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--exp_name", type=str, default="default")
    return parser.parse_args()


def build_chat(tokenizer, prompt, model_name):
    """Build chat format prompt for different models."""
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                          add_generation_prompt=True, tokenize=False)
    return prompt


def post_process(response, model_name):
    """Post-process model output."""
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_path, model_name, device):
    """Load model and tokenizer without any PQ patches."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use bfloat16 for efficiency
    # Use flash_attention_2 if available
    attn_impl = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=attn_impl,
            use_cache=True,
        )
        print(f"Model loaded with {attn_impl}")
    except Exception as e:
        print(f"Failed to load with {attn_impl}: {e}")
        print("Falling back to standard attention")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    model = model.eval().to(device)

    return model, tokenizer


def get_max_length(model_name):
    """Get max sequence length for different models."""
    maxlen_map = {
        "mistral-7b-Instruct-32k": 32000,
        "llama-3.1": 130000,
        "qwen-2.5-7b": 130000,
        "llama-7b": 4096,
        "llama2-7b-chat-4k": 4096,
        "llama2-7b-32K": 32768,
        "xgen-7b-8k": 8192,
        "internlm-7b-8k": 8192,
        "chatglm2-6b": 8192,
        "chatglm2-6b-32k": 32768,
        "chatglm3-6b-32k": 32768,
        "vicuna-v1.5-7b-16k": 16384,
    }
    return maxlen_map.get(model_name, 4096)


def get_dataset_list(args):
    """Get list of datasets to evaluate."""
    if args.e:
        return ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
                "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        return ["qasper"]
        return ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
                "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht",
                "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
                


def get_prompt_format(dataset):
    """Get prompt format for dataset."""
    dataset2prompt = {
        "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
        "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
        "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
        "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
        "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
        "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
        "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
        "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
        "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
        "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
        "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
        "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
        "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
        "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
        "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
    }
    return dataset2prompt.get(dataset, "{context}\n{input}")


def get_max_gen_length(dataset):
    """Get max generation length for different datasets."""
    dataset2maxlen = {
        "narrativeqa": 128,
        "qasper": 32,
        "multifieldqa_en": 32,
        "multifieldqa_zh": 32,
        "hotpotqa": 32,
        "2wikimqa": 32,
        "musique": 32,
        "dureader": 64,
        "gov_report": 512,
        "qmsum": 512,
        "multi_news": 512,
        "vcsum": 512,
        "trec": 64,
        "triviaqa": 32,
        "samsum": 128,
        "lsht": 64,
        "passage_count": 32,
        "passage_retrieval_en": 32,
        "passage_retrieval_zh": 32,
        "lcc": 64,
        "repobench-p": 64,
    }
    return dataset2maxlen.get(dataset, 128)


def predict(model, tokenizer, device, data, max_length, max_gen, prompt_format, dataset, model_name, out_path):
    """Run prediction on dataset."""
    line_num = 0
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            while True:
                l = f.readline()
                if l == "":
                    break
                line_num += 1

    for i, json_obj in tqdm(enumerate(data)):
        if i < line_num:
            continue

        # Format prompt
        prompt = prompt_format.format(**json_obj)
        # prompt = prompt[-2048:]
        # Tokenize
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt").input_ids[0]
        original_token_cnt = len(tokenized_prompt)

        # Truncate if needed
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                     tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            original_token_cnt = max_length

        # Build chat format for applicable datasets
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)

        # Prepare input
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        # Generate
        with torch.no_grad():
            begin_gen = time.perf_counter()
            if dataset == "samsum":
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length + 1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = model.generate(
                    input_ids=input.input_ids,
                    attention_mask=input.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]
            end_gen = time.perf_counter()
            print(torch.cuda.memory_allocated() / 1024**2)
            print(torch.cuda.memory_reserved() / 1024**2)
        # Decode output
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)

        # Save result
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj.get("all_classes", []),
                "length": json_obj["length"],
                "request_time": {"batch_time": end_gen - begin_gen, "batch_size": 1},
                "input_tokens": int(original_token_cnt)
            }, f, ensure_ascii=False)
            f.write('\n')


def main():
    seed_everything(42)
    args = parse_args()

    # Load config
    model2path = json.load(open("config/model2path.json", "r"))
    model_name = args.model

    if model_name not in model2path:
        raise ValueError(f"Model {model_name} not found in config/model2path.json")

    model_path = model2path[model_name]
    max_length = args.max_length if args.max_length else get_max_length(model_name)
    datasets = get_dataset_list(args)

    device = torch.device("cuda:0")

    print(f"Loading model: {model_name} from {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path, model_name, device)
    print(f"Model loaded successfully")

    for dataset in datasets:
        print(f"Evaluating dataset: {dataset}")

        # Load data
        if args.e:
            data = load_dataset('./data', f"{dataset}_e", split='test')
        else:
            data = load_dataset('json', data_files='./data/' + dataset + '.jsonl', split='train')

        # Output path - use full.json
        if args.e:
            out_dir = f"pred_e/{model_name}"
        else:
            out_dir = f"pred/{model_name}/{dataset}/{args.exp_name}"

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "full.jsonl")

        prompt_format = get_prompt_format(dataset)
        max_gen = get_max_gen_length(dataset)
        data_all = [data_sample for data_sample in data]

        predict(model, tokenizer, device, data_all, max_length, max_gen,
                prompt_format, dataset, model_name, out_path)

        print(f"Dataset {dataset} done, output: {out_path}")

    print("All evaluation done.")


if __name__ == '__main__':
    main()
