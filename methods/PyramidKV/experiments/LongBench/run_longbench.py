import os
import sys
import json
import random
import argparse
import gc

import numpy as np
from tqdm import tqdm
from datasets import load_dataset

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

datasets = [
    'narrativeqa',
    'qasper',
    'trec',
    'lcc',
]  
datasets = [
    'narrativeqa',
    'qasper',
    '2wikimqa',
    'musique',
    'gov_report',
    'multi_news',
    'triviaqa',
    'samsum',
    'passage_count',
    'passage_retrieval_en',
    'lcc',
    'repobench-p',
]      
dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
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
    "repobench-p": 64
}

model2prompt = {
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



model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 130000,
    "llama-3": 130000,
    "mistral": 31500,
    "qwen2": 31500,
    "qwen2.5": 31500,
    "glm": 1000000,
    "chatglm": 1000000
}



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat(tokenizer, prompt, model_name):
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                            add_generation_prompt=True, tokenize=False)
    return prompt


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "encoder") and hasattr(model.transformer.encoder, "layers"):
        return model.transformer.encoder.layers
    raise ValueError("Unsupported model architecture for layer traversal")


def get_self_attention_module(layer):
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if hasattr(layer, "self_attention"):
        return layer.self_attention
    raise ValueError("Unsupported layer attention module")


def get_config_holder(model):
    if hasattr(model, "model") and hasattr(model.model, "config"):
        return model.model.config
    if hasattr(model, "transformer") and hasattr(model.transformer, "config"):
        return model.transformer.config
    return model.config


# def build_prompt(prompt, dataset):

#     SYSTEM_PROMPT = model2prompt[dataset]

#     prompt = f"<<SYS>>\n {SYSTEM_PROMPT} \n<</SYS>>\n\n{prompt}"
#     return prompt

def main(args):


    print("Loading data...")

    test_data = []

    prompts = []
    inputs = []
    contexts = []
    answerss = []
    lengths = []
    datasets = []
    languages = []
    all_classess = []
    _ids = []

    input_max_len = 0

    model_path = args.model_path.lower()
    model_name = args.model_path.split("/")[-1]


    model_max_len = None
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]
            break
    if model_max_len is None:
        model_max_len = 32768



    output_max_len = dataset2maxlen[args.dataset]

    prompt_printed = False
    dataset_hf = load_dataset("THUDM/LongBench", args.dataset, split="test")
    for example_raw in dataset_hf:
        example = {
            "input": example_raw["input"],
            "context": example_raw["context"],
            "answers": example_raw["answers"],
            "length": example_raw["length"],
            "dataset": example_raw["dataset"],
            "language": example_raw["language"],
            "all_classes": example_raw["all_classes"],
            "_id": example_raw["_id"],
        }
        length = example["length"]
        if length > input_max_len: input_max_len = length

        template = model2prompt[args.dataset]
        prompt = template.format(**example)

        if args.dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
            if not prompt_printed:
                print(f"[build_chat] dataset={args.dataset}, model={model_name}, prompt_head={prompt[:200]!r}")
                prompt_printed = True

        example["prompt"] = prompt

        test_data.append(example)

    print(f"Max Length is {input_max_len}")

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        if args.sample_method == "random":
            test_data = random.sample(test_data, args.max_num_examples)
        elif args.sample_method == "topk":
            test_data = test_data[:args.max_num_examples]


    for example in test_data:

        prompts.append(example["prompt"])
        inputs.append(example["input"])
        contexts.append(example["context"])
        answerss.append(example["answers"])
        lengths.append(example["length"])
        datasets.append(example["dataset"])
        languages.append(example["language"])
        all_classess.append(example["all_classes"])
        _ids.append(example["_id"])

    print("Finish loading model and tokenizer")

    model_name = model_path.split("/")[-1]

    model_save_dir = os.path.join(args.save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    output_file = os.path.join(model_save_dir, f"{args.dataset}-{args.method}_{args.max_capacity_prompts}.json")
    

    start_idx = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if len(lines) > 0:
            start_idx = len(lines)
            print(f"[断点续传] {args.dataset} 已处理 {start_idx}/{len(prompts)} 个样本，继续从第 {start_idx} 个开始")
    
    if start_idx >= len(prompts):
        print(f"[跳过] {args.dataset} 已完成全部 {len(prompts)} 个样本")
        return
    
    fout = open(output_file, "a" if start_idx > 0 else "w")

    for i in tqdm(range(start_idx, len(prompts), args.eval_batch_size)):

        batch_prompts = prompts[i:i+args.eval_batch_size]
        batch_inputs = inputs[i:i+args.eval_batch_size]
        batch_contexts = contexts[i:i+args.eval_batch_size]
        batch_answerss = answerss[i:i+args.eval_batch_size]
        batch_lengths = lengths[i:i+args.eval_batch_size]

        batch_datasets = datasets[i:i+args.eval_batch_size]
        batch_languages = languages[i:i+args.eval_batch_size]
        batch_all_classess = all_classess[i:i+args.eval_batch_size]
        batch__ids = _ids[i:i+args.eval_batch_size]

        tokenized_prompts = tokenizer(
            batch_prompts,
            padding="longest",
            truncation=True,
            max_length=model_max_len,
            return_tensors="pt",
            add_special_tokens=True,
        ).to('cuda')
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        # # default to True
        # if args.method == "DynamicKV":
        #     args.output_attentions = True
        # else:
        #     args.output_attentions=False

        if args.max_capacity_prompts != -1:
            max_capacity_prompts = args.max_capacity_prompts
        elif args.max_capacity_prompts_ratio != -1:
            max_capacity_prompts = round(batch_input_ids.shape[1] * args.max_capacity_prompts_ratio)


        if args.method != "FullKV":
            if args.method.lower() in ["snapkv","pyramidkv","h2o"]:
                window_sizes = 32
            elif args.method.lower() in ["streamingllm"]:
                window_sizes = max_capacity_prompts - 4

            kernel_sizes = 7
            pooling = "maxpool"

            layers_list = get_transformer_layers(model)
            layers = len(layers_list)

            config_holder = get_config_holder(model)
            config_holder.base_capacity = max_capacity_prompts if not isinstance(max_capacity_prompts, list) else max_capacity_prompts[0]
            config_holder.window_size = window_sizes if not isinstance(window_sizes, list) else window_sizes[0]
            config_holder.head_choice = 'random'
            
            if not isinstance(window_sizes, list):
                window_sizes = [window_sizes] * layers
            if not isinstance(max_capacity_prompts, list):
                max_capacity_prompts = [max_capacity_prompts] * layers
            if not isinstance(kernel_sizes, list):
                kernel_sizes = [kernel_sizes] * layers
            for i in range(layers):
                attn_module = get_self_attention_module(layers_list[i])
                if not hasattr(attn_module, "config"):
                    attn_module.config = model.config
                attn_module.config.window_size = window_sizes[i]
                attn_module.config.max_capacity_prompt = max_capacity_prompts[i]
                attn_module.config.kernel_size = kernel_sizes[i]
                attn_module.config.pooling = pooling
                attn_module.config.pyram_beta = args.pyram_beta

        context_length = batch_input_ids.shape[-1]

        generation_kwargs = dict(
            output_attentions=args.output_attentions,
            max_new_tokens=output_max_len,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            top_p=None,
            min_length=context_length + 1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        if "glm" in model_path or "chatglm" in model_path:

            generation_kwargs["eos_token_id"] = model.generation_config.eos_token_id
        else:
            generation_kwargs["num_logits_to_keep"] = 1

        try:
            with torch.inference_mode():
                output = model.generate(
                    **tokenized_prompts,
                    **generation_kwargs,
                )

            generated_token_ids = [output[idx][context_length:] for idx in range(output.shape[0])]
            batch_outputs = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
            batch_generations = batch_outputs
        except torch.OutOfMemoryError:
            print(f"[OOM] dataset={args.dataset}, sample_idx={i}, input_len={context_length}; 直接抛出")
            raise
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[OOM-RuntimeError] dataset={args.dataset}, sample_idx={i}, input_len={context_length}; 直接抛出")
                raise
            else:
                raise
        finally:
            if "output" in locals():
                del output
            if "generated_token_ids" in locals():
                del generated_token_ids
            if "batch_outputs" in locals():
                del batch_outputs
            del tokenized_prompts, batch_input_ids, attention_mask
            gc.collect()
            torch.cuda.empty_cache()

        for j in range(len(batch_prompts)):

            result = {
                "pred": batch_generations[j],
                "answers": batch_answerss[j],
                "all_classes": batch_all_classess[j],
                "length": batch_lengths[j]
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")



if __name__ == "__main__":

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")

    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")

    parser.add_argument("--max_new_tokens", type=int, default=None, help="")

    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")

    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str,  default=None)
    parser.add_argument("--max_capacity_prompts", type=int, default=1024, help="")
    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1, help="")
    parser.add_argument("--pyram_beta", type=int, default=20, help="PyramidKV beta parameter")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--tasks", type=str, default=None, help="comma-separated list of tasks to run (overrides hardcoded list)")

    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )

    args = parser.parse_args()

    set_seed(args.seed)


    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=True,
        padding_side="left"
    )


    from pyramidkv.monkeypatch import replace_llama, replace_qwen2, replace_chatglm
    replace_llama(args.method.lower())
    replace_qwen2(args.method.lower())

    if 'llama-3' in args.model_path.lower():
        revision = '5206a32e0bd3067aef1ce90f5528ade7d866253f'
    elif 'mistral' in args.model_path.lower():
        revision = 'b70aa86578567ba3301b21c8a27bea4e8f6d6d61'
    elif 'qwen2' in args.model_path.lower():
        revision = None
    elif 'glm' in args.model_path.lower() or 'chatglm' in args.model_path.lower():
        revision = None
    else:
        revision = None

    if 'glm' in args.model_path.lower() or 'chatglm' in args.model_path.lower():
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        config.use_cache = args.use_cache
        config._attn_implementation = args.attn_implementation
        chatglm_cls = get_class_from_dynamic_module(
            "modeling_chatglm.ChatGLMForConditionalGeneration",
            args.model_path,
            revision=revision,
            local_files_only=True,
        )
        model = chatglm_cls.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            local_files_only=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=args.use_cache,
            attn_implementation=args.attn_implementation
        )

    # replace_chatglm(args.method.lower(), model)




    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id



    if os.getenv("PYRAMIDKV_DEBUG_CACHE_SHAPES", "0") == "1":
        for layer in get_transformer_layers(model):
            attn = get_self_attention_module(layer)
            attn.debug_cache_shapes = True

    model.eval()

    save_dir = args.save_dir


    max_capacity_prompts = args.max_capacity_prompts






    if args.tasks:
        run_datasets = [t.strip() for t in args.tasks.split(',')]
        print(f"[LongBench] Using filtered tasks: {run_datasets}")
    else:
        run_datasets = [args.dataset] if args.dataset else datasets

    for idx, dataset in enumerate(run_datasets):

        print(f"Working on max_capacity_prompts {args.max_capacity_prompts} dataset {dataset} - {idx}/{len(run_datasets)}")

        args.dataset = dataset

        main(args)
