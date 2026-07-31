import os
import sys
from pathlib import Path
import json
import random
import argparse

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import yaml
import importlib
import math
import traceback

def read_manifest(manifest_path):
    """Read JSONL file"""
    data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data



current_file_dir = Path(__file__).resolve().parent

headkv_parent_dir = current_file_dir.parent.parent  

if str(headkv_parent_dir) not in sys.path:
    sys.path.insert(0, str(headkv_parent_dir))

important_head_dir = headkv_parent_dir / "Important_Head"
if str(important_head_dir) not in sys.path:
    sys.path.insert(0, str(important_head_dir))
"""
Add a new task (required arguments):

TASK_NAME: {
    'tokens_to_generate': how many tokens we want to generate.
    'template': the template with at least {context} and {query}.
}
"""

TASKS = {
    'niah': {
        'tokens_to_generate': 128,
        'template': """Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?""",
        'answer_prefix': """ The special magic {type_needle_v} for {query} mentioned in the provided text are"""
    },
    
    'variable_tracking': {
        'tokens_to_generate': 30,
        'template': """Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\nQuestion: Find all variables that are assigned the value {query} in the text above.""",
        'answer_prefix': """ Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assgined the value {query}, they are: """
    },
    
    'common_words_extraction': {
        'tokens_to_generate': 120,
        'template': """Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n{context}\nQuestion: What are the 10 most common words in the above list?""",
        'answer_prefix': """ Answer: The top 10 words that appear most often in the list are:"""
    },
    
    'freq_words_extraction' : {
        'tokens_to_generate': 50,
        'template': """Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. {context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text?""",
        'answer_prefix': """ Answer: According to the coded text above, the three most frequently appeared words are:"""
    },

    'qa': {
        'tokens_to_generate': 32, 
        'template': """Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query}""",
        'answer_prefix': """ Answer:""",
    },
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def main(args):
    
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    

    tasks_base = TASKS
    with open(os.path.join(curr_folder, f"./{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in config_tasks.yaml')

    config = tasks_customized.get(args.task)
    config.update(tasks_base[config['task']])

    task_file = args.task_data
    pred_file = args.save_dir / f'{args.task}.jsonl'
        
    print(f'Predict {args.task} \nfrom {task_file}\nto {pred_file}')
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if os.path.exists(pred_file): 
        pred_index = [sample['index'] for sample in read_manifest(pred_file)]
        data = [sample for sample in read_manifest(task_file) if sample['index'] not in pred_index]
    else:
        data = read_manifest(task_file)
    
    if args.steps > 0:
        args.max_num_examples = args.steps

    if args.max_num_examples > 0:
        data = data[:args.max_num_examples]
    
  
    model.model.config.window_size = args.window_size
    model.model.config.base_capacity = args.max_capacity_prompts
    model.model.config.head_choice = args.head_choice
    model.model.config.beta = args.beta
    model.model.config.temp = args.temp
    
    model.model.config.kernel_size = args.kernel_size
    model.model.config.skip = args.skip
    model.model.config.normalize = args.normalize
    model.model.config.pooling = args.pooling
    model.model.config.floor = args.floor
    model.model.config.pyram_beta = args.pyram_beta
    
    if args.method.lower() in ["pyramidkv", "snapkv", "h2o", "streamingllm"]:
        layers = len(model.model.layers)
        max_capacity_list = [args.max_capacity_prompts] * layers
        window_sizes_list = [args.window_size] * layers
        kernel_sizes_list = [args.kernel_size] * layers
        
        for i in range(layers):
            model.model.layers[i].self_attn.config.window_size = window_sizes_list[i]
            model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_list[i]
            model.model.layers[i].self_attn.config.kernel_size = kernel_sizes_list[i]
            model.model.layers[i].self_attn.config.pooling = args.pooling

    def get_output(idx, index, input, outputs, others, truncation, length):
        
        tokenized_prompts = tokenizer(input, return_tensors="pt", add_special_tokens=True).to('cuda')
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask
        max_new_tokens=config["tokens_to_generate"]
        context_length = batch_input_ids.shape[-1] # seq_len
        
        generation_kwargs = dict(
            **tokenized_prompts,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            min_length=context_length + 1,
            eos_token_id=[tokenizer.eos_token_id],
        )

        output_ids = model.generate(**generation_kwargs)[0]
        num_gen_tokens = len(output_ids) - context_length

        generated_text = tokenizer.decode(output_ids[context_length:], skip_special_tokens=True)
        
        outputs_parallel[idx] = {
            'index': index,
            'pred': generated_text,
            'input': input,
            'outputs': outputs,
            'others': others,
            'truncation': truncation,
            'length': length,
        }
        print(f"Index: {index} | Pred: {generated_text}")

    
    outputs_parallel = [{} for _ in range(len(data))]
    
    # setting buffering=1 to force to dump the output after every line
    with open(pred_file, 'at', encoding="utf-8", buffering=1) as fout:
        for idx, data_point in tqdm(enumerate(data), total=len(data)):

            get_output(idx=idx,
                    index=data_point['index'],
                    input=data_point['input'],
                    outputs=data_point['outputs'], 
                    others=data_point.get('others', {}),
                    truncation=data_point.get('truncation', -1),
                    length=data_point.get('length', -1),
                    
                    )
            
            if len(outputs_parallel[idx]) > 0:
                        fout.write(json.dumps(outputs_parallel[idx]) + '\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--task_data", type=Path, required=True, help='path to load the dataset jsonl files')
    parser.add_argument("--save_dir", type=Path, required=True, help='path to save the prediction jsonl files')
    parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
    parser.add_argument("--task", type=str, required=True, help='Options: tasks in benchmark')

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")

    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str,  default=None)
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")

    parser.add_argument("--head_choice", type=str, default='random', choices=['random', 'copy', 'reason'])
    parser.add_argument('--beta', type=float, default=1.5)
    parser.add_argument('--temp', type=float, default=1.0)

    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1, help="")
    parser.add_argument("--max_num_examples", type=int, default=-1, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task (alias for max_num_examples).")

    parser.add_argument("--floor", type=float, default=0.2, help="Floor ratio for AdativeKV")
    parser.add_argument("--skip", type=int, default=0, help="Number of layers to skip for AdativeKV")
    parser.add_argument("--normalize", type=bool, default=True, help="Whether to normalize attention scores in AdativeKV")
    parser.add_argument("--pyram_beta", type=int, default=20, help="Beta parameter for PyramidKV")
    parser.add_argument("--window_size", type=int, default=8, help="Window size for PyramidKV/SnapKV/H2O")
    parser.add_argument("--kernel_size", type=int, default=7, help="Kernel size for PyramidKV/SnapKV")
    parser.add_argument("--pooling", type=str, default="maxpool", help="Pooling method for PyramidKV/SnapKV")
    
    parser.add_argument(
        "--tokenizer_path", type=str, default=None, help="tokenizer path"
    )

    args = parser.parse_args()

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    set_seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left"
    )

    if args.method.lower() != 'fullkv':
        from headkv.monkeypatch import replace_llama, replace_qwen2
        replace_llama(args.method)
        replace_qwen2(args.method)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation
    )

    model.eval()
    torch.cuda.empty_cache()
    
    main(args)
