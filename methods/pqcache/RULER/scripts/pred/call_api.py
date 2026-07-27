# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl: 
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""

import argparse
import json
import yaml
import os
import sys
import threading
import importlib
import math
import time
from tqdm import tqdm
from pathlib import Path
import traceback

TOPP_SAVE_TOPK = os.environ.get("TOPP_SAVE_TOPK")
TOPK_SAVE_TOPP = os.environ.get("TOPK_SAVE_TOPP")
# from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
def read_manifest(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

SERVER_TYPES = (
    'trtllm',
    'vllm',
    'sglang',
    'openai',
    'gemini',
    'hf',
    'mamba',
    'pq',
)


class ServerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.server_type = values


parser = argparse.ArgumentParser()
# Data
parser.add_argument("--data_dir", type=Path, required=True, help='path to load the dataset jsonl files')
parser.add_argument("--save_dir", type=Path, required=True, help='path to save the prediction jsonl files')
parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
parser.add_argument("--task", type=str, required=True, help='Options: tasks in benchmark')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--chunk_idx", type=int, default=0, help='index of current split chunk')
parser.add_argument("--chunk_amount", type=int, default=1, help='size of split chunk')

# Server
parser.add_argument("--server_type", default='nemo', action=ServerAction, choices=SERVER_TYPES)
parser.add_argument("--server_host", type=str, default='127.0.0.1')
parser.add_argument("--server_port", type=str, default='5000')
parser.add_argument("--ssh_server", type=str)
parser.add_argument("--ssh_key_path", type=str)
parser.add_argument("--model_name_or_path", type=str, default='gpt-3.5-turbo',
                    help='supported models from OpenAI or HF (provide a key or a local path to the checkpoint)')
# PQ-specific arguments
parser.add_argument("--pq_compress_ratio", type=float, default=0.1,
                    help='KV cache compression ratio')
parser.add_argument("--pq_fixbudget", action='store_true', default=False,
                    help='Enable fixed budget mode')
parser.add_argument("--pq_budget", type=int, default=1024,
                    help='Fixed budget size (used when fixbudget is enabled)')
parser.add_argument("--pq_important_ratio", type=float, default=0.5,
                    help='Ratio of important tokens to retrieve')
parser.add_argument("--pq_recent_ratio", type=float, default=0.5,
                    help='Ratio of recent tokens to preserve')
parser.add_argument("--pq_recent_size", type=int, default=32,
                    help='Number of recent tokens to keep')
parser.add_argument("--pq_sink_size", type=int, default=32,
                    help='Number of most recent tokens to always keep')
parser.add_argument("--pq_compressor", type=str, default='pq_search',
                    help='Compression method: pq_search, sparq_f, infllm, h2o, original')
parser.add_argument("--pq_n_subvec_per_head", type=int, default=2,
                    help='Number of PQ subvectors per head')
parser.add_argument("--pq_n_subbits", type=int, default=6,
                    help='Bits per PQ subvector')
parser.add_argument("--pq_topr", type=int, default=32,
                    help='Top-k tokens to retrieve during decoding')
parser.add_argument("--pq_gqa", type=str, default='True',
                    help='Whether to use grouped-query attention')
parser.add_argument("--pq_max_seq_len", type=int, default=32768,
                    help='Maximum sequence length')
parser.add_argument("--pq_cache_block_size", type=int, default=128,
                    help='Block size for cache management')
parser.add_argument("--pq_global_cache_size", type=int, default=4096,
                    help='Size of global cache')
parser.add_argument("--pq_cache_topk", type=int, default=32,
                    help='Number of top-k tokens for cache retrieval')
parser.add_argument("--pq_score_func", type=str, default='sum',
                    help='Score function: sum or max')
parser.add_argument("--pq_drop_ratio", type=float, default=0,
                    help='Drop ratio for tokens')
parser.add_argument("--pq_max_iter", type=int, default=0,
                    help='K-means iterations (0 for auto)')
parser.add_argument("--pq_preserve_layer", type=int, default=0,
                    help='Number of layers to preserve without compression')
parser.add_argument("--fixthreshold", type=float, default=0.85,
                    help='Fixed threshold for topp attention (used with no_drop_lb_topp compressor)')

# Inference
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=32)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--stop_words", type=str, default='')
parser.add_argument("--sliding_window_size", type=int)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=1)

args = parser.parse_args()
args.stop_words = list(filter(None, args.stop_words.split(',')))
if args.server_type == 'hf' or args.server_type == 'gemini'or args.server_type == 'pq':
    args.threads = 1


def get_llm(tokens_to_generate):
    if args.server_type == 'trtllm':
        from client_wrappers import TRTLLMClient
        llm = TRTLLMClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
            max_attention_window_size=args.sliding_window_size,
        )

    elif args.server_type == 'vllm':
        from client_wrappers import VLLMClient
        llm = VLLMClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == 'sglang':
        from client_wrappers import SGLClient
        llm = SGLClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )
        
    elif args.server_type == 'openai':
        from client_wrappers import OpenAIClient
        llm = OpenAIClient(
            model_name=args.model_name_or_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == 'gemini':
        from client_wrappers import GeminiClient
        llm = GeminiClient(
            model_name=args.model_name_or_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )
        
    elif args.server_type == 'hf':
        print(33)
        from model_wrappers import HuggingFaceModel
        llm = HuggingFaceModel(
            name_or_path=args.model_name_or_path,
            do_sample=args.temperature > 0,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
        )
    elif args.server_type == 'pq':
        from model_wrappers import PQModel
        # Infer model_type from model name/path: qwen* -> qwen, others -> llama
        pq_model_type = 'llama'
        if 'qwen' in args.model_name_or_path.lower():
            pq_model_type = 'qwen'
        llm = PQModel(
            name_or_path=args.model_name_or_path,
            model_type=pq_model_type,
            fixbudget=args.pq_fixbudget,
            budget=args.pq_budget,
            compress_ratio=args.pq_compress_ratio,
            important_ratio=args.pq_important_ratio,
            recent_ratio=args.pq_recent_ratio,
            recent_size=args.pq_recent_size,
            sink_size=args.pq_sink_size,
            compressor=args.pq_compressor,
            n_subvec_per_head=args.pq_n_subvec_per_head,
            n_subbits=args.pq_n_subbits,
            topr=args.pq_topr,
            gqa=args.pq_gqa == 'True',
            max_seq_len=args.pq_max_seq_len,
            cache_block_size=args.pq_cache_block_size,
            global_cache_size=args.pq_global_cache_size,
            cache_topk=args.pq_cache_topk,
            score_func=args.pq_score_func,
            drop_ratio=args.pq_drop_ratio,
            max_iter=args.pq_max_iter,
            preserve_layer=args.pq_preserve_layer,
            fixthreshold=args.fixthreshold,
            temperature=args.temperature,
            # top_k=args.top_k,
            # top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
        )
    elif args.server_type == 'mamba':
        from model_wrappers import MambaModel
        # mamba uses its own generation function, do not pass in do_sample
        # https://github.com/state-spaces/mamba/blob/009bec5ee37f586844a3fc89c040a9c1a9d8badf/mamba_ssm/utils/generation.py#L121
        llm = MambaModel(
            name_or_path=args.model_name_or_path,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
        )
        
    else:
        raise RuntimeError(f'Unsupported server type {args.server_type}')

    return llm


def main():
    print("begin")
    start_time = time.time()
    
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    
    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in config_tasks.yaml')
        
    config = tasks_customized.get(args.task)
    config.update(tasks_base[config['task']])

    task_file = args.data_dir / args.task / f'{args.subset}.jsonl'

    if args.chunk_amount > 1:
        pred_file = args.save_dir / f'{args.task}-{args.chunk_idx}.jsonl'
    else:
        pred_file = args.save_dir / f'{args.task}.jsonl'

    print(f'Predict {args.task} \nfrom {task_file}\nto {pred_file}')
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if os.path.exists(pred_file):
        pred_index = [sample['index'] for sample in read_manifest(pred_file)]
        data = [sample for sample in read_manifest(task_file) if sample['index'] not in pred_index]
    else:
        data = read_manifest(task_file)

    # Skip loading LLM if no data to process
    if len(data) == 0:
        print(f'All samples already processed, skipping LLM loading.')
        return

    # Load api
    llm = get_llm(config['tokens_to_generate'])

    def get_output(idx_list, index_list, input_list, outputs_list, others_list, truncation_list, length_list):
        nonlocal llm

        while True:
            try:
                pred_list = llm.process_batch(prompts=input_list)
                break
            except Exception as e:
                traceback.print_exc()

        zipped_iter = zip(pred_list, idx_list, index_list, input_list,
                          outputs_list, others_list, truncation_list, length_list)

        data_id = 0
        for pred, idx, index, input, outputs, others, truncation, length in zipped_iter:
            if isinstance(pred['text'], str):
                pred_text = pred['text']
            elif len(pred['text']) > 0:
                pred_text = pred['text'][0]
            else:
                pred_text = ''

            outputs_parallel[idx] = {
                'index': index,
                'pred': pred_text,
                'outputs': outputs,
                'input': input,
                'others': others,
                'truncation': truncation,
                'length': length,
            }

    threads = []
    outputs_parallel = [{} for _ in range(len(data))]

    batched_data = []
    batch = []
    for idx, data_point in enumerate(data):
        data_point['idx'] = idx

        if len(batch) >= args.batch_size:
            batched_data.append(batch)
            batch = []

        batch.append(data_point)

    if len(batch):
        batched_data.append(batch)

    # Parse task and max_length from save_dir path
    # save_dir format: .../MODEL_NAME/COMPRESSOR/BENCHMARK/MAX_SEQ_LENGTH/pred
    save_dir_parts = str(args.save_dir).split('/')
    if len(save_dir_parts) >= 2:
        max_length = save_dir_parts[-2]  # MAX_SEQ_LENGTH is the parent dir of 'pred'
        # benchmark = save_dir_parts[-3]  # BENCHMARK is the grandparent dir
        benchmark = args.task
    else:
        max_length = 114514
        benchmark = args.task

    if TOPK_SAVE_TOPP is not None or TOPP_SAVE_TOPK is not None:
        tmp = TOPK_SAVE_TOPP if TOPK_SAVE_TOPP is not None else TOPP_SAVE_TOPK
        os.makedirs("record", exist_ok=True)
        with open(f"record/{tmp}.txt", "a") as f:
            f.write(f"{benchmark},length:{max_length}\n")

    # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
    with open(pred_file, 'at', encoding="utf-8", buffering=1) as fout:
        # the data is processed sequentially, so we can store the start and end of current processing window
        start_idx = 0  # window: [start_idx, end_idx]

        use_thread = args.threads > 1

        for batch_idx, batch in tqdm(enumerate(batched_data), total=len(batched_data)):
            idx_list = [data_point['idx'] for data_point in batch]
            end_idx = idx_list[-1]  # the data in a batch is ordered

            if use_thread:
                thread = threading.Thread(
                    target=get_output,
                    kwargs=dict(
                        idx_list=idx_list,
                        index_list=[data_point['index'] for data_point in batch],
                        input_list=[data_point['input'] for data_point in batch],
                        outputs_list=[data_point['outputs'] for data_point in batch],
                        others_list=[data_point.get('others', {}) for data_point in batch],
                        truncation_list=[data_point.get('truncation', -1) for data_point in batch],
                        length_list=[data_point.get('length', -1) for data_point in batch],
                    ),
                )
                thread.start()
                threads.append(thread)

                is_last_batch = (batch_idx == len(batched_data) - 1)

                if (len(threads) == args.threads) or is_last_batch:
                    for thread in threads:
                        thread.join()
                    threads = []
            else:
                # Direct call without threading
                
                get_output(
                    idx_list=idx_list,
                    index_list=[data_point['index'] for data_point in batch],
                    input_list=[data_point['input'] for data_point in batch],
                    outputs_list=[data_point['outputs'] for data_point in batch],
                    others_list=[data_point.get('others', {}) for data_point in batch],
                    truncation_list=[data_point.get('truncation', -1) for data_point in batch],
                    length_list=[data_point.get('length', -1) for data_point in batch],
                )

            # dump the results in current processing window on disk
            for idx in range(start_idx, end_idx + 1):
                if len(outputs_parallel[idx]) > 0:
                    fout.write(json.dumps(outputs_parallel[idx]) + '\n')

            start_idx = end_idx + 1

    # Clean up PQ objects if using pq_search compressor
    if args.server_type == 'pq' and args.pq_compressor == 'pq_search':
        from vq_method.retrieval_based.pq_search import del_objects
        del_objects()

    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")


if __name__ == '__main__':
    import torch
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)
    main()
