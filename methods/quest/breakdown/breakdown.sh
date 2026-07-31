#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")"
# --input_lengths 4096,8192,16384,32768,65536,131072 \
python3 breakdown.py \
    --model_name Llama \
    --model_type llama \
    --dtype bfloat16 \
    --input_lengths 4096,65536 \
    --max_new_tokens 32 \
    --token_budget 1024 \
    --page_size 16 \
    --warmup_iteration 2 \
    --iteration 3 \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path ../../../benchmarks/myinput.txt \
    --output_dir ../breakdown_results

