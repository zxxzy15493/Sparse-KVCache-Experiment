#!/bin/bash
# DuoAttention timing test

cd "$(dirname "$0")"

4k
python run_timing_test.py \
    --model Meta-Llama-3.1-8B-Instruct \
    --input_max_token 4096 \
    --sparsity 0.5

python run_timing_test.py \
    --model Meta-Llama-3.1-8B-Instruct \
    --input_max_token 65536 \
    --sparsity 0.5