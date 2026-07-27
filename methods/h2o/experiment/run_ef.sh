#!/bin/bash

Qwen="Qwen/Qwen2.5-7B-Instruct"
Llama="meta-llama/Llama-3.1-8B-Instruct"
DATA_ROOT="../../benchmarks/myinput.txt"
Qwen_1M="Qwen/Qwen2.5-7B-Instruct-1M"

BUDGET=1024
RECENT=32
HH=$((BUDGET - RECENT))

MODELS=("${Llama}" "${Qwen_1M}")

TEST_LENGTHS=(4096 8192 16384 32768 65536 131072)

for MODEL in "${MODELS[@]}"; do
    for TEST_LEN in "${TEST_LENGTHS[@]}"; do
        echo "Running Model: ${MODEL} | Input Length: ${TEST_LEN}"
        
        python Efficency.py \
            --model_name_or_path "${MODEL}" \
            --data_root "${DATA_ROOT}" \
            --heavy_hitter_size "${HH}" \
            --recent_size "${RECENT}" \
            --out_len 32 \
            --test_len "${TEST_LEN}" \
            --enable_h2o_cache
            
    done
done