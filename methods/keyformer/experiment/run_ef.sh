#!/bin/bash

Llama="meta-llama/Llama-3.1-8B-Instruct"
Qwen="Qwen/Qwen2.5-7B-Instruct"
DATA_ROOT="../../../benchmarks/myinput.txt"
Qwen_1M="Qwen/Qwen2.5-7B-Instruct-1M"

BUDGET=1024
RECENT_SIZE=32
KEY_SIZE=$((BUDGET - RECENT_SIZE))

MODELS=("${Llama}" "${Qwen_1M}")

TEST_LENGTHS=(4096 8192 16384 32768 65536 131072)

for MODEL in "${MODELS[@]}"; do
    for TEST_LEN in "${TEST_LENGTHS[@]}"; do
        echo "Running Model: ${MODEL} | Input Length: ${TEST_LEN}"
        
        python Efficency.py \
            --model_name_or_path "${MODEL}" \
            --data_root "${DATA_ROOT}" \
            --key_size "${KEY_SIZE}" \
            --recent_size "${RECENT_SIZE}" \
            --out_len 32 \
            --test_len "${TEST_LEN}" \
            --keyformer
            
    done
done