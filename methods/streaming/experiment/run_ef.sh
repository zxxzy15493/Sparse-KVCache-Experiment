#!/bin/bash

Llama="meta-llama/Llama-3.1-8B-Instruct"
Qwen="Qwen/Qwen2.5-7B-Instruct"
Glm="THUDM/glm-4-9b-chat-1m"
Qwen_1M="Qwen/Qwen2.5-7B-Instruct-1M"
DATA_ROOT="../../../benchmarks/myinput.txt"

BUDGET=1024
START_SIZE=16
RECENT=$((BUDGET - START_SIZE))

MODELS=("${Llama}" "${Qwen}")

TEST_LENGTHS=(4096 8192 16384 32768 65536 131072)

for MODEL in "${MODELS[@]}"; do
    for TEST_LEN in "${TEST_LENGTHS[@]}"; do
        echo "Running Model: ${MODEL} | Input Length: ${TEST_LEN}"
        
        python Efficency.py \
            --model_name_or_path "${MODEL}" \
            --data_root "${DATA_ROOT}" \
            --start_size "${START_SIZE}" \
            --recent_size "${RECENT}" \
            --out_len 32 \
            --test_len "${TEST_LEN}" \
            --enable_streaming
            
    done
done