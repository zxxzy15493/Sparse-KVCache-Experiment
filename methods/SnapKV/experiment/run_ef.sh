#!/bin/bash

Llama="meta-llama/Llama-3.1-8B-Instruct"
Qwen_1M="Qwen/Qwen2.5-7B-Instruct-1M"
DATASET="../../../benchmarks/myinput.txt"
COMPRESS_CONFIG_DIR="./config"

BUDGET=1024
COMPRESS_ARGS_PATH="${COMPRESS_CONFIG_DIR}/ablation_c${BUDGET}_w32_k7_maxpool.json"

MODELS=("${Llama}" "${Qwen_1M}")

TEST_LENGTHS=(4096 8192 16384 32768 65536 131072)

for MODEL in "${MODELS[@]}"; do
    for TEST_LEN in "${TEST_LENGTHS[@]}"; do
        echo "Running Model: ${MODEL} | Input Length: ${TEST_LEN}"
        
        python Efficency.py \
            --model_name_or_path "${MODEL}" \
            --data_root "$DATASET" \
            --compress_args_path "$COMPRESS_ARGS_PATH" \
            --test_len "${TEST_LEN}" \
            --out_len 32
            
    done
done
