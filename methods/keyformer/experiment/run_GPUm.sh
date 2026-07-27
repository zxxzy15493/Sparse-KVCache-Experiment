#!/bin/bash

Llama="meta-llama/Llama-3.1-8B-Instruct"
Qwen_1M="Qwen/Qwen2.5-7B-Instruct-1M"
DATA_ROOT="../../benchmarks/myinput.txt"

RECENT_SIZE=32
BUDGET=1024
KEY_SIZE=$((BUDGET - RECENT_SIZE))

TEST_LENS=(4096 8192 16384 32768 65536 131072)

MODELS=("$Llama" "$Qwen_1M")

for TEST_LEN in "${TEST_LENS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "Running Keyformer: Model=${MODEL}, TestLen=${TEST_LEN}, OutLen=2, Budget=${BUDGET}"
        python GPUm.py \
            --model_name_or_path "${MODEL}" \
            --data_root "${DATA_ROOT}" \
            --key_size "${KEY_SIZE}" \
            --recent_size "${RECENT_SIZE}" \
            --out_len 2 \
            --test_len "${TEST_LEN}" \
            --keyformer
    done
done
