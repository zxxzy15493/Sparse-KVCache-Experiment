#!/bin/bash

Qwen="Qwen/Qwen2.5-7B-Instruct"
Qwen_1M="Qwen/Qwen2.5-7B-Instruct-1M"
Llama="meta-llama/Llama-3.1-8B-Instruct"
Glm="glm-4-9b-chat-1m"
DATA_ROOT="../../benchmarks/myinput.txt"

RECENT=32
BUDGET=1024
HH=$((BUDGET - RECENT))

TEST_LENS=(4096 8192 16384 32768 65536 131072)

MODELS=("$Llama" "$Qwen_1M")

for TEST_LEN in "${TEST_LENS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "Running H2O: Model=${MODEL}, TestLen=${TEST_LEN}, OutLen=2, Budget=${BUDGET}"
        python GPUm.py \
            --model_name_or_path "${MODEL}" \
            --data_root "${DATA_ROOT}" \
            --heavy_hitter_size "${HH}" \
            --recent_size "${RECENT}" \
            --out_len 2 \
            --test_len "${TEST_LEN}" \
            --enable_h2o_cache
    done
done
