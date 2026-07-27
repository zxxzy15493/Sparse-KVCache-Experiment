#!/bin/bash

Llama="meta-llama/Llama-3.1-8B-Instruct"
Qwen_1M="Qwen/Qwen2.5-7B-Instruct-1M"
DATA_ROOT="../../benchmarks/myinput.txt"

START_SIZE=16
BUDGET=1024
RECENT=$((BUDGET - START_SIZE))

TEST_LENS=(262144)

MODELS=("$Llama" "$Qwen_1M")

for TEST_LEN in "${TEST_LENS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo ">>> Running StreamingLLM: Model=${MODEL}, TestLen=${TEST_LEN}, OutLen=2, Budget=${BUDGET}"
        python GPUm.py \
            --model_name_or_path "${MODEL}" \
            --data_root "${DATA_ROOT}" \
            --start_size "${START_SIZE}" \
            --recent_size "${RECENT}" \
            --out_len 2 \
            --test_len "${TEST_LEN}" \
            --enable_streaming
    done
done

