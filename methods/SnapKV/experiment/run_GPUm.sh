#!/bin/bash

Llama="meta-llama/Llama-3.1-8B-Instruct"
Qwen_1M="Qwen/Qwen2.5-7B-Instruct-1M"
DATASET="../../benchmarks/myinput.txt"
COMPRESS_CONFIG_DIR="./config"

BUDGET=1024
COMPRESS_ARGS_PATH="${COMPRESS_CONFIG_DIR}/ablation_c${BUDGET}_w32_k7_maxpool.json"

TEST_LENS=(262144)


MODELS=("$Llama" "$Qwen_1M")

for TEST_LEN in "${TEST_LENS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo ">>> Running SnapKV: Model=${MODEL}, TestLen=${TEST_LEN}, OutLen=2, Budget=${BUDGET}"
        python GPUm.py \
            --model_name_or_path "$MODEL" \
            --data_root "$DATASET" \
            --compress_args_path "$COMPRESS_ARGS_PATH" \
            --test_len "${TEST_LEN}" \
            --out_len 2
    done
done


