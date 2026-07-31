#!/bin/bash

Qwen_1M="Qwen/Qwen2.5-7B-Instruct-1M"
Llama="meta-llama/Llama-3.1-8B-Instruct"
Qwen="Qwen/Qwen2.5-7B-Instruct"

LONGBENCH_ROOT="../../../benchmarks/Longbench_recall"


run_eval() {
    local MODEL_PATH=$1
    local DATA_ROOT=$2
    local DATASET_NAME=$3
    local CONFIG_NAME=$4

    echo "Starting SnapKV Experiment:"
    echo "   Model       : $MODEL_PATH"
    echo "   Dataset     : $DATASET_NAME"
    echo "   Config File : config/$CONFIG_NAME"
    echo "   Root        : $DATA_ROOT"

    python recall.py \
        --model "$MODEL_PATH" \
        --dataset "$DATA_ROOT" \
        --dataset_name "$DATASET_NAME" \
        --compress_args_path "$CONFIG_NAME"

    if [ $? -eq 0 ]; then
        echo -e "\nSUCCESS: SnapKV for $DATASET_NAME with $MODEL_PATH using $CONFIG_NAME completed.\n"
    else
        echo -e "\nFAIL: SnapKV for $DATASET_NAME with $MODEL_PATH using $CONFIG_NAME encountered an error!\n"
    fi
}



LONGBENCH_TASKS=(
    "qasper"
    "narrativeqa"
)

SNAPKV_CONFIGS=(
    "ablation_c1024_w32_k7_maxpool.json"
    "ablation_c128_w32_k7_maxpool.json"
    "ablation_c256_w32_k7_maxpool.json"
    "ablation_c512_w32_k7_maxpool.json"
    
)

echo "Starting SnapKV Ablation Loop (K=7, MaxPool)..."

for CONFIG in "${SNAPKV_CONFIGS[@]}"; do
    
    echo "Entering Loop Zone: Loading Ablation Config [ $CONFIG ]"

    for DATASET in "${LONGBENCH_TASKS[@]}"; do
        run_eval "$Llama" "$LONGBENCH_ROOT" "$DATASET" "$CONFIG"
        run_eval "$Qwen" "$LONGBENCH_ROOT" "$DATASET" "$CONFIG"
    done
done

echo "ALL SNAPKV ABLATION EXPERIMENTS ARE FINISHED!"