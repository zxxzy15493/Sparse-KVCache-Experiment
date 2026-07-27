#!/bin/bash

Qwen_1M="Qwen/Qwen2.5-7B-Instruct-1M"
Llama="meta-llama/Llama-3.1-8B-Instruct"
Qwen="Qwen/Qwen2.5-7B-Instruct"

LONGBENCH_ROOT="../../benchmarks/Longbench_recall"


run_eval() {
    local MODEL_PATH=$1
    local DATA_ROOT=$2
    local DATASET_NAME=$3

    echo "=========================================================="
    echo "Starting Keyformer Experiment:"
    echo "   Model   : $MODEL_PATH"
    echo "   Dataset : $DATASET_NAME"
    echo "   Budget  : Total=$((KEY_SIZE + RECENT_SIZE)) (KeySize=$KEY_SIZE, Recent=$RECENT_SIZE)"
    echo "   Root    : $DATA_ROOT"
    echo "=========================================================="

    python recall.py \
        --model_name_or_path "$MODEL_PATH" \
        --data_root "$DATA_ROOT/$DATASET_NAME" \
        --dataset_name "$DATASET_NAME" \
        --key_size "$KEY_SIZE" \
        --recent_size "$RECENT_SIZE" \
        --tau_init 1.0 \
        --tau_delta 0.01

    if [ $? -eq 0 ]; then
        echo -e "\nSUCCESS: Keyformer for $DATASET_NAME with $MODEL_PATH (Budget: $((KEY_SIZE + RECENT_SIZE))) completed.\n"
    else
        echo -e "\nFAIL: Keyformer for $DATASET_NAME with $MODEL_PATH (Budget: $((KEY_SIZE + RECENT_SIZE))) encountered an error!\n"
    fi
}


LONGBENCH_TASKS=(
    "narrativeqa"
    "qasper"
)

BUDGET_LIST=(1024 512 256 128)

echo "Starting Keyformer Multi-Budget Evaluation Loop..."

for BUDGET in "${BUDGET_LIST[@]}"; do
    RECENT_SIZE=32
    KEY_SIZE=$((BUDGET - RECENT_SIZE))
    
    echo "----------------------------------------------------------"
    echo "Entering Loop Zone: Total Budget = $BUDGET"
    echo "----------------------------------------------------------"

    for DATASET in "${LONGBENCH_TASKS[@]}"; do
        run_eval "$Qwen" "$LONGBENCH_ROOT" "$DATASET"
        run_eval "$Llama" "$LONGBENCH_ROOT" "$DATASET"
    done
done

echo "ALL KEYFORMER EXPERIMENTS FOR ALL BUDGETS ARE FINISHED!"