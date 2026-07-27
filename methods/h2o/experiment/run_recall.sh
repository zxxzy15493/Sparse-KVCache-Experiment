#!/bin/bash

Qwen_1M="Qwen/Qwen2.5-7B-Instruct-1M"
Llama="meta-llama/Llama-3.1-8B-Instruct"
Qwen="Qwen/Qwen2.5-7B-Instruct"

LONGBENCH_ROOT="../../benchmarks/Longbench_recall"


run_eval() {
    local MODEL_PATH=$1
    local DATA_ROOT=$2
    local DATASET_NAME=$3


    echo "   Model   : $MODEL_PATH"
    echo "   Dataset : $DATASET_NAME"
    echo "   Budget  : Total=$((HEAVY_HITTER_SIZE + RECENT_SIZE)) (H2O=$HEAVY_HITTER_SIZE, Recent=$RECENT_SIZE)"
    echo "   Root    : $DATA_ROOT"
    python recall.py \
        --model_name_or_path "$MODEL_PATH" \
        --data_root "$DATA_ROOT/$DATASET_NAME" \
        --dataset_name "$DATASET_NAME" \
        --heavy_hitter_size "$HEAVY_HITTER_SIZE" \
        --recent_size "$RECENT_SIZE"

    if [ $? -eq 0 ]; then
        echo -e "\nSUCCESS: $DATASET_NAME with $MODEL_PATH (Budget: $((HEAVY_HITTER_SIZE + RECENT_SIZE))) completed.\n"
    else
        echo -e "\nFAIL: $DATASET_NAME with $MODEL_PATH (Budget: $((HEAVY_HITTER_SIZE + RECENT_SIZE))) encountered an error!\n"
    fi
}


LONGBENCH_TASKS=(
    "narrativeqa"
    "qasper"
)

BUDGET_LIST=(128 256 512 1024)

echo "Starting Multi-Budget Evaluation Loop..."

for BUDGET in "${BUDGET_LIST[@]}"; do
    RECENT_SIZE=32
    HEAVY_HITTER_SIZE=$((BUDGET - RECENT_SIZE))
    echo "ntering Loop Zone: Total Budget = $BUDGET"

    for DATASET in "${LONGBENCH_TASKS[@]}"; do
        run_eval "$Qwen" "$LONGBENCH_ROOT" "$DATASET"
    done
done

echo "ALL ASSIGNED EXPERIMENTS FOR ALL BUDGETS ARE FINISHED!"
