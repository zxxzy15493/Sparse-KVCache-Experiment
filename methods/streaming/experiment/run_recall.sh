#!/bin/bash

Qwen_1M="Qwen/Qwen2.5-7B-Instruct-1M"
Llama="meta-llama/Llama-3.1-8B-Instruct"
Qwen="Qwen/Qwen2.5-7B-Instruct"

LONGBENCH_ROOT="../../benchmarks/Longbench_recall"


run_eval() {
    local MODEL_PATH=$1
    local DATA_ROOT=$2
    local DATASET_NAME=$3

    echo "Starting StreamingLLM Experiment:"
    echo "   Model   : $MODEL_PATH"
    echo "   Dataset : $DATASET_NAME"
    echo "   Budget  : Total=$((START_SIZE + RECENT_SIZE)) (Sink=$START_SIZE, Window=$RECENT_SIZE)"
    echo "   Root    : $DATA_ROOT"

    python recall.py \
        --model_name_or_path "$MODEL_PATH" \
        --data_root "$DATA_ROOT/$DATASET_NAME" \
        --dataset_name "$DATASET_NAME" \
        --start_size "$START_SIZE" \
        --recent_size "$RECENT_SIZE"

    if [ $? -eq 0 ]; then
        echo -e "\nSUCCESS: StreamingLLM for $DATASET_NAME with $MODEL_PATH (Budget: $((START_SIZE + RECENT_SIZE))) completed.\n"
    else
        echo -e "\nFAIL: StreamingLLM for $DATASET_NAME with $MODEL_PATH (Budget: $((START_SIZE + RECENT_SIZE))) encountered an error!\n"
    fi
}


LONGBENCH_TASKS=(
    "qasper"
    "narrativeqa"   
)

BUDGET_LIST=(512 256 128)

echo "Starting StreamingLLM Multi-Budget Evaluation Loop..."

for BUDGET in "${BUDGET_LIST[@]}"; do
    START_SIZE=16
    RECENT_SIZE=$((BUDGET - START_SIZE))
    
    echo "Entering Loop Zone: Total Budget = $BUDGET"

    for DATASET in "${LONGBENCH_TASKS[@]}"; do
        run_eval "$Llama" "$LONGBENCH_ROOT" "$DATASET"
        run_eval "$Qwen" "$LONGBENCH_ROOT" "$DATASET"
    done
done

echo "ALL STREAMINGLLM EXPERIMENTS FOR ALL BUDGETS ARE FINISHED!"