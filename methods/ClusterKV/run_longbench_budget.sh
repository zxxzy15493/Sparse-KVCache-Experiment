#!/usr/bin/env bash
# Run the ClusterKV LongBench accuracy-versus-budget experiment.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LONG_BENCH_DIR="$SCRIPT_DIR/accuracy/LongBench"
DEVICE=${CUDA_VISIBLE_DEVICES:-0}

MODELS=(
    llama3.1-8b-chat-32k
    qwen2.5-7b-chat-32k
)
DATASETS=(
    narrativeqa
    qasper
    trec
    lcc
)
BUDGETS=(128 256 512 1024)

cd "$LONG_BENCH_DIR"

for model in "${MODELS[@]}"; do
    for budget in "${BUDGETS[@]}"; do
        CUDA_VISIBLE_DEVICES="$DEVICE" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python mypred.py \
            --model "$model" \
            --cluster \
            --token_budget "$budget" \
            --task "${DATASETS[@]}"
    done

    # Evaluate 
    python eval.py --model "$model" 
done
