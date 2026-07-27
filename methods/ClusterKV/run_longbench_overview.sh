#!/usr/bin/env bash
# Run the ClusterKV LongBench accuracy overview.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LONG_BENCH_DIR="$SCRIPT_DIR/accuracy/LongBench"
DEVICE=${CUDA_VISIBLE_DEVICES:-0}

# The default dataset list is defined in accuracy/LongBench/mypred.py.
MODELS=(
    llama3.1-8b-chat-32k
    qwen2.5-7b-chat-32k
    glm4-9b-chat-1m
)

cd "$LONG_BENCH_DIR"

for model in "${MODELS[@]}"; do
    CUDA_VISIBLE_DEVICES="$DEVICE" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python mypred.py \
        --model "$model" \
        --cluster \
        --token_budget 1024

    python eval.py --model "$model"
done
