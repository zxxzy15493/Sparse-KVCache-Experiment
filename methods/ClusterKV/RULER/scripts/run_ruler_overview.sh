#!/usr/bin/env bash
# Run the RULER accuracy overview at the fixed ClusterKV budget.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

MODELS=(
    llama-3.1-8b
    qwen-2.5-7b-1m
)
SEQ_LENGTHS="4096 8192 16384 32768 65536"
TOKEN_BUDGET=1024

for MODEL_NAME in "${MODELS[@]}"; do
    SEQ_LENGTHS="$SEQ_LENGTHS" \
    TOKEN_BUDGET="$TOKEN_BUDGET" \
    bash ./run.sh \
        "$MODEL_NAME" \
        synthetic
done
