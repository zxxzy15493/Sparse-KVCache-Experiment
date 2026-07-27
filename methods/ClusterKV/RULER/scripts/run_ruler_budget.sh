#!/usr/bin/env bash
# Run the RULER accuracy-versus-budget experiment at 64K input length.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

MODELS=(
    llama-3.1-8b
    qwen-2.5-7b-1m
)
BUDGETS=(
    128
    384
    1024
    4096
)
SEQ_LENGTHS="65536"

for MODEL_NAME in "${MODELS[@]}"; do
    for TOKEN_BUDGET in "${BUDGETS[@]}"; do
        SEQ_LENGTHS="$SEQ_LENGTHS" \
        TOKEN_BUDGET="$TOKEN_BUDGET" \
        bash ./run.sh \
            "$MODEL_NAME" \
            synthetic
    done
done
