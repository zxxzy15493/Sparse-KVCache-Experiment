#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
XATTN_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

cd "$XATTN_ROOT"
export PYTHONPATH="$XATTN_ROOT:${PYTHONPATH:-}"

models="Qwen2.5-7B-Instruct-1M"
methods="xattn"
tasks="qasper narrativeqa trec lcc"

for p in 0.95 0.8 0.85 0.9; do
    for model in $models; do
        for task in $tasks; do
            for method in $methods; do
                python -u eval/LongBench/budget_qwen_pred.py \
                    --model "$model" \
                    --task "$task" \
                    --method "$method" \
                    --p "$p"
            done
        done
    done
done
