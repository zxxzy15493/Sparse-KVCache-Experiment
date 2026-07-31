#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
XATTN_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

cd "$XATTN_ROOT"
export PYTHONPATH="$XATTN_ROOT:${PYTHONPATH:-}"

models="Llama-3.1-8B-Instruct"
methods="xattn"
tasks="samsum narrativeqa qasper triviaqa 2wikimqa musique gov_report multi_news passage_count passage_retrieval_en lcc repobench-p"

for model in $models; do
    for task in $tasks; do
        for method in $methods; do
            python -u eval/LongBench/llama_pred.py \
                --model "$model" \
                --task "$task" \
                --method "$method"
        done
    done
done
