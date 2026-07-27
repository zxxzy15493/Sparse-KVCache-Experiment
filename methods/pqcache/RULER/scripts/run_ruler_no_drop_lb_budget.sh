#!/bin/bash
set -euo pipefail

PQ_COMPRESSOR=no_drop_lb
EXP_NAME=budget
PQ_RECENT_SIZE=0
PQ_SINK_SIZE=0
SEQ_LENGTHS="65536"
TASKS="niah_single_3 vt cwe fwe qa_1"

export PQ_COMPRESSOR
export EXP_NAME
export PQ_RECENT_SIZE
export PQ_SINK_SIZE
export SEQ_LENGTHS
export TASKS

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
models=(llama-3.1-8b qwen-2.5-7b-1m)
budgets=(128 384 1024 4096)

for PQ_BUDGET in "${budgets[@]}"; do
    export PQ_BUDGET

    for MODEL_NAME in "${models[@]}"; do
        case "$MODEL_NAME" in
            llama-3.1-8b)
                MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
                TOKENIZER_PATH=meta-llama/Llama-3.1-8B-Instruct
                MODEL_TEMPLATE_TYPE=meta-llama3
                ;;
            qwen-2.5-7b-1m)
                MODEL_PATH=Qwen/Qwen2.5-7B-Instruct-1M
                TOKENIZER_PATH=Qwen/Qwen2.5-7B-Instruct-1M
                MODEL_TEMPLATE_TYPE=qwen-chat
                ;;
        esac

        export MODEL_NAME
        export MODEL_PATH
        export TOKENIZER_PATH
        export MODEL_TEMPLATE_TYPE
        "$script_dir/run_ruler_method.sh"
    done
done
