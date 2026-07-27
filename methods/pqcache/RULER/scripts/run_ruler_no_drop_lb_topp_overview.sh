#!/bin/bash
set -euo pipefail

PQ_COMPRESSOR=no_drop_lb_topp
EXP_NAME=overview
PQ_BUDGET=1024
PQ_FIXTHRESHOLD=0.9
PQ_RECENT_SIZE=0
PQ_SINK_SIZE=0
SEQ_LENGTHS="4096 8192 16384 32768 65536"
TASKS="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2"

export PQ_COMPRESSOR
export EXP_NAME
export PQ_BUDGET
export PQ_FIXTHRESHOLD
export PQ_RECENT_SIZE
export PQ_SINK_SIZE
export SEQ_LENGTHS
export TASKS

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

models=(llama-3.1-8b qwen-2.5-7b-1m)

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
