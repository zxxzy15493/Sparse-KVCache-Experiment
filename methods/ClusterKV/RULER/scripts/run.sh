#!/usr/bin/env bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Run one ClusterKV RULER experiment:
#   bash run.sh <model_name> <benchmark_name>
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> <benchmark_name>"
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Root directories
GPUS=${GPUS:-1}
ROOT_DIR=${ROOT_DIR:-../../../../benchmarks/ruler/benchmark_root}
MODEL_DIR="../.."
ENGINE_DIR="."
BATCH_SIZE=1

if [[ "$ROOT_DIR" = /* ]]; then
    echo "ROOT_DIR must be a relative path: $ROOT_DIR"
    exit 1
fi

# ClusterKV parameters
TOKEN_BUDGET=${TOKEN_BUDGET:-1024}
NLIST=400
SINK=16
HEAD_SEL="truc"
BALANCE=""
FIT_ITER=20
DIST_T="cosine"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Model and tokenizer
source config_models.sh
if [[ -n ${SEQ_LENGTHS:-} ]]; then
    read -r -a SEQ_LENGTHS <<< "$SEQ_LENGTHS"
else
    SEQ_LENGTHS=(
        4096
        8192
        16384
        32768
        65536
    )
fi

MODEL_NAME=${1}
if ! MODEL_CONFIG=$(MODEL_SELECT "$MODEL_NAME"); then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi
IFS=":" read -r \
    MODEL_PATH \
    MODEL_TEMPLATE_TYPE \
    MODEL_FRAMEWORK \
    TOKENIZER_PATH \
    TOKENIZER_TYPE \
    OPENAI_API_KEY \
    GEMINI_API_KEY \
    AZURE_ID \
    AZURE_SECRET \
    AZURE_ENDPOINT <<< "$MODEL_CONFIG"

export OPENAI_API_KEY
export GEMINI_API_KEY
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}

# Benchmark and tasks
source config_tasks.sh
NUM_SAMPLES=50

synthetic=(
    # "niah_single_1"
    # "niah_single_2"
    "niah_single_3"
    # "niah_multikey_1"
    # "niah_multikey_2"
    # "niah_multikey_3"
    # "niah_multivalue"
    # "niah_multiquery"
    "vt"
    # "cwe"
    "fwe"
    "qa_1"
    # "qa_2"
)
BENCHMARK=${2}
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

# Start a server for non-ClusterKV frameworks when one is configured.
if [ "$MODEL_FRAMEWORK" == "vllm" ]; then
    python pred/serve_vllm.py \
        --model "$MODEL_PATH" \
        --tensor-parallel-size "$GPUS" \
        --dtype bfloat16 \
        --disable-custom-all-reduce &
elif [ "$MODEL_FRAMEWORK" == "trtllm" ]; then
    python pred/serve_trt.py \
        --model_path "$MODEL_PATH" &
elif [ "$MODEL_FRAMEWORK" == "sglang" ]; then
    python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --tp "$GPUS" \
        --port 5000 \
        --enable-flashinfer &
fi

# Prepare data, predict, and evaluate.
total_time=0
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}_budget${TOKEN_BUDGET}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p "$DATA_DIR"
    mkdir -p "$PRED_DIR"

    for TASK in "${TASKS[@]}"; do
        python data/prepare.py \
            --save_dir "$DATA_DIR" \
            --benchmark "$BENCHMARK" \
            --task "$TASK" \
            --tokenizer_path "$TOKENIZER_PATH" \
            --tokenizer_type "$TOKENIZER_TYPE" \
            --max_seq_length "$MAX_SEQ_LENGTH" \
            --model_template_type "$MODEL_TEMPLATE_TYPE" \
            --num_samples "$NUM_SAMPLES" \
            ${REMOVE_NEWLINE_TAB}

        start_time=$(date +%s)
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        python pred/call_api.py \
            --data_dir "$DATA_DIR" \
            --save_dir "$PRED_DIR" \
            --benchmark "$BENCHMARK" \
            --task "$TASK" \
            --server_type "$MODEL_FRAMEWORK" \
            --model_name_or_path "$MODEL_PATH" \
            --model_name "$MODEL_NAME" \
            --temperature "$TEMPERATURE" \
            --top_k "$TOP_K" \
            --top_p "$TOP_P" \
            --batch_size "$BATCH_SIZE" \
            ${STOP_WORDS} \
            --token_budget "$TOKEN_BUDGET" \
            --nlist "$NLIST" \
            --sink "$SINK" \
            --head_sel "$HEAD_SEL" \
            --fit_iter "$FIT_ITER" \
            --dist_t "$DIST_T" \
            ${BALANCE:+--balance}
        end_time=$(date +%s)
        total_time=$((total_time + end_time - start_time))
    done

    python eval/evaluate.py \
        --data_dir "$PRED_DIR" \
        --benchmark "$BENCHMARK"
done

echo "Total time spent on call_api: $total_time seconds"
