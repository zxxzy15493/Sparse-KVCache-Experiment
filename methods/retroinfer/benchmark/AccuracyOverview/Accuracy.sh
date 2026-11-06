#!/bin/bash

set -euo pipefail

MODEL_NAME=${1}
ATTN_TYPE=${2:-"Full_Flash_Attn"}
BENCHMARK=${3}
TASK=${4}
MAX_LEN=${5:-65536}
NUM_EXAMPLES=${6:--1}
BUDGET=${7:-1024}
ESTIMATE_RATIO=${8:-0.25}
BUDGET_RATIO=${9:-0.018}
RATIO_OR_FIXED=${10:-1}

DATA_ROOT="../../../../benchmarks"

if [ "$BENCHMARK" = "LongBench" ]; then
    DATA_DIR="${DATA_ROOT}/LongBench"
elif [ "$BENCHMARK" = "Synthetic" ]; then
    case "$MODEL_NAME" in
        llama*) MODEL_SHORT="llama-3.1-8b" ;;
        qwen3-32b|qwen3*) MODEL_SHORT="qwen3-32b" ;;
        qwen*)  MODEL_SHORT="qwen-2.5-7b-1m" ;;
        glm*)      MODEL_SHORT="glm-4-9b" ;;
        *)         MODEL_SHORT="$MODEL_NAME" ;;
    esac
    DATA_DIR="${DATA_ROOT}/ruler/benchmark_root/${MODEL_SHORT}/synthetic/${MAX_LEN}/data/${TASK}"
else
    echo "Unknown benchmark: $BENCHMARK"
    exit 1
fi

if [ "$BENCHMARK" = "Synthetic" ]; then
    RESULT_DIR="./results/pred/${MODEL_NAME}/${ATTN_TYPE}/${BENCHMARK}/${MAX_LEN}/budget-${BUDGET}"
    LOG_DIR="./log/pred/${MODEL_NAME}/${ATTN_TYPE}/${BENCHMARK}/${MAX_LEN}/budget-${BUDGET}"
else
    RESULT_DIR="./results/pred/${MODEL_NAME}/${ATTN_TYPE}/${BENCHMARK}/budget-${BUDGET}"
    LOG_DIR="./log/pred/${MODEL_NAME}/${ATTN_TYPE}/${BENCHMARK}/budget-${BUDGET}"
fi

mkdir -p ${RESULT_DIR}
mkdir -p ${LOG_DIR}

echo "Parameters: model=${MODEL_NAME} attn_type=${ATTN_TYPE} benchmark=${BENCHMARK} task=${TASK} max_len=${MAX_LEN} num_examples=${NUM_EXAMPLES}"
echo "Budget: budget=${BUDGET} estimate_ratio=${ESTIMATE_RATIO} budget_ratio=${BUDGET_RATIO} ratio_or_fixed=${RATIO_OR_FIXED}"
echo "DATA_DIR: ${DATA_DIR}"
echo "Start to predict..."

# LongBench is fetched from the Hub and can involve a large dataset README or
# dataset script.  Keep the caller's explicit values, but avoid the Hub's
# short default metadata timeout on slower mirrors.
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

python -u pred.py \
    --model_name ${MODEL_NAME} \
    --attn_type ${ATTN_TYPE} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    --max_len ${MAX_LEN} \
    --save_dir ${RESULT_DIR} \
    --dtype bf16 \
    --device cuda:0 \
    --num_samples ${NUM_EXAMPLES} \
    --budget ${BUDGET} \
    --estimate_ratio ${ESTIMATE_RATIO} \
    --budget_ratio ${BUDGET_RATIO} \
    --ratio_or_fixed ${RATIO_OR_FIXED} \
    >${LOG_DIR}/${TASK}.log 2>&1

echo "Start to evaluate..."
if [ "$BENCHMARK" = "LongBench" ]; then
    python -u eval.py \
        --model ${MODEL_NAME} \
        --benchmark ${BENCHMARK} \
        --task ${TASK} \
        --save_dir ${RESULT_DIR}
    echo "Results:"
    cat "${RESULT_DIR}/result.json"
elif [ "$BENCHMARK" = "Synthetic" ]; then
    python -u evaluate.py \
        --data-dir ${RESULT_DIR} \
        --tasks ${TASK}
    echo "Results:"
    cat "${RESULT_DIR}/summary.csv"
fi
