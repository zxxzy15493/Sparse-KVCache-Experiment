#!/bin/bash

MODEL_NAME=${1}
ATTN_TYPE=${2:-"Full_Flash_Attn"}
BENCHMARK=${3}
TASK=${4}
NUM_EXAMPLES=${5:--1}
BUDGET=${6:-1024}
ESTIMATE_RATIO=${7:-0.25}
BUDGET_RATIO=${8:-0.018}
RATIO_OR_FIXED=${9:-1}

RESULT_DIR="./results/pred/${MODEL_NAME}/${ATTN_TYPE}/${BENCHMARK}"
LOG_DIR="./log/pred/${MODEL_NAME}/${ATTN_TYPE}/${BENCHMARK}"

mkdir -p ${RESULT_DIR}
mkdir -p ${LOG_DIR}

echo "Parameters: model=${MODEL_NAME} attn_type=${ATTN_TYPE} benchmark=${BENCHMARK} task=${TASK} num_examples=${NUM_EXAMPLES}"
echo "Budget: budget=${BUDGET} estimate_ratio=${ESTIMATE_RATIO} budget_ratio=${BUDGET_RATIO} ratio_or_fixed=${RATIO_OR_FIXED}"
echo "Start to predict..."

python -u pred.py \
    --model_name ${MODEL_NAME} \
    --attn_type ${ATTN_TYPE} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
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
python -u eval.py \
    --model ${MODEL_NAME} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    --save_dir ${RESULT_DIR}

echo "Results:"
cat "${RESULT_DIR}/result.json"
