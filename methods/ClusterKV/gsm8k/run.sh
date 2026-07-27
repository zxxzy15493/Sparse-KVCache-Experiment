#!/bin/bash
set -euo pipefail

# ============================================================
# ClusterKV GSM8K eval
# Mirrors gsm8k/run_full.sh but enables ClusterKV sparse
# attention via accuracy/patch.py (the same pipeline used by
# accuracy/LongBench/recall_pred.py).
#
# Cluster hyperparameters (per user spec):
#   token_budget = 360
#   sink         = 16
#   recent       = 32
#   fit_iter     = 10
#   nlist        = 40
#
# All hyperparameters can be overridden through env vars, e.g.
#   TOKEN_BUDGET=512 NLIST=64 bash run.sh
# ============================================================

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"

TOKEN_BUDGET="${TOKEN_BUDGET:-360}"
SINK="${SINK:-16}"
RECENT="${RECENT:-32}"
FIT_ITER="${FIT_ITER:-10}"
NLIST="${NLIST:-40}"
HEAD_SEL="${HEAD_SEL:-truc}"

TAG="cluster_b${TOKEN_BUDGET}_s${SINK}_r${RECENT}_nl${NLIST}_it${FIT_ITER}"

LOG_DIR="./log/${TAG}/${MODEL}"
SAVE_DIR="./results/${TAG}/${MODEL}"

mkdir -p "${LOG_DIR}"
mkdir -p "${SAVE_DIR}"

# Output filename must match get_out_filename() in pred.py:
#   gsm8k-h<head_sel><nlist>fi<fit_iter>sink<sink>_<token_budget>r<recent>.jsonl
PRED_FILE="gsm8k-h${HEAD_SEL}${NLIST}fi${FIT_ITER}sink${SINK}_${TOKEN_BUDGET}r${RECENT}.jsonl"
export PYTHONPATH=..

python -u pred.py \
    --model "${MODEL}" \
    --save_dir "${SAVE_DIR}" \
    --num_shots 8 \
    --cot_type gsm8k-cot \
    --cluster \
    --token_budget "${TOKEN_BUDGET}" \
    --sink "${SINK}" \
    --recent "${RECENT}" \
    --fit_iter "${FIT_ITER}" \
    --nlist "${NLIST}" \
    --head_sel "${HEAD_SEL}" \
    > "${LOG_DIR}/gsm8k.log" 2>&1

python -u evaluate.py \
    --input "${SAVE_DIR}/${PRED_FILE}" \
    --output "${SAVE_DIR}/gsm8k_eval.jsonl" \
    --force

python -u ./tool/data_infos.py \
    --data-dir "${SAVE_DIR}" \
    --model "${MODEL}" \
    --task gsm8k
