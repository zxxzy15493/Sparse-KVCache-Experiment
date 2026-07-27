#!/usr/bin/env bash
set -euo pipefail
set -x

# PQCache GSM8K prediction followed by answer extraction and accuracy scoring.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

SEED=${SEED:-4321}
DEVICE=${DEVICE:-0}
MODEL_PATH=${MODEL_PATH:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
MODEL_NAME=${MODEL_NAME:-"deepseek-qwen-1.5b"}
COMPRESSOR=pq_search
BUDGET=${BUDGET:-360}

COMPRESS=${COMPRESS:-0.1}
TOPK=${TOPK:-0.5}
RECENT_RATIO=${RECENT_RATIO:-0.5}
SINK_SIZE=${SINK_SIZE:-16}
RECENT_SIZE=${RECENT_SIZE:-32}
SUBVEC=${SUBVEC:-2}
SUBBITS=${SUBBITS:-6}
TOPR=${TOPR:-1}
GQA=${GQA:-True}
MAX_ITER=${MAX_ITER:-0}
MAX_CPU_IN_USE=${MAX_CPU_IN_USE:-48}

SAVE_DIR=${SAVE_DIR:-"pred/gsm8k/${MODEL_NAME}/${COMPRESSOR}"}
LOG_DIR=${LOG_DIR:-log}
LOG_FILE=${LOG_FILE:-"${LOG_DIR}/${MODEL_NAME}_${COMPRESSOR}_b${BUDGET}.log"}
mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

MAX_CPU_IN_USE=${MAX_CPU_IN_USE} \
RANDOM_SEED=${SEED} \
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
CUDA_VISIBLE_DEVICES=${DEVICE} \
TOKENIZERS_PARALLELISM=false \
python pred.py \
    --model "${MODEL_PATH}" \
    --model_name "${MODEL_NAME}" \
    --save_dir "${SAVE_DIR}" \
    --compress_ratio "${COMPRESS}" \
    --fixbudget \
    --budget "${BUDGET}" \
    --important_ratio "${TOPK}" \
    --recent_ratio "${RECENT_RATIO}" \
    --recent_size "${RECENT_SIZE}" \
    --drop_ratio 0 \
    --enable_vq_cache \
    --fp16 \
    --pp-size 1 \
    --sink-size "${SINK_SIZE}" \
    --score_func sum \
    --preserve_layer 0 \
    --keyformer_mode 0 \
    --compressor "${COMPRESSOR}" \
    --threshold 100000 \
    --n_subvec_per_head "${SUBVEC}" \
    --n_subbits "${SUBBITS}" \
    --topr "${TOPR}" \
    --gqa "${GQA}" \
    --sparq_mean_v_trick False \
    --max_iter "${MAX_ITER}" \
    > "${LOG_FILE}" 2>&1

python evaluate.py \
    --input "${SAVE_DIR}/gsm8k.jsonl" \
    --out-dir "${SAVE_DIR}" \
    --force \
    --write-correct-field
