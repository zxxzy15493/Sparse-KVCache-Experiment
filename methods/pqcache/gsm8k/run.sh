#!/bin/bash
set -x
set -euo pipefail

SEED=${SEED:-"4321"}

MODEL_PATH=${1:-${MODEL_PATH:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}}
MODEL_NAME=${2:-${MODEL_NAME:-"deepseek-qwen-1.5b"}}
COMPRESSOR=${3:-${COMPRESSOR:-"no_drop_lb"}}
BUDGET=${4:-${BUDGET:-360}}
FIXTHRESHOLD=${5:-${FIXTHRESHOLD:-0.90}}
THRESHOLD=${6:-${THRESHOLD:-100000}}
SINK_SIZE=${7:-${SINK_SIZE:-0}}
RECENT_SIZE=${8:-${RECENT_SIZE:-0}}

MODE=${MODE:-"off"}
DEVICE=${DEVICE:-0}
COMPRESS=${COMPRESS:-0.1}
CORE_OFFSET=${CORE_OFFSET:-0}
TOPK=${TOPK:-0.5}
RECENT_RATIO=${RECENT_RATIO:-0.5}
FIXBUDGET=${FIXBUDGET:-"--fixbudget"}

SUBVEC=${SUBVEC:-2}
SUBBITS=${SUBBITS:-6}
TOPR=${TOPR:-1}
METRIC=${METRIC:-"euc"}
GQA=${GQA:-"True"}
MEAN_V_TRICK=${MEAN_V_TRICK:-"False"}
MAX_ITER=${MAX_ITER:-0}

MAX_CPU_IN_USE=${MAX_CPU_IN_USE:-48}
DROP=${DROP:-0}
PRESERVE_LAYER=${PRESERVE_LAYER:-0}
SCORE_FUNC=${SCORE_FUNC:-"sum"}
KEYFORMER_MODE=${KEYFORMER_MODE:-0}
USE_LINGUA=${USE_LINGUA:-0}

SAVE_DIR=${SAVE_DIR:-"pred/gsm8k/${MODEL_NAME}/${COMPRESSOR}"}
LOG_DIR=${LOG_DIR:-"log"}
LOG_FILE=${LOG_FILE:-"${LOG_DIR}/${MODEL_NAME}_${COMPRESSOR}_b${BUDGET}.log"}

export CORE_OFFSET=${CORE_OFFSET}

mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

MAX_CPU_IN_USE=${MAX_CPU_IN_USE} \
RANDOM_SEED=${SEED} \
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
CUDA_VISIBLE_DEVICES=${DEVICE} \
TOKENIZERS_PARALLELISM=false \
SUBVEC=${SUBVEC} \
SUBBITS=${SUBBITS} \
METRIC=${METRIC} \
python pred.py \
    --model "${MODEL_PATH}" \
    --model_name "${MODEL_NAME}" \
    --save_dir "${SAVE_DIR}" \
    --compress_ratio "${COMPRESS}" \
    ${FIXBUDGET} \
    --budget "${BUDGET}" \
    --fixthreshold "${FIXTHRESHOLD}" \
    --important_ratio "${TOPK}" \
    --recent_ratio "${RECENT_RATIO}" \
    --recent_size "${RECENT_SIZE}" \
    --drop_ratio "${DROP}" \
    --enable_vq_cache \
    --fp16 \
    --pp-size 1 \
    --sink-size "${SINK_SIZE}" \
    --score_func "${SCORE_FUNC}" \
    --preserve_layer "${PRESERVE_LAYER}" \
    --keyformer_mode "${KEYFORMER_MODE}" \
    --compressor "${COMPRESSOR}" \
    --threshold "${THRESHOLD}" \
    --n_subvec_per_head "${SUBVEC}" \
    --n_subbits "${SUBBITS}" \
    --topr "${TOPR}" \
    --gqa "${GQA}" \
    --sparq_mean_v_trick "${MEAN_V_TRICK}" \
    --max_iter "${MAX_ITER}" \
    > "${LOG_FILE}" 2>&1