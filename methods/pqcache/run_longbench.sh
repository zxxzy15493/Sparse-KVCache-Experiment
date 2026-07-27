#!/bin/bash
set -euo pipefail
set -x

# Method-specific scripts provide these values as environment variables.
: "${MODEL:?MODEL is required}"
: "${COMPRESSOR:?COMPRESSOR is required}"
: "${EXP_NAME:?EXP_NAME is required}"
: "${SINK_SIZE:?SINK_SIZE is required}"
: "${RECENT_SIZE:?RECENT_SIZE is required}"
: "${BUDGET:?BUDGET is required}"

DEVICE=${DEVICE:-0}
SEED=${SEED:-4321}
COMPRESS=${COMPRESS:-0.1}
TOPK=${TOPK:-0.5}
RECENT_RATIO=${RECENT_RATIO:-0.5}
FIXTHRESHOLD=${FIXTHRESHOLD:--1}
SUBVEC=${SUBVEC:-2}
SUBBITS=${SUBBITS:-6}
TOPR=${TOPR:-1}
GQA=${GQA:-True}
MEAN_V_TRICK=${MEAN_V_TRICK:-False}
MAX_ITER=${MAX_ITER:-0}
MAX_CPU_IN_USE=${MAX_CPU_IN_USE:-24}
DROP=${DROP:-0}
PRESERVE_LAYER=${PRESERVE_LAYER:-0}
THRESHOLD=${THRESHOLD:-100000}
SCORE_FUNC=${SCORE_FUNC:-sum}
KEYFORMER_MODE=${KEYFORMER_MODE:-0}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

export MAX_CPU_IN_USE
RANDOM_SEED=$SEED
export RANDOM_SEED
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF
CUDA_VISIBLE_DEVICES=$DEVICE
export CUDA_VISIBLE_DEVICES
TOKENIZERS_PARALLELISM=false
export TOKENIZERS_PARALLELISM
HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_DATASETS_TRUST_REMOTE_CODE
export SUBVEC
export SUBBITS

python vq_pred.py \
    --model "$MODEL" \
    --compress_ratio "$COMPRESS" \
    --fixbudget \
    --budget "$BUDGET" \
    --fixthreshold "$FIXTHRESHOLD" \
    --important_ratio "$TOPK" \
    --recent_ratio "$RECENT_RATIO" \
    --recent_size "$RECENT_SIZE" \
    --drop_ratio "$DROP" \
    --enable_vq_cache \
    --fp16 \
    --pp-size 1 \
    --sink-size "$SINK_SIZE" \
    --exp_name "$EXP_NAME" \
    --score_func "$SCORE_FUNC" \
    --preserve_layer "$PRESERVE_LAYER" \
    --keyformer_mode "$KEYFORMER_MODE" \
    --compressor "$COMPRESSOR" \
    --threshold "$THRESHOLD" \
    --n_subvec_per_head "$SUBVEC" \
    --n_subbits "$SUBBITS" \
    --topr "$TOPR" \
    --gqa "$GQA" \
    --sparq_mean_v_trick "$MEAN_V_TRICK" \
    --max_iter "$MAX_ITER"

python eval.py --model "$MODEL" --exp_name "$EXP_NAME"
