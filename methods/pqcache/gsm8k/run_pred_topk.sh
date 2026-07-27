#!/bin/bash
set -x
set -e
# topk version: no_drop_lb compressor, budget 360, DeepSeek-R1-Distill-Qwen-1.5B
SEED="4321"
COMPRESSOR="no_drop_lb"
MODE="off" # profile or off
DEVICE=0
COMPRESS=0.1
CORE_OFFSET=0 # 100 160
TOPK=0.5
RECENT_RATIO=0.5
SINK_SIZE=0
FIXBUDGET="--fixbudget"  # Use "--fixbudget" to enable fixed budget mode
BUDGET=360   # Fixed budget size (used when fixbudget is enabled)
SUBVEC=2
SUBBITS=6
TOPR=1
METRIC="euc" # euc ip
GQA="True" # True False
MEAN_V_TRICK="False"
MAX_ITER=0 # 0 for dynamic setting

MAX_CPU_IN_USE=48
DROP=0
RECENT_SIZE=0
PRESERVE_LAYER=0
THRESHOLD=100000
SCORE_FUNC="sum" # sum, max
KEYFORMER_MODE=0
USE_LINGUA=0

# Model — passed directly to pred.py (no config lookup)
# DeepSeek-R1-Distill-Qwen-1.5B is a Qwen2-based model, so the "qwen" branch
# in load_model_and_tokenizer will pick the VQQwen2ForCausalLM patch.
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_NAME="deepseek-qwen-1.5b"

# Output directory for gsm8k predictions
SAVE_DIR="pred/gsm8k/${MODEL_NAME}/${COMPRESSOR}"

# Log file (named after the compressor)
LOG_DIR="log"
LOG_FILE="${LOG_DIR}/${COMPRESSOR}.log"

export CORE_OFFSET=${CORE_OFFSET}

mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

MAX_CPU_IN_USE=${MAX_CPU_IN_USE} \
RANDOM_SEED=${SEED} \
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
CUDA_VISIBLE_DEVICES=${DEVICE} \
TOKENIZERS_PARALLELISM=false \
SUBVEC=${SUBVEC} SUBBITS=${SUBBITS} \
METRIC=${METRIC} \
python pred.py \
    --model ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --save_dir ${SAVE_DIR} \
    --compress_ratio ${COMPRESS} \
    ${FIXBUDGET} \
    --budget ${BUDGET} \
    --important_ratio ${TOPK} \
    --recent_ratio ${RECENT_RATIO} \
    --recent_size ${RECENT_SIZE} \
    --drop_ratio ${DROP} \
    --enable_vq_cache \
    --fp16 \
    --pp-size 1 \
    --sink-size ${SINK_SIZE} \
    --score_func ${SCORE_FUNC} \
    --preserve_layer ${PRESERVE_LAYER} \
    --keyformer_mode ${KEYFORMER_MODE} \
    --compressor ${COMPRESSOR} \
    --threshold ${THRESHOLD} \
    --n_subvec_per_head ${SUBVEC} \
    --n_subbits ${SUBBITS} \
    --topr ${TOPR} \
    --gqa ${GQA} \
    --sparq_mean_v_trick ${MEAN_V_TRICK} \
    --max_iter ${MAX_ITER} \
    > "${LOG_FILE}" 2>&1

python evaluate.py \
    --force \
    --input "${SAVE_DIR}/gsm8k.jsonl" \
    --out-dir "${SAVE_DIR}" \
    --write-correct-field
