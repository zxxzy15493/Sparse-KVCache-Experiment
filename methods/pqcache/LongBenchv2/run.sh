#!/bin/bash
set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# PQ compression params
SEED="4321"
COMPRESSOR="pq_search"
COMPRESSOR="original"
DEVICE="0"
COMPRESS=0.1
TOPK=0.5
RECENT_RATIO=0.5
SINK_SIZE=16
BUDGET=1024
SUBVEC=3
SUBBITS=8
TOPR=32
METRIC="euc"
GQA="True"
MEAN_V_TRICK="False"
MAX_ITER=0
SCORE_FUNC="sum"
MAX_CPU_IN_USE=25
RECENT_SIZE=32
FIXTHRESHOLD=0.9

# data paths
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct-1M"
MODEL_NAME="qwen-2.5-7b"
DATA_FILE="../../../benchmarks/longbenchv2/filtered_longbench_v2_64k-192k.jsonl"

export CUDA_VISIBLE_DEVICES=${DEVICE}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

export PYTHONPATH="..${PYTHONPATH:+:${PYTHONPATH}}"

python mypred.py \
    --model_name ${MODEL_NAME} \
    --model_path ${MODEL_PATH} \
    --data_file ${DATA_FILE} \
    --max_context_len 234800 \
    --max_new_tokens 128 \
    --enable_vq_cache \
    --pp_size 1 \
    --compressor ${COMPRESSOR} \
    --compress_ratio ${COMPRESS} \
    --fixbudget \
    --budget ${BUDGET} \
    --fixthreshold ${FIXTHRESHOLD} \
    --important_ratio ${TOPK} \
    --recent_ratio ${RECENT_RATIO} \
    --recent_size ${RECENT_SIZE} \
    --sink_size ${SINK_SIZE} \
    --n_subvec_per_head ${SUBVEC} \
    --n_subbits ${SUBBITS} \
    --topr ${TOPR} \
    --gqa ${GQA} \
    --sparq_mean_v_trick ${MEAN_V_TRICK} \
    --max_iter ${MAX_ITER} \
    --score_func ${SCORE_FUNC} \
    # --cot \
