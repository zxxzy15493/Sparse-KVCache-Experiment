#!/bin/bash
set -x
set -e

# Usage:
# bash run_recall_sweep.sh
# bash run_recall_sweep.sh 0

DEVICE=${1:-0}
DATA_DIR="../../benchmarks/Longbench_recall"

SEED="4321"
COMPRESSOR="pq_search"
EXP_NAME="recall_test"

COMPRESS=0.1
CORE_OFFSET=0
TOPK=0.5
RECENT_RATIO=0.5
SINK_SIZE=16
FIXBUDGET="--fixbudget"

SUBVEC=2
SUBBITS=6
TOPR=1
METRIC="euc" # euc or ip
GQA="True"
MEAN_V_TRICK="False"
MAX_ITER=0

MAX_CPU_IN_USE=16
DROP=0
RECENT_SIZE=32
PRESERVE_LAYER=0
THRESHOLD=100000
SCORE_FUNC="sum"
KEYFORMER_MODE=0

MODELS=("llama-3.1-8b" "qwen-2.5-7b-1m")
DATASETS=("qasper" "narrativeqa")
BUDGETS=(128 256 512 1024)

export CORE_OFFSET=${CORE_OFFSET}

mkdir -p logs_recall

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for BUDGET in "${BUDGETS[@]}"; do

            RECALL_NAME="${MODEL}_${DATASET}_budget_${BUDGET}_recall_${COMPRESSOR}"
            LOG_FILE="logs_recall/${RECALL_NAME}.log"

            echo "=================================================="
            echo "Running recall test:"
            echo "  model      = ${MODEL}"
            echo "  dataset    = ${DATASET}"
            echo "  budget     = ${BUDGET}"
            echo "  recall dir = recall_list/${RECALL_NAME}"
            echo "=================================================="

            CHECK_RECALL=1 \
            RECALL_NAME=${RECALL_NAME} \
            MAX_CPU_IN_USE=${MAX_CPU_IN_USE} \
            RANDOM_SEED=${SEED} \
            PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
            CUDA_VISIBLE_DEVICES=${DEVICE} \
            TOKENIZERS_PARALLELISM=false \
            SUBVEC=${SUBVEC} \
            SUBBITS=${SUBBITS} \
            METRIC=${METRIC} \
            python vq_pred_recall.py \
                --model ${MODEL} \
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
                --exp_name ${EXP_NAME} \
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
                --data_dir ${DATA_DIR} \
                --datasets ${DATASET} \
                > ${LOG_FILE} 2>&1

            echo "Analyzing recall: ${RECALL_NAME}"

            python analyze_recall.py \
                --recall-name ${RECALL_NAME} \
                >> ${LOG_FILE} 2>&1

            echo "Done: ${RECALL_NAME}"
            echo "Log: ${LOG_FILE}"
            echo "Analyze output: recall_list/${RECALL_NAME}/analyze"

        done
    done
done

echo "All recall tests done."
