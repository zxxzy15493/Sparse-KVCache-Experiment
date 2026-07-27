#!/bin/bash
set -x
set -e

# Usage:
# bash run_longbench_recall_topk.sh
# bash run_longbench_recall_topk.sh 0
#
# Variant of run_longbench_recall.sh that exercises the **topk** (no_drop_lb)
# compressor with sink=0, recent=0 — i.e. an oracle topk over the full KV cache
# (budget = high_score_budget, no sink/recent slices).
# Drives the new CHECK_RECALL instrumentation added to
# fullKVLimitBasedCompressor.restore in vq_method/baseline_compressor.py.

DEVICE=${1:-0}
DATA_DIR="../../benchmarks/Longbench_recall"

SEED="4321"
COMPRESSOR=${COMPRESSOR:-"no_drop_lb"}
TOPK_VARIANT=${TOPK_VARIANT:-"topk"}
EXP_NAME=${EXP_NAME:-"recall_test_${TOPK_VARIANT}"}

COMPRESS=0.1
CORE_OFFSET=0
TOPK=0.5
RECENT_RATIO=0.5
SINK_SIZE=0
RECENT_SIZE=0
FIXBUDGET="--fixbudget"

# PQ-only knobs — kept for parser compatibility (the no_drop_lb path ignores
# them, but the call to vq_pred_recall.py still accepts the flags).
SUBVEC=2
SUBBITS=6
TOPR=1
GQA="True"
MEAN_V_TRICK="False"
MAX_ITER=0

MAX_CPU_IN_USE=16
DROP=0
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

            RECALL_NAME="${MODEL}_${DATASET}_budget_${BUDGET}_${TOPK_VARIANT}_recall_${COMPRESSOR}"
            LOG_FILE="logs_recall/${RECALL_NAME}.log"

            echo "=================================================="
            echo "Running recall test (topk/no_drop_lb):"
            echo "  model      = ${MODEL}"
            echo "  dataset    = ${DATASET}"
            echo "  budget     = ${BUDGET}"
            echo "  sink       = ${SINK_SIZE}"
            echo "  recent     = ${RECENT_SIZE}"
            echo "  recall dir = recall_list/${RECALL_NAME}"
            echo "=================================================="

            CHECK_RECALL=1 \
            RECALL_NAME=${RECALL_NAME} \
            MAX_CPU_IN_USE=${MAX_CPU_IN_USE} \
            RANDOM_SEED=${SEED} \
            PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
            CUDA_VISIBLE_DEVICES=${DEVICE} \
            TOKENIZERS_PARALLELISM=false \
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
                --gqa ${GQA} \
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

echo "All topk/no_drop_lb recall tests done."
