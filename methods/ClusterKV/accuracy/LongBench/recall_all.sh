#!/bin/bash
set -euo pipefail

# Usage:
#   bash recall_all_longbench.sh
#
# It will run:
#   CHECK_RECALL=1 RECALL_NAME=<model>_<dataset>_bud<budget> \
#   python pred.py --model <model> --task <dataset> --token_budget <budget> --cluster
#
# then optionally analyze:
#   python analyze_recall.py --recall-name <RECALL_NAME>

# =========================
# Config
# =========================

PRED_SCRIPT=${PRED_SCRIPT:-recall_pred.py}
ANALYZE_SCRIPT=${ANALYZE_SCRIPT:-analyze_recall.py}
RUN_ANALYZE=${RUN_ANALYZE:-1}

# Models
MODELS=(
    "llama3.1-8b-chat-32k"
    "qwen2.5-7b-chat-32k"
)

# LongBench datasets
TASKS=(
    "qasper"
    "narrativeqa"
)

# Token budgets
BUDGETS=(
    128
    256
    512
    1024
)

# =========================
# Checks
# =========================

if [ ! -f "${PRED_SCRIPT}" ]; then
    echo "[ERROR] Cannot find ${PRED_SCRIPT}"
    exit 1
fi

if [ "${RUN_ANALYZE}" = "1" ] && [ ! -f "${ANALYZE_SCRIPT}" ]; then
    echo "[ERROR] Cannot find ${ANALYZE_SCRIPT}"
    exit 1
fi

# =========================
# Run
# =========================

for MODEL_NAME in "${MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        for TOKEN_BUDGET in "${BUDGETS[@]}"; do

            RECALL_NAME="${MODEL_NAME}_${TASK}_bud${TOKEN_BUDGET}"

            echo
            echo "============================================================"
            echo "Running ClusterKV LongBench recall"
            echo "MODEL_NAME   : ${MODEL_NAME}"
            echo "TASK         : ${TASK}"
            echo "TOKEN_BUDGET : ${TOKEN_BUDGET}"
            echo "RECALL_NAME  : ${RECALL_NAME}"
            echo "============================================================"

            # CHECK_RECALL and RECALL_NAME are consumed by the Python recall-recording code.
            CHECK_RECALL=1 \
            RECALL_NAME="${RECALL_NAME}" \
            python "${PRED_SCRIPT}" \
                --model "${MODEL_NAME}" \
                --task "${TASK}" \
                --token_budget "${TOKEN_BUDGET}" \
                --cluster

            if [ "${RUN_ANALYZE}" = "1" ]; then
                echo
                echo "Analyzing ${RECALL_NAME} ..."
                python "${ANALYZE_SCRIPT}" --recall-name "${RECALL_NAME}"
            fi

            echo
            echo "[DONE] ${RECALL_NAME}"
        done
    done
done

echo
echo "All ClusterKV LongBench recall runs and analyses finished."