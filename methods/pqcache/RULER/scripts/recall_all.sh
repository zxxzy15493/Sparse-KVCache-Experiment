#!/bin/bash
set -euo pipefail

# Usage:
#   bash run_recall_all.sh
#
# It will run:
#   CHECK_RECALL=1 RECALL_NAME=<model>_<task>_bud<budget> bash recall.sh <model_name> <benchmark_name> <synthetic_tasks|all> <pq_budget>
# then analyze:
#   python analyze_recall.py --recall-name <RECALL_NAME>

# =========================
# Config
# =========================

RECALL_SCRIPT=${RECALL_SCRIPT:-recall.sh}
ANALYZE_SCRIPT=${ANALYZE_SCRIPT:-analyze_recall.py}
RECALL_COMPRESSOR=${RECALL_COMPRESSOR:-pq_search}

# benchmark_name
BENCHMARK_NAME=${BENCHMARK_NAME:-synthetic}

# Models
MODELS=(
    "qwen-2.5-7b-1m"
    "llama-3.1-8b"
)

# Synthetic tasks / datasets
TASKS=(
    "niah_single_3"
    "vt"
    "fwe"
)

# PQ budgets
BUDGETS=(
    128
    384
    1024
    4096
)

# =========================
# Checks
# =========================

if [ ! -f "${RECALL_SCRIPT}" ]; then
    echo "[ERROR] Cannot find ${RECALL_SCRIPT}"
    exit 1
fi

if [ ! -f "${ANALYZE_SCRIPT}" ]; then
    echo "[ERROR] Cannot find ${ANALYZE_SCRIPT}"
    exit 1
fi

# =========================
# Run
# =========================

for MODEL_NAME in "${MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        for PQ_BUDGET in "${BUDGETS[@]}"; do

            RECALL_NAME="${MODEL_NAME}_${TASK}_bud${PQ_BUDGET}_recall_${RECALL_COMPRESSOR}"

            echo
            echo "============================================================"
            echo "Running recall"
            echo "MODEL_NAME   : ${MODEL_NAME}"
            echo "BENCHMARK    : ${BENCHMARK_NAME}"
            echo "TASK         : ${TASK}"
            echo "PQ_BUDGET    : ${PQ_BUDGET}"
            echo "RECALL_NAME  : ${RECALL_NAME}"
            echo "============================================================"

            # CHECK_RECALL is the variable used in your Python code.
            CHECK_RECALL=1 \
            RECALL_NAME="${RECALL_NAME}" \
            bash "${RECALL_SCRIPT}" "${MODEL_NAME}" "${BENCHMARK_NAME}" "${TASK}" "${PQ_BUDGET}"

            echo
            echo "Analyzing ${RECALL_NAME} ..."
            python "${ANALYZE_SCRIPT}" --recall-name "${RECALL_NAME}"

            echo
            echo "[DONE] ${RECALL_NAME}"
        done
    done
done

echo
echo "All recall runs and analyses finished."
