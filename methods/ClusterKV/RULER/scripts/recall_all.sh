#!/bin/bash
set -euo pipefail

# Usage:
#   bash recall_all.sh
#
# It will run:
#   CHECK_RECALL=1 RECALL_NAME=<model>_<task>_bud<budget> bash recall.sh <model_name> <benchmark_name> <synthetic_tasks|all> <token_budget>
# then analyze:
#   python analyze_recall.py --recall-name <RECALL_NAME>

# =========================
# Config
# =========================

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

RECALL_SCRIPT=${RECALL_SCRIPT:-recall.sh}
ANALYZE_SCRIPT=${ANALYZE_SCRIPT:-analyze_recall.py}
RUN_ANALYZE=${RUN_ANALYZE:-1}

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

# ClusterKV token budgets
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
            echo "Running ClusterKV recall"
            echo "MODEL_NAME   : ${MODEL_NAME}"
            echo "BENCHMARK    : ${BENCHMARK_NAME}"
            echo "TASK         : ${TASK}"
            echo "TOKEN_BUDGET : ${TOKEN_BUDGET}"
            echo "RECALL_NAME  : ${RECALL_NAME}"
            echo "============================================================"

            # CHECK_RECALL and RECALL_NAME are consumed by the Python recall-recording code.
            CHECK_RECALL=1 \
            RECALL_NAME="${RECALL_NAME}" \
            bash "${RECALL_SCRIPT}" "${MODEL_NAME}" "${BENCHMARK_NAME}" "${TASK}" "${TOKEN_BUDGET}"

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
echo "All ClusterKV recall runs and analyses finished."
