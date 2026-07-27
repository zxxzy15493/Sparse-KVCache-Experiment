#!/bin/bash
set -euo pipefail

# Usage:
#   bash recall_all_topk.sh
#
# Variant of recall_all.sh that drives the **topk** (no_drop_lb) recall sweep
# with sink=0, recent=0 (oracle topk over the full KV cache).
# Calls recall_topk.sh, then runs analyze_recall.py.
#
# Output CSV: recall_list/<RECALL_NAME>/<RECALL_NAME>_<TIMESTAMP>.csv
# RECALL_NAME format: <model>_<task>_bud<budget>_recall_<compressor>

# =========================
# Config
# =========================

# Pin to the new recall_topk.sh — never silently fall back to recall.sh.
RECALL_SCRIPT=${RECALL_SCRIPT:-recall_topk.sh}
ANALYZE_SCRIPT=${ANALYZE_SCRIPT:-analyze_recall.py}
RECALL_SUFFIX=${RECALL_SUFFIX:-no_drop_lb}

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

# PQ budgets (topk / no_drop_lb budget = high_score_budget)
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

            # Keep the compressor suffix in the directory name so analysis reads
            # the CSV written by this exact run.
            RECALL_NAME="${MODEL_NAME}_${TASK}_bud${PQ_BUDGET}_recall_${RECALL_SUFFIX}"

            echo
            echo "============================================================"
            echo "Running recall (topk / no_drop_lb, sink=0, recent=0)"
            echo "MODEL_NAME   : ${MODEL_NAME}"
            echo "BENCHMARK    : ${BENCHMARK_NAME}"
            echo "TASK         : ${TASK}"
            echo "PQ_BUDGET    : ${PQ_BUDGET}"
            echo "RECALL_NAME  : ${RECALL_NAME}"
            echo "RECALL_SCRIPT: ${RECALL_SCRIPT}"
            echo "============================================================"

            # CHECK_RECALL is the variable read by both:
            #   - pq_search.py:23          (CHECK_RECALL=1 → recall CSV in pq path)
            #   - baseline_compressor.py:21 (CHECK_RECALL=1 → recall CSV in no_drop_lb path)
            # We pass it through to the bash invocation so recall_topk.sh
            # exports it into the env that call_api.py sees.
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
echo "All topk/no_drop_lb recall runs and analyses finished."
