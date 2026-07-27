#!/bin/bash
set -x
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:128"}
export PYTHONPATH=..

mkdir -p log_latency

CSV_FILE=${CSV_FILE:-"log_latency/latency_tradeoff_pq.csv"}
INPUT_FILE=${INPUT_FILE:-"./myinput.txt"}
# Ensure myinput.txt exists at INPUT_FILE; copy from benchmarks/ via relative path if missing
if [ ! -f "${INPUT_FILE}" ]; then
    _SRC="$(dirname "$0")/../../../benchmarks/myinput.txt"
    if [ -f "${_SRC}" ]; then
        case "${INPUT_FILE}" in
            /*) _DEST="${INPUT_FILE}" ;;
            *)  _DEST="$(dirname "$0")/${INPUT_FILE#./}" ;;
        esac
        mkdir -p "$(dirname "${_DEST}")"
        cp "${_SRC}" "${_DEST}"
    fi
fi
unset _SRC _DEST
PY_SCRIPT=${PY_SCRIPT:-"latency_test_once.py"}

LLAMA_MODEL=${LLAMA_MODEL:-"meta-llama/Llama-3.1-8B-Instruct"}
QWEN_MODEL=${QWEN_MODEL:-"Qwen/Qwen2.5-7B-Instruct-1M"}

OUTPUT_LEN=${OUTPUT_LEN:-32}
WARMUP_ROUNDS=${WARMUP_ROUNDS:-2}
MEASURE_ROUNDS=${MEASURE_ROUNDS:-3}

BUDGETS_5200=(128 256 512 1024)
BUDGETS_29900=(128 256 512 1024)
run_one () {
    MODEL_FAMILY=$1
    MODEL_NAME=$2
    INPUT_LEN=$3
    BUDGET=$4
    GROUP_NAME=$5

    LOG_FILE="log_latency/budget_${GROUP_NAME}_${MODEL_FAMILY}_in${INPUT_LEN}_out${OUTPUT_LEN}_budget${BUDGET}.log"

    echo "======================================================" | tee -a "${LOG_FILE}"
    echo "Running budget sweep: group=${GROUP_NAME}, ${MODEL_FAMILY}, input=${INPUT_LEN}, output=${OUTPUT_LEN}, budget=${BUDGET}" | tee -a "${LOG_FILE}"
    echo "======================================================" | tee -a "${LOG_FILE}"

    python "${PY_SCRIPT}" \
        --model "${MODEL_NAME}" \
        --input-file "${INPUT_FILE}" \
        --input-lens "${INPUT_LEN}" \
        --max-new-tokens "${OUTPUT_LEN}" \
        --budget "${BUDGET}" \
        --csv "${CSV_FILE}" \
        --warmup-rounds "${WARMUP_ROUNDS}" \
        --measure-rounds "${MEASURE_ROUNDS}" \
        >> "${LOG_FILE}" 2>&1
}

run_budget_group () {
    GROUP_NAME=$1
    INPUT_LEN=$2
    shift 2
    BUDGETS=("$@")

    for BUDGET in "${BUDGETS[@]}"; do
        run_one llama "${LLAMA_MODEL}" "${INPUT_LEN}" "${BUDGET}" "${GROUP_NAME}"
        run_one qwen "${QWEN_MODEL}" "${INPUT_LEN}" "${BUDGET}" "${GROUP_NAME}"
    done
}

run_budget_group 5200 5200 "${BUDGETS_5200[@]}"
run_budget_group 29900 29900 "${BUDGETS_29900[@]}"

echo "All budget latency experiments done. Results saved to ${CSV_FILE}"