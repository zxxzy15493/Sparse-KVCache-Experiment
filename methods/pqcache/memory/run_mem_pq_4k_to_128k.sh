#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

mkdir -p "${SCRIPT_DIR}/log_mem"

CSV_FILE="${SCRIPT_DIR}/log_mem/mem_pq_4k_to_128k_results.csv"
INPUT_FILE="${ROOT_DIR}/myinput.txt"
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

LLAMA_MODEL="meta-llama/Llama-3.1-8B-Instruct"
QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct-1M"
OUTPUT_LEN=2
BUDGET=1024
# INPUT_LENS=(4096 8192 16384 32768 65536 131072)
INPUT_LENS=(262144)
FAILED=0

run_one () {
    MODEL_FAMILY=$1
    MODEL_NAME=$2
    INPUT_LEN=$3

    LOG_FILE="${SCRIPT_DIR}/log_mem/${MODEL_FAMILY}_pq_in${INPUT_LEN}_out${OUTPUT_LEN}_budget${BUDGET}.log"

    echo "======================================================" | tee -a "${LOG_FILE}"
    echo "Running PQCache: ${MODEL_FAMILY}, input=${INPUT_LEN}, output=${OUTPUT_LEN}, budget=${BUDGET}" | tee -a "${LOG_FILE}"
    echo "======================================================" | tee -a "${LOG_FILE}"

    if ! "${PYTHON}" "${SCRIPT_DIR}/mem_test_once.py" \
        --model "${MODEL_NAME}" \
        --model-family "${MODEL_FAMILY}" \
        --input-file "${INPUT_FILE}" \
        --input-len "${INPUT_LEN}" \
        --max-new-tokens "${OUTPUT_LEN}" \
        --budget "${BUDGET}" \
        --csv "${CSV_FILE}" \
        >> "${LOG_FILE}" 2>&1; then
        echo "FAILED: ${MODEL_FAMILY}, input=${INPUT_LEN}, output=${OUTPUT_LEN}, budget=${BUDGET}" | tee -a "${LOG_FILE}"
        FAILED=1
    fi
}

for INPUT_LEN in "${INPUT_LENS[@]}"; do
    run_one llama "${LLAMA_MODEL}" "${INPUT_LEN}"
done

for INPUT_LEN in "${INPUT_LENS[@]}"; do
    run_one qwen "${QWEN_MODEL}" "${INPUT_LEN}"
done

echo "PQCache memory experiments done. Results saved to ${CSV_FILE}"
exit "${FAILED}"
