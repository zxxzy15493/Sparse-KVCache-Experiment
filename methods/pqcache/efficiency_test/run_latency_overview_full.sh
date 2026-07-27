#!/bin/bash
set -x
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export PYTHONPATH=..

mkdir -p log_latency

CSV_FILE=${CSV_FILE:-"log_latency/full_attention_latency_overview.csv"}
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
PY_SCRIPT=${PY_SCRIPT:-"latency_full_test_once.py"}

LLAMA_MODEL=${LLAMA_MODEL:-"meta-llama/Llama-3.1-8B-Instruct"}
QWEN_MODEL=${QWEN_MODEL:-"Qwen/Qwen2.5-7B-Instruct-1M"}

OUTPUT_LEN=${OUTPUT_LEN:-32}
WARMUP_ROUNDS=${WARMUP_ROUNDS:-2}
MEASURE_ROUNDS=${MEASURE_ROUNDS:-3}
DTYPE=${DTYPE:-"bfloat16"}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-"flash_attention_2"}

INPUT_LENS=(
    4096
    8192
    16384
    32768
    65536
    131072
    # 196608
    # 262144
)

run_one_model () {
    MODEL_NAME=$1

    MODEL_TAG=$(basename "${MODEL_NAME}")
    LOG_FILE="log_latency/full_overview_${MODEL_TAG}_out${OUTPUT_LEN}.log"

    echo "======================================================" | tee -a "${LOG_FILE}"
    echo "Running full attention overview: model=${MODEL_NAME}, output=${OUTPUT_LEN}" | tee -a "${LOG_FILE}"
    echo "Input lens: ${INPUT_LENS[*]}" | tee -a "${LOG_FILE}"
    echo "Dtype: ${DTYPE}" | tee -a "${LOG_FILE}"
    echo "Attention implementation: ${ATTN_IMPLEMENTATION}" | tee -a "${LOG_FILE}"
    echo "======================================================" | tee -a "${LOG_FILE}"

    python "${PY_SCRIPT}" \
        --model "${MODEL_NAME}" \
        --input-file "${INPUT_FILE}" \
        --input-lens "${INPUT_LENS[@]}" \
        --max-new-tokens "${OUTPUT_LEN}" \
        --csv "${CSV_FILE}" \
        --warmup-rounds "${WARMUP_ROUNDS}" \
        --measure-rounds "${MEASURE_ROUNDS}" \
        --dtype "${DTYPE}" \
        --attn-implementation "${ATTN_IMPLEMENTATION}" \
        >> "${LOG_FILE}" 2>&1
}

# run_one_model "${LLAMA_MODEL}"
run_one_model "${QWEN_MODEL}"

echo "All full attention overview latency experiments done. Results saved to ${CSV_FILE}"