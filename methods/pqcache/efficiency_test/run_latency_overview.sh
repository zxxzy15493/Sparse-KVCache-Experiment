#!/bin/bash
set -x
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:128"}
export PYTHONPATH=..
mkdir -p log_latency

CSV_FILE=${CSV_FILE:-"log_latency/latency_overview.csv"}
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

# rm -f "${CSV_FILE}"

LLAMA_MODEL=${LLAMA_MODEL:-"meta-llama/Llama-3.1-8B-Instruct"}
QWEN_MODEL=${QWEN_MODEL:-"Qwen/Qwen2.5-7B-Instruct-1M"}

OUTPUT_LEN=${OUTPUT_LEN:-32}
BUDGET=${BUDGET:-1024}
WARMUP_ROUNDS=${WARMUP_ROUNDS:-2}
MEASURE_ROUNDS=${MEASURE_ROUNDS:-3}

export MAX_CPU_IN_USE=${MAX_CPU_IN_USE:-16}
export CORE_OFFSET=${CORE_OFFSET:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# DEBUG_DISABLED: these extra thread limits were removed while narrowing CPU utilization behavior.
# export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
# export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
# export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
# export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}

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
    LOG_FILE="log_latency/overview_${MODEL_TAG}_out${OUTPUT_LEN}_budget${BUDGET}.log"

    echo "======================================================" | tee -a "${LOG_FILE}"
    echo "Running overview: model=${MODEL_NAME}, output=${OUTPUT_LEN}, budget=${BUDGET}" | tee -a "${LOG_FILE}"
    echo "Input lens: ${INPUT_LENS[*]}" | tee -a "${LOG_FILE}"
    echo "======================================================" | tee -a "${LOG_FILE}"

    python "${PY_SCRIPT}" \
        --model "${MODEL_NAME}" \
        --input-file "${INPUT_FILE}" \
        --input-lens "${INPUT_LENS[@]}" \
        --max-new-tokens "${OUTPUT_LEN}" \
        --budget "${BUDGET}" \
        --csv "${CSV_FILE}" \
        --warmup-rounds "${WARMUP_ROUNDS}" \
        --measure-rounds "${MEASURE_ROUNDS}" \
        >> "${LOG_FILE}" 2>&1
}

run_one_model "${LLAMA_MODEL}"
run_one_model "${QWEN_MODEL}"

echo "All overview latency experiments done. Results saved to ${CSV_FILE}"
