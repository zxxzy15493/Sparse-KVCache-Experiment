#!/bin/bash
set -xeuo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export PYTHONPATH=${PYTHONPATH:-".."}
if [[ "${DEBUG:-0}" == "1" ]]; then
    export CUDA_LAUNCH_BLOCKING=1
fi

mkdir -p log_mem

CSV_FILE=${CSV_FILE:-"log_mem/my_mem_results.csv"}
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
PY_SCRIPT=${PY_SCRIPT:-"my_mem_test_once.py"}

RUN_LLAMA=${RUN_LLAMA:-1}
RUN_QWEN=${RUN_QWEN:-1}
OFFLOAD=${OFFLOAD:-1}

LLAMA_MODEL=${LLAMA_MODEL:-"meta-llama/Llama-3.1-8B-Instruct"}
QWEN_MODEL=${QWEN_MODEL:-"Qwen/Qwen2.5-7B-Instruct-1M"}

METHOD=${METHOD:-"clusterkv"}
IMPL=${IMPL:-"myllama"}
DTYPE=${DTYPE:-"float16"}

NLIST=${NLIST:-200}
NITER=${NITER:-20}
WINDOW_NLIST=${WINDOW_NLIST:-8}

run_one () {
    MODEL_FAMILY=$1
    MODEL_NAME=$2
    INPUT_LEN=$3
    OUTPUT_LEN=$4
    BUDGET=$5

    LOG_FILE="log_mem/my_${MODEL_FAMILY}_in${INPUT_LEN}_out${OUTPUT_LEN}_budget${BUDGET}.log"

    OFFLOAD_ARGS=()
    if [[ "${OFFLOAD}" == "1" ]]; then
        OFFLOAD_ARGS=(--offload)
    fi

    echo "======================================================" | tee -a "${LOG_FILE}"
    echo "Running my ${MODEL_FAMILY}, input=${INPUT_LEN}, output=${OUTPUT_LEN}, budget=${BUDGET}, impl=${IMPL}, offload=${OFFLOAD}" | tee -a "${LOG_FILE}"
    echo "======================================================" | tee -a "${LOG_FILE}"

    python "${PY_SCRIPT}" \
        --model "${MODEL_NAME}" \
        --model-family "${MODEL_FAMILY}" \
        --impl "${IMPL}" \
        --input-file "${INPUT_FILE}" \
        --input-len "${INPUT_LEN}" \
        --max-new-tokens "${OUTPUT_LEN}" \
        --budget "${BUDGET}" \
        --csv "${CSV_FILE}" \
        --method "${METHOD}" \
        --dtype "${DTYPE}" \
        --nlist "${NLIST}" \
        --niter "${NITER}" \
        --window-nlist "${WINDOW_NLIST}" \
        "${OFFLOAD_ARGS[@]}" \
        --device cuda:0 \
        >> "${LOG_FILE}" 2>&1
}

if [[ "${RUN_LLAMA}" == "1" ]]; then
    run_one llama "${LLAMA_MODEL}" 1024 2 64
    run_one llama "${LLAMA_MODEL}" 1024 2 512
    run_one llama "${LLAMA_MODEL}" 1024 4096 64
    run_one llama "${LLAMA_MODEL}" 1024 4096 512
    
fi

if [[ "${RUN_QWEN}" == "1" ]]; then
    run_one qwen "${QWEN_MODEL}" 1024 2 64
    run_one qwen "${QWEN_MODEL}" 1024 2 512
    run_one qwen "${QWEN_MODEL}" 1024 4096 64
    run_one qwen "${QWEN_MODEL}" 1024 4096 512
fi

echo "All my memory experiments done. Results saved to ${CSV_FILE}"
