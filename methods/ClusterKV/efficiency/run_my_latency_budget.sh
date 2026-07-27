#!/bin/bash
set -x
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:128"}
export PYTHONPATH=${PYTHONPATH:-".."}

mkdir -p log_latency ../codexlog

CSV_FILE=${CSV_FILE:-"log_latency/myllama_latency_budget.csv"}
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
PY_SCRIPT=${PY_SCRIPT:-"./my_textgen.py"}

LLAMA_MODEL=${LLAMA_MODEL:-"meta-llama/Llama-3.1-8B-Instruct"}
QWEN_MODEL=${QWEN_MODEL:-"Qwen/Qwen2.5-7B-Instruct-1M"}
RUN_QWEN=${RUN_QWEN:-1}

OUTPUT_LEN=${OUTPUT_LEN:-32}
WARMUP_ROUNDS=${WARMUP_ROUNDS:-2}
MEASURE_ROUNDS=${MEASURE_ROUNDS:-3}

METHOD=${METHOD:-"clusterkv"}
IMPL=${IMPL:-"myllama"}
DTYPE=${DTYPE:-"float16"}
NLIST=${NLIST:-200}
NITER=${NITER:-20}
SINK=${SINK:-16}
WINDOW=${WINDOW:-320}
WINDOW_NLIST=${WINDOW_NLIST:-8}
OFFLOAD=${OFFLOAD:-1}

BUDGETS_4K=(128 256 512 1024)
BUDGETS_64K=(128 384 1024 4096)
# BUDGETS_64K=(128 384 1024 4096 16384)

EXTRA_ARGS=(
    --method "${METHOD}"
    --impl "${IMPL}"
    --dtype "${DTYPE}"
    --nlist "${NLIST}"
    --niter "${NITER}"
    --sink "${SINK}"
    --window "${WINDOW}"
    --window-nlist "${WINDOW_NLIST}"
)

if [[ "${OFFLOAD}" == "1" ]]; then
    EXTRA_ARGS+=(--offload)
fi

run_one () {
    MODEL_NAME=$1
    INPUT_LEN=$2
    BUDGET=$3
    GROUP_NAME=$4

    MODEL_TAG=$(basename "${MODEL_NAME}")
    LOG_FILE="log_latency/myllama_budget_${GROUP_NAME}_${MODEL_TAG}_in${INPUT_LEN}_out${OUTPUT_LEN}_budget${BUDGET}.log"

    echo "======================================================" | tee -a "${LOG_FILE}"
    echo "Running MyLlama/MyQwen budget sweep: group=${GROUP_NAME}, model=${MODEL_NAME}, input=${INPUT_LEN}, output=${OUTPUT_LEN}, budget=${BUDGET}" | tee -a "${LOG_FILE}"
    echo "Args: method=${METHOD}, impl=${IMPL}, dtype=${DTYPE}, nlist=${NLIST}, niter=${NITER}, sink=${SINK}, window=${WINDOW}, window_nlist=${WINDOW_NLIST}, offload=${OFFLOAD}" | tee -a "${LOG_FILE}"
    echo "======================================================" | tee -a "${LOG_FILE}"

    python "${PY_SCRIPT}" \
        --model "${MODEL_NAME}" \
        --input-file "${INPUT_FILE}" \
        --input-len "${INPUT_LEN}" \
        --max-new-tokens "${OUTPUT_LEN}" \
        --budget "${BUDGET}" \
        --csv "${CSV_FILE}" \
        --warmup-rounds "${WARMUP_ROUNDS}" \
        --measure-rounds "${MEASURE_ROUNDS}" \
        "${EXTRA_ARGS[@]}" \
        >> "${LOG_FILE}" 2>&1
}

run_input_group () {
    INPUT_LEN=$1
    GROUP_NAME=$2
    shift 2
    BUDGETS=("$@")

    for BUDGET in "${BUDGETS[@]}"; do
        run_one "${LLAMA_MODEL}" "${INPUT_LEN}" "${BUDGET}" "${GROUP_NAME}"
        if [[ "${RUN_QWEN}" == "1" ]]; then
            run_one "${QWEN_MODEL}" "${INPUT_LEN}" "${BUDGET}" "${GROUP_NAME}"
        fi
    done
}

run_input_group 4096 4k "${BUDGETS_4K[@]}"
run_input_group 65536 64k "${BUDGETS_64K[@]}"

echo "All MyLlama/MyQwen budget latency experiments done. Results saved to ${CSV_FILE}"
