#!/bin/bash
# set -euo pipefail

# Usage:
#   bash recall.sh <model_name> <benchmark_name> <synthetic_tasks|all> <token_budget>
#
# Examples:
#   bash recall.sh qwen2.5-7b synthetic niah_single_3 1024
#   bash recall.sh qwen2.5-7b synthetic niah_single_3,vt,fwe 2048
#   bash recall.sh qwen2.5-7b synthetic all 4096
#
# For recall recording, run it like:
#   CHECK_RECALL=1 RECALL_NAME=qwen2.5-7b_niah_single_3_bud1024 bash recall.sh qwen2.5-7b synthetic niah_single_3 1024

if [ $# -ne 4 ]; then
    echo "Usage: $0 <model_name> <benchmark_name> <synthetic_tasks|all> <token_budget>"
    echo "Example: $0 qwen2.5-7b synthetic niah_single_3 1024"
    echo "Example: $0 qwen2.5-7b synthetic niah_single_3,vt,fwe 2048"
    echo "Example: $0 qwen2.5-7b synthetic all 4096"
    exit 1
fi

# =========================
# Root directories
# =========================

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

GPUS=${GPUS:-1}
ROOT_DIR=${ROOT_DIR:-recall_test}
MODEL_DIR=${MODEL_DIR:-../..}
ENGINE_DIR=${ENGINE_DIR:-.}
BATCH_SIZE=${BATCH_SIZE:-1}

# =========================
# ClusterKV parameters
# =========================

TOKEN_BUDGET=${4}
NLIST=${NLIST:-400}
SINK=${SINK:-16}
HEAD_SEL=${HEAD_SEL:-truc}
BALANCE=${BALANCE:-}
FIT_ITER=${FIT_ITER:-20}
DIST_T=${DIST_T:-cosine}

# =========================
# System parameters
# =========================

DEVICE=${DEVICE:-0}
SEED=${SEED:-4321}
NUM_SAMPLES=${NUM_SAMPLES:-5}

# Keep this default consistent with the PQCache recall script.
# Override by: SEQ_LENGTHS_STR="131072" bash recall.sh ...
SEQ_LENGTHS_STR=${SEQ_LENGTHS_STR:-65536}
read -ra SEQ_LENGTHS <<< "${SEQ_LENGTHS_STR}"

# =========================
# Model and tokenizer
# =========================

source config_models.sh
SEQ_LENGTHS=(
    # 4096
    # 8192
    # 16384
    # 32768
    65536
    # 131072
)
MODEL_NAME=${1}
BENCHMARK=${2}
SYNTHETIC_TASKS_ARG=${3}

MODEL_CONFIG=$(MODEL_SELECT "${MODEL_NAME}" "${MODEL_DIR}" "${ENGINE_DIR}")
IFS=":" read -r MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "${MODEL_CONFIG}"

if [ -z "${MODEL_PATH}" ]; then
    echo "[ERROR] Model ${MODEL_NAME} is not supported"
    exit 1
fi

export OPENAI_API_KEY=${OPENAI_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}

# =========================
# Benchmark and tasks
# =========================

source config_tasks.sh
NUM_SAMPLES=5
# Synthetic tasks are read from the 3rd command-line argument.
# Use "all" to run all default synthetic tasks.
if [ "${SYNTHETIC_TASKS_ARG}" = "all" ]; then
    synthetic=(
        "niah_single_1"
        "niah_single_2"
        "niah_single_3"
        "niah_multikey_1"
        "niah_multikey_2"
        "niah_multikey_3"
        "niah_multivalue"
        "niah_multiquery"
        "vt"
        "cwe"
        "fwe"
        "qa_1"
        "qa_2"
    )
else
    IFS=',' read -ra synthetic <<< "${SYNTHETIC_TASKS_ARG}"
fi

if ! declare -p "${BENCHMARK}" >/dev/null 2>&1; then
    echo "[ERROR] Benchmark ${BENCHMARK} is not supported"
    exit 1
fi

declare -n TASKS="${BENCHMARK}"
if [ "${#TASKS[@]}" -eq 0 ]; then
    echo "[ERROR] Benchmark ${BENCHMARK} contains no tasks"
    exit 1
fi

echo "MODEL_NAME          : ${MODEL_NAME}"
echo "BENCHMARK           : ${BENCHMARK}"
echo "SYNTHETIC_TASKS_ARG : ${SYNTHETIC_TASKS_ARG}"
echo "TOKEN_BUDGET        : ${TOKEN_BUDGET}"
echo "SEQ_LENGTHS         : ${SEQ_LENGTHS[*]}"
echo "NUM_SAMPLES         : ${NUM_SAMPLES}"
echo "CHECK_RECALL        : ${CHECK_RECALL:-0}"
echo "RECALL_NAME         : ${RECALL_NAME:-}"

# =========================
# Start server if needed
# =========================

if [ "${MODEL_FRAMEWORK}" = "vllm" ]; then
    python pred/serve_vllm.py \
        --model="${MODEL_PATH}" \
        --tensor-parallel-size="${GPUS}" \
        --dtype bfloat16 \
        --disable-custom-all-reduce \
        &

elif [ "${MODEL_FRAMEWORK}" = "trtllm" ]; then
    python pred/serve_trt.py \
        --model_path="${MODEL_PATH}" \
        &

elif [ "${MODEL_FRAMEWORK}" = "sglang" ]; then
    python -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --tp "${GPUS}" \
        --port 5000 \
        --enable-flashinfer \
        &
    # use sglang/test/killall_sglang.sh to kill sglang server if it hangs
fi

# =========================
# Run client
# =========================

total_time=0
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}_budget${TOKEN_BUDGET}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="../../../../benchmarks/ruler/benchmark_root/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p "${DATA_DIR}" "${PRED_DIR}"

    for TASK in "${TASKS[@]}"; do
        echo
        echo "Running ${BENCHMARK} - ${TASK} - ${MAX_SEQ_LENGTH}"

        python data/prepare.py \
            --save_dir "${DATA_DIR}" \
            --benchmark "${BENCHMARK}" \
            --task "${TASK}" \
            --tokenizer_path "${TOKENIZER_PATH}" \
            --tokenizer_type "${TOKENIZER_TYPE}" \
            --max_seq_length "${MAX_SEQ_LENGTH}" \
            --model_template_type "${MODEL_TEMPLATE_TYPE}" \
            --num_samples "${NUM_SAMPLES}" \
            ${REMOVE_NEWLINE_TAB:-}

        start_time=$(date +%s)
        echo "Calling model API..."

        RANDOM_SEED="${SEED}" \
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
        CUDA_VISIBLE_DEVICES="${DEVICE}" \
        TOKENIZERS_PARALLELISM=false \
        python pred/call_api.py \
            --data_dir "${DATA_DIR}" \
            --save_dir "${PRED_DIR}" \
            --benchmark "${BENCHMARK}" \
            --task "${TASK}" \
            --server_type "${MODEL_FRAMEWORK}" \
            --model_name_or_path "${MODEL_PATH}" \
            --model_name "${MODEL_NAME}" \
            --temperature "${TEMPERATURE}" \
            --top_k "${TOP_K}" \
            --top_p "${TOP_P}" \
            --batch_size "${BATCH_SIZE}" \
            ${STOP_WORDS:-} \
            --token_budget "${TOKEN_BUDGET}" \
            --nlist "${NLIST}" \
            --sink "${SINK}" \
            --head_sel "${HEAD_SEL}" \
            --fit_iter "${FIT_ITER}" \
            --dist_t "${DIST_T}" \
            ${BALANCE:+--balance}

        end_time=$(date +%s)
        time_diff=$((end_time - start_time))
        total_time=$((total_time + time_diff))
        echo "Time spent on ${TASK}: ${time_diff} seconds"
    done

    python eval/evaluate.py \
        --data_dir "${PRED_DIR}" \
        --benchmark "${BENCHMARK}"
done

echo "Total time spent on call_api: ${total_time} seconds"
