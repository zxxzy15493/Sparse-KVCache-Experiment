
set -euo pipefail

MODEL_NAME=${1}
BENCHMARK=${2}
TASK=${3}

MAX_LEN=${4:-65536}
NUM_EXAMPLES=${5:--1}

DATA_ROOT="../../../../benchmarks"

if [ "$BENCHMARK" = "LongBench" ]; then
    DATA_DIR="${DATA_ROOT}/LongBench"
elif [ "$BENCHMARK" = "Synthetic" ]; then
    case "$MODEL_NAME" in
        llama*) MODEL_SHORT="llama-3.1-8b" ;;
        # Qwen3-4B and Qwen3-32B share the Qwen3 chat/tokenizer format.
        # This workspace currently provides the Qwen3-formatted RULER data
        # under the qwen3-32b directory.
        qwen3-4b|qwen3-32b) MODEL_SHORT="qwen3-32b" ;;
        qwen*)  MODEL_SHORT="qwen-2.5-7b-1m" ;;
        glm*)      MODEL_SHORT="glm-4-9b" ;;
        *)         MODEL_SHORT="$MODEL_NAME" ;;
    esac
    DATA_DIR="${DATA_ROOT}/ruler/benchmark_root/${MODEL_SHORT}/synthetic/${MAX_LEN}/data/${TASK}"
else
    echo "Unknown benchmark: $BENCHMARK"
    exit 1
fi

if [ "$BENCHMARK" = "Synthetic" ]; then
    RESULT_DIR="./results/pred/${MODEL_NAME}/${BENCHMARK}/${MAX_LEN}"
    LOG_DIR="./log/pred/${MODEL_NAME}/${BENCHMARK}/${MAX_LEN}"
else
    RESULT_DIR="./results/pred/${MODEL_NAME}/${BENCHMARK}"
    LOG_DIR="./log/pred/${MODEL_NAME}/${BENCHMARK}"
fi

mkdir -p ${RESULT_DIR}
mkdir -p ${LOG_DIR}

# LongBench has a Python dataset loading script.  Use project-local writable
# caches instead of assuming the shared HF_HOME is writable.  Reuse a prepared
# LongBench cache from a previous model run when available, so each model does
# not download the same data.zip again.
HF_CACHE_ROOT="./.cache/huggingface/${BENCHMARK}"
EXISTING_LONG_BENCH_CACHE=""
if [ "${BENCHMARK}" = "LongBench" ] && [ -z "${HF_DATASETS_CACHE:-}" ]; then
    EXISTING_LONG_BENCH_DATA_DIR=$(find ./results/pred -type d -path '*/LongBench/.hf_datasets/THUDM___long_bench' -print -quit)
    if [ -n "${EXISTING_LONG_BENCH_DATA_DIR}" ]; then
        EXISTING_LONG_BENCH_CACHE=$(dirname "${EXISTING_LONG_BENCH_DATA_DIR}")
    fi
fi
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_CACHE_ROOT}/modules}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${EXISTING_LONG_BENCH_CACHE:-${HF_CACHE_ROOT}/datasets}}"

MODEL_PATH=$(python -c "import json; print(json.load(open('config/model2path.json'))['${MODEL_NAME}'])")

echo "Parameters: ${MODEL_PATH} ${BENCHMARK} ${TASK} ${NUM_EXAMPLES} ${MAX_LEN}"
echo "DATA_DIR: ${DATA_DIR}"
echo "Start to predict..."

python -u pred.py \
    --model_name ${MODEL_PATH} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    --save_dir ${RESULT_DIR} \
    --dtype bf16 \
    --device auto \
    --num_samples ${NUM_EXAMPLES} \
    --data_dir ${DATA_DIR} \
    >${LOG_DIR}/${TASK}.log 2>&1

echo "Start to evaluate..."
if [ "$BENCHMARK" = "LongBench" ]; then
    python -u eval.py \
        --model ${MODEL_NAME} \
        --benchmark ${BENCHMARK} \
        --task ${TASK} \
        --save_dir ${RESULT_DIR} \
        --data_dir ${DATA_DIR}
    echo "Results:"
    cat "${RESULT_DIR}/result.json"
elif [ "$BENCHMARK" = "Synthetic" ]; then
    python -u evaluate.py \
        --data-dir ${RESULT_DIR} \
        --tasks ${TASK}
    echo "Results:"
    cat "${RESULT_DIR}/summary.csv"
fi
