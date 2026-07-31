#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
XATTN_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
REPO_ROOT=$(cd -- "$XATTN_ROOT/../.." && pwd)

cd "$XATTN_ROOT"
BLOCK_SPARSE_BUILD="$(find ./Block-Sparse-Attention/build -maxdepth 1 -type d -name 'lib.*' -print -quit 2>/dev/null || true)"
export PYTHONPATH=".:./Block-Sparse-Attention:${BLOCK_SPARSE_BUILD:-}:${PYTHONPATH:-}"

GPUS="${GPUS:-1}"
ROOT_DIR="${ROOT_DIR:-$REPO_ROOT/benchmarks/Ruler_recall}"
MODEL_DIR="${MODEL_DIR:-}"
ENGINE_DIR="${ENGINE_DIR:-.}"
BATCH_SIZE="${BATCH_SIZE:-1}"

source ./eval/recall/RULER/scripts/config_models.sh
MODEL_CONFIG_NAME="qwen-2.5-7b-1m"
MODEL_DATA_NAME="qwen-2.5-7b-1m"


MODEL_CONFIG=$(MODEL_SELECT ${MODEL_CONFIG_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"

echo "TOKENIZER_TYPE: ${TOKENIZER_TYPE}"

source ./eval/recall/RULER/scripts/config_tasks.sh
BENCHMARK="synthetic"
METRIC="xattn"
STRIDE=16

PRINT_DETAIL=${PRINT_DETAIL:-""}

THRESHOLD=${THRESHOLD:-""}


MODEL_PATH="Qwen/Qwen2.5-7B-Instruct-1M"

TASKS="niah_single_3 vt fwe"
total_time=0

RES_DIR="./efficiency"


total_time=0
for MAX_SEQ_LENGTH in 65536; do
    for p in 0.8 0.85 0.9 0.95; do
        data_DIR="${ROOT_DIR}/${MODEL_DATA_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
        DATA_DIR="${data_DIR}/data"
        # mkdir -p ${DATA_DIR}
        
        for TASK in ${TASKS}; do
            start_time=$(date +%s)
            save_path="./efficiency/attn_score/ruler/qwen/${p}"
            python ./eval/recall/RULER/scripts/budget_pred/qwen_call_api.py \
                --data_dir ${DATA_DIR} \
                --save_dir tmp/xattn_unused_pred \
                --benchmark ${BENCHMARK} \
                --task ${TASK} \
                --server_type ${MODEL_FRAMEWORK} \
                --model_name_or_path ${MODEL_PATH} \
                --temperature ${TEMPERATURE} \
                --top_k ${TOP_K} \
                --p ${p} \
                --save_path ${save_path} \
                --top_p ${TOP_P} \
                --batch_size ${BATCH_SIZE} \
                ${STOP_WORDS} \
                --metric ${METRIC} \
                ${THRESHOLD} \
                --stride ${STRIDE} \
                ${PRINT_DETAIL}
            end_time=$(date +%s)
            time_diff=$((end_time - start_time))
            total_time=$((total_time + time_diff))
        done
    done
done

echo "Total time spent on call_api: $total_time seconds"
