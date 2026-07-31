#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
XATTN_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
DATA_ROOT_DEFAULT=$(cd -- "$XATTN_ROOT/../../benchmarks/ruler/benchmark_root" && pwd)

cd "$XATTN_ROOT/eval/RULER/scripts"
export PYTHONPATH="$XATTN_ROOT:${PYTHONPATH:-}"
GPUS="${GPUS:-1}"
ROOT_DIR="${ROOT_DIR:-$DATA_ROOT_DEFAULT}"
MODEL_DIR="${MODEL_DIR:-}"
ENGINE_DIR="${ENGINE_DIR:-.}"
BATCH_SIZE="${BATCH_SIZE:-1}"

source config_models.sh
MODEL_NAME="llama-3.1-8b"


MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"

echo "TOKENIZER_TYPE: ${TOKENIZER_TYPE}"

source config_tasks.sh
BENCHMARK="synthetic"
METRIC="xattn" # Default: xattn
STRIDE=16

PRINT_DETAIL=${PRINT_DETAIL:-""}

THRESHOLD=${THRESHOLD:-""}


MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

TASKS="niah_single_3 vt fwe qa_1"
total_time=0



RES_DIR="${RES_DIR:-$XATTN_ROOT/output_budget/ruler}"


total_time=0
# for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
for MAX_SEQ_LENGTH in 65536; do
    for p in 0.8 0.85 0.9 0.95; do
        data_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
        pred_DIR="${RES_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"

        DATA_DIR="${data_DIR}/data"
        PRED_DIR="${pred_DIR}/pred"
        mkdir -p ${DATA_DIR}
        mkdir -p ${PRED_DIR}
        
        PRED_DIR_final="${PRED_DIR}/${p}"
        
        for TASK in ${TASKS}; do
            python budget_pred/call_api.py \
                --data_dir ${DATA_DIR} \
                --save_dir ${PRED_DIR_final} \
                --benchmark ${BENCHMARK} \
                --task ${TASK} \
                --server_type ${MODEL_FRAMEWORK} \
                --model_name_or_path ${MODEL_PATH} \
                --temperature ${TEMPERATURE} \
                --top_k ${TOP_K} \
                --p ${p} \
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
        python eval/evaluate.py \
            --data_dir ${PRED_DIR_final} \
            --benchmark ${BENCHMARK}
    done
done

echo "Total time spent on call_api: $total_time seconds"
