#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/../../

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR="budget"
BATCH_SIZE=1
RECENT_SIZE=32

TAU_INIT=1.0
TAU_DELTA=0.01

# Source model & task configs
source config_models.sh
source config_tasks.sh

# Fixed settings
BENCHMARK="synthetic"
MAX_SEQ_LENGTH=65536
BUDGETS=(128 384 1024 4096)
MODELS=("llama-3.1-8b" "qwen-2.5-7b-1m")
TASKS=("niah_single_3" "vt" "fwe" "qa_1" "cwe")

TEMPERATURE="0.0"
TOP_P="1.0"
TOP_K="32"

total_time=0

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} "." ".")
    IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE \
         OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"

    if [ -z "${MODEL_PATH}" ]; then
        echo "Model: ${MODEL_NAME} is not supported"
        exit 1
    fi

    for BUDGET in "${BUDGETS[@]}"; do
        KEY_SIZE=$((BUDGET - RECENT_SIZE))

        RESULTS_DIR="${ROOT_DIR}/budget${BUDGET}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
        DATA_DIR="${SCRIPT_DIR}/../../../../../benchmarks/ruler/benchmark_root/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}/data"
        PRED_DIR="${RESULTS_DIR}"
        mkdir -p ${PRED_DIR}

        for TASK in "${TASKS[@]}"; do
            echo "  Model:  ${MODEL_NAME}"
            echo "  Task:   ${TASK}"
            echo "  Budget: ${BUDGET} (Key=${KEY_SIZE}, Recent=${RECENT_SIZE})"
            echo "  SeqLen: ${MAX_SEQ_LENGTH}"

            start_time=$(date +%s)
            python pred/call_api.py \
                --data_dir ${DATA_DIR} \
                --save_dir ${PRED_DIR} \
                --benchmark ${BENCHMARK} \
                --task ${TASK} \
                --server_type ${MODEL_FRAMEWORK} \
                --model_name_or_path ${MODEL_PATH} \
                --temperature ${TEMPERATURE} \
                --top_k ${TOP_K} \
                --top_p ${TOP_P} \
                --batch_size ${BATCH_SIZE} \
                --key_size ${KEY_SIZE} \
                --recent_size ${RECENT_SIZE} \
                --tau_init ${TAU_INIT} \
                --tau_delta ${TAU_DELTA} \
                --enable_keyformer \
                ${STOP_WORDS}

            if [ $? -ne 0 ]; then
                echo "FAIL: ${MODEL_NAME} / ${TASK} / budget=${BUDGET}"
            fi
            end_time=$(date +%s)
            time_diff=$((end_time - start_time))
            total_time=$((total_time + time_diff))
        done

        python eval/evaluate.py \
            --data_dir ${PRED_DIR} \
            --benchmark ${BENCHMARK}
    done
done

echo ""
echo "ALL DONE!"
echo "Total time spent: $total_time seconds"
