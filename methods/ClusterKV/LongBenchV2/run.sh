#!/bin/bash
set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# =========================
# basic parameters
# =========================
SEED="4321"
DEVICE="0"

# attention method to use: cluster, quest, or original (no extra flag)
# mirrors the prior cluster_0509 invocation
METHOD="cluster"

# =========================
# data paths and model config
# =========================
# corresponds to --model in mypred.py's parse_common_args
MODEL_NAME="qwen2.5-7b-chat-32k"  
# force a model path (overrides the default in config/model2path.json)
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct-1M" 
DATA_FILE="../../../benchmarks/longbenchv2/filtered_longbench_v2_64k-192k.jsonl"

MAX_CONTEXT_LEN=234800
MAX_NEW_TOKENS=128

# output and log directories
SAVE_DIR="results"
LOG_DIR="logs"
mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

# =========================
# environment variables
# =========================
export CUDA_VISIBLE_DEVICES=${DEVICE}
export TOKENIZERS_PARALLELISM=false

# reduce GPU memory fragmentation (using the expandable_segments setting)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# adjust based on your actual project path
export PYTHONPATH="..${PYTHONPATH:+:${PYTHONPATH}}"

# =========================
# auto-restart parameters
# =========================
MAX_RESTARTS=100000
RESTART_SLEEP=10
RESTART_COUNT=0

# =========================
# dynamically build the Python command
# =========================
CMD="python mypred.py \
    --model \"${MODEL_NAME}\" \
    --model_path \"${MODEL_PATH}\" \
    --data_file \"${DATA_FILE}\" \
    --save_dir \"${SAVE_DIR}\" \
    --seed \"${SEED}\" \
    --max_context_len ${MAX_CONTEXT_LEN} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --cot \
    --token_budget 4096 \
    --nlist 1600 "

# add Patch arguments based on the chosen METHOD
if [ "$METHOD" = "cluster" ]; then
    CMD="$CMD --cluster"
elif [ "$METHOD" = "quest" ]; then
    # when using quest, append token_budget / chunk_size as needed
    CMD="$CMD --quest --token_budget 4096"
fi

# =========================
# main loop: auto-restart on abnormal exit
# =========================
while true; do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${LOG_DIR}/run_${MODEL_NAME}_${METHOD}_${TIMESTAMP}.log"

    echo "========================================"
    echo "Start running at ${TIMESTAMP}"
    echo "Restart count: ${RESTART_COUNT}"
    echo "Log file: ${LOG_FILE}"
    echo "Command: ${CMD}"
    echo "========================================"

    # execute the command and tee to the log file
    eval $CMD 2>&1 | tee "${LOG_FILE}"

    EXIT_CODE=${PIPESTATUS[0]}

    echo "Python exit code: ${EXIT_CODE}"

    if [ "${EXIT_CODE}" -eq 0 ]; then
        echo "Job finished successfully."
        break
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))

    if [ "${RESTART_COUNT}" -ge "${MAX_RESTARTS}" ]; then
        echo "Reached max restart count: ${MAX_RESTARTS}"
        exit 1
    fi

    echo "Job crashed or was killed. Restarting after ${RESTART_SLEEP} seconds..."
    sleep "${RESTART_SLEEP}"
done
