#!/bin/bash
set -x
set -euo pipefail

SCRIPT=${SCRIPT:-"./run.sh"}

# MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# MODEL_NAME="deepseek-llama-8b"
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_NAME="deepseek-qwen-1.5b"

BUDGET=${BUDGET:-360}
FIXTHRESHOLD=${FIXTHRESHOLD:-0.90}
THRESHOLD=${THRESHOLD:-100000}
SINK_SIZE=${SINK_SIZE:-16}
RECENT_SIZE=${RECENT_SIZE:-32}
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# export TOKENIZERS_PARALLELISM=false

COMPRESSORS=(
    # original
    # no_drop_lb
    # no_drop_lb_topp
    pq_search
)
export MAX_CPU_IN_USE=24

for COMPRESSOR in "${COMPRESSORS[@]}"; do
    LOG_DIR="log"
    mkdir -p "${LOG_DIR}"

    nohup bash "${SCRIPT}" \
        "${MODEL_PATH}" \
        "${MODEL_NAME}" \
        "${COMPRESSOR}" \
        "${BUDGET}" \
        "${FIXTHRESHOLD}" \
        "${THRESHOLD}" \
        "${SINK_SIZE}" \
        "${RECENT_SIZE}" \
        > "${LOG_DIR}/run_${MODEL_NAME}_${COMPRESSOR}_b${BUDGET}.nohup.log" 2>&1 &

    echo "Started ${COMPRESSOR}, PID=$!"
done

wait
echo "All finished."
