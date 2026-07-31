#!/bin/bash
# Usage examples:
#   bash gsm8k_run.sh deepseek-r1-distill-qwen-1.5b
#   bash gsm8k_run.sh llama-3.1-8b
#   bash gsm8k_run.sh qwen-2.5-7b
#
# The model name corresponds to a key in config/model2path.json.

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model> [num_shots] [cot_type]"
    echo "  model: model name key in config/model2path.json"
    echo "  num_shots: number of few-shot examples (default: 8)"
    echo "  cot_type: chain-of-thought type (default: gsm8k-cot)"
    exit 1
fi

MODEL=${1}
NUM_SHOTS=${2:-8}
COT_TYPE=${3:-gsm8k-cot}

RESULT_DIR="./results/pred/${MODEL}/${COT_TYPE}"

source ./config.sh

LOG_DIR="./log/${MODEL}/${COT_TYPE}"
mkdir -p ${LOG_DIR}
echo "Start predict..."

python -u ./pred.py \
    --model_name ${MODEL} \
    --save_dir ${RESULT_DIR} \
    --dtype bf16 \
    --device auto \
    --cot_type ${COT_TYPE} \
    --num_shots ${NUM_SHOTS} \
    > ${LOG_DIR}/gsm8k_run.log 2>&1

echo "Prediction done. Running evaluation..."

python -u ./evaluate.py \
    --input ${RESULT_DIR}/gsm8k.jsonl \
    --output ${RESULT_DIR}/gsm8k_eval.jsonl \
    --force

python -u ./tool/data_infos.py \
    --data-dir ${RESULT_DIR} \
    --model ${MODEL} \
    --task gsm8k

echo "Done. Results saved to ${RESULT_DIR}"
