cd "$(dirname "$0")"
REPO_ROOT=$(cd ../../../.. && pwd)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
BUDGET=360

METHOD="PyramidKV"
SAVE_DIR="./results1/${METHOD}/${MODEL_NAME}_budget${BUDGET}"
LOG_DIR="./log/${METHOD}/${MODEL_NAME}_budget${BUDGET}"
mkdir -p ${LOG_DIR} ${SAVE_DIR}

echo "========================================="
echo "Running ${METHOD}"
echo "Model: ${MODEL_NAME}, Budget: ${BUDGET}"
echo "========================================="

python -u pred_pyramidkv.py \
    --model "${MODEL_NAME}" \
    --save_dir "${SAVE_DIR}" \
    --method "${METHOD}" \
    --max_capacity_prompts ${BUDGET} \
    --window_size 64 \
    --kernel_size 5 \
    --pooling avgpool \
    --num_shots 8 \
    --max_new_tokens 10000 2>&1 | tee "${LOG_DIR}/gsm8k.log"
python -u evaluate.py \
    --input ${SAVE_DIR}/gsm8k.jsonl \
    --output ${SAVE_DIR}/gsm8k_eval.jsonl \
    --force

python -u ./tool/data_infos.py \
    --data-dir ${SAVE_DIR} \
    --model ${MODEL_NAME} \
    --task gsm8k
echo ""
echo "Evaluation complete."
echo "Results: ${SAVE_DIR}/gsm8k.jsonl"
