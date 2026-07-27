cd "$(dirname "$0")"
REPO_ROOT=$(cd ../../../.. && pwd)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
BUDGET=360

METHOD="ReasonKV"
HEAD_CHOICE="reason"
SAVE_DIR="./results/${METHOD}/${MODEL_NAME}_budget${BUDGET}"
LOG_DIR="./log/${METHOD}/${MODEL_NAME}_budget${BUDGET}"
mkdir -p ${LOG_DIR} ${SAVE_DIR}

echo "========================================="
echo "Running ${METHOD} (head_choice=${HEAD_CHOICE})"
echo "Model: ${MODEL_NAME}, Budget: ${BUDGET}"
echo "========================================="

python -u pred_headkv.py \
    --model "${MODEL_NAME}" \
    --save_dir "${SAVE_DIR}" \
    --method "${METHOD}" \
    --head_choice "${HEAD_CHOICE}" \
    --max_capacity_prompts ${BUDGET} \
    --beta 1.5 \
    --temp 1.0 \
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
