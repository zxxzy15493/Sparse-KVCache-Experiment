cd "$(dirname "$0")"
REPO_ROOT=$(cd ../../../.. && pwd)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ATTN_PATTERN="../../attn_patterns/deepseek-r1-distill-qwen-1.5b"
SPARSITY="0.5"

LOG_DIR="./log/DuoAttention/${MODEL}"
SAVE_DIR="./results/DuoAttention/${MODEL}"

mkdir -p ${LOG_DIR}
mkdir -p ${SAVE_DIR}

python -u pred_duo.py \
    --model ${MODEL} \
    --save_dir ${SAVE_DIR} \
    --attn_load_dir ${ATTN_PATTERN} \
    --sparsity ${SPARSITY} \
    --sink_size 128 \
    --recent_size 256 \
    --num_shots 8 \
    --cot_type gsm8k-cot > ${LOG_DIR}/gsm8k.log 2>&1

python -u evaluate.py \
    --input ${SAVE_DIR}/gsm8k.jsonl \
    --output ${SAVE_DIR}/gsm8k_eval.jsonl \
    --force

python -u ./tool/data_infos.py \
    --data-dir ${SAVE_DIR} \
    --model ${MODEL} \
    --task gsm8k
