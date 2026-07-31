cd "$(dirname "$0")"
METHODS_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="${METHODS_DIR}:${PYTHONPATH:-}"


MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LOG_DIR="./log/CakeKV/${MODEL}"
SAVE_DIR="./results/CakeKV/${MODEL}"

mkdir -p ${LOG_DIR}
mkdir -p ${SAVE_DIR}

python -u pred_cake.py \
    --model ${MODEL} \
    --save_dir ${SAVE_DIR} \
    --cache_size 360 \
    --window_size 32 \
    --gamma 200.0 \
    --compress \
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
