
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LOG_DIR="./log/Full/${MODEL}"
SAVE_DIR="./results/Full/${MODEL}"

mkdir -p ${LOG_DIR}
mkdir -p ${SAVE_DIR}

# python -u pred_full.py \
#     --save_dir ${SAVE_DIR} \
#     --num_shots 8\
#     --cot_type gsm8k-cot \
#     --model ${MODEL} > ${LOG_DIR}/gsm8k.log 2>&1


python -u evaluate.py \
    --input ${SAVE_DIR}/gsm8k.jsonl \
    --output ${SAVE_DIR}/gsm8k_eval.jsonl \
    --force

python -u ./tool/data_infos.py \
    --data-dir ${SAVE_DIR} \
    --model ${MODEL} \
    --task gsm8k 