# !/bin/bash
# bash gsm8k_run.sh llama-3.1-8b RetroInfer Full_Flash_Attn 1024 0 gsm8k-simple
# bash gsm8k_run.sh qwen2.5-7b RetroInfer Full_Flash_Attn 1024 0 gsm8k-simple

# bash gsm8k_run.sh llama-3.1-8b Full_Flash_Attn minfer 1024 0 gsm8k-simple
# bash gsm8k_run.sh qwen2.5-7b Full_Flash_Attn minfer 1024 0 gsm8k-simple

if [ $# -ne 6 ]; then
    echo "Usage: $0 <model> $1 <attn_type> $2 <prefill_method> $3 <budget> $4 <NUM_EXAMPLES> $5 <COT_TYPE>"
    exit 1
fi

MODEL=${1}
ATTN_TYPE=${2}
prefill_method=${3}
BUDGET=${4}
NUM_EXAMPLES=${5}
COT_TYPE=${6}

BUDGET_RATIO=0.018
ESTIMATE_RATIO=0.232

RATIO_OR_FIXED=1
RECALL=0
MEASURE_TIME=0
FIXED_OUTPUT_LENGTH=0
# -1 


RESULT_DIR="./results/pred/${MODEL}/${prefill_method}_${ATTN_TYPE}/${COT_TYPE}/${BUDGET}"

source ./config.sh

LOG_DIR="./log/${MODEL}/${prefill_method}_${ATTN_TYPE}/${COT_TYPE}/${BUDGET}"
mkdir -p ${LOG_DIR}
echo "Start predict..."

# python -u ./evaluation_gsm8k.py \
#     --save_dir ${RESULT_DIR} \
#     --attn_type ${ATTN_TYPE} \
#     --model ${MODEL} \
#     --dtype bf16 \
#     --device auto \
#     --budget_ratio ${BUDGET_RATIO} \
#     --estimate_ratio ${ESTIMATE_RATIO} \
#     --budget ${BUDGET} \
#     --ratio_or_fixed ${RATIO_OR_FIXED} \
#     --cot_type ${COT_TYPE} \
#     ${RECALL} \
#     ${MEASURE_TIME} \
#     --prefill_method "${prefill_method}" \
#     --fixed_output_length ${FIXED_OUTPUT_LENGTH} \
#     --num_shots ${NUM_EXAMPLES}  > ${LOG_DIR}/gsm8k_run.log 2>&1

python -u ./evaluate.py \
    --input ${RESULT_DIR}/gsm8k.jsonl \
    --output ${RESULT_DIR}/gsm8k_eval.jsonl \
    --force

python -u ./tool/data_infos.py \
    --data-dir ${RESULT_DIR} \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --task gsm8k \


