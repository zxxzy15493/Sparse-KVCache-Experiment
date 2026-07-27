if [ $# -ne 6 ]; then
    echo "Usage: $0 <model> $1{prefill_method}  $2 <attn_type> $3 <budget> $4 {input_max_token} $5{fixed_output_length}"
    exit 1
fi

MODEL=${1}
PREFILL_METHOD=${2}
ATTN_TYPE=${3}
BUDGET=${4}
INPUT_MAX_TOKEN=${5}
FIXED_OUTPUT_LENGTH=${6}
# -1 
RATIO_OR_FIXED=1
BUDGET_RATIO=0.018
ESTIMATE_RATIO=0.232
RECALL=0
MEASURE_TIME=0
NUM_EXAMPLES=-1

source config.sh

RESULT_DIR="./results/VRAMOverview/${MODEL}/${PREFILL_METHOD}_${ATTN_TYPE}/${INPUT_MAX_TOKEN}/${BUDGET}"

mkdir -p ${RESULT_DIR}

LOG_DIR="./log/VRAMOverview/${MODEL}/${PREFILL_METHOD}_${ATTN_TYPE}/${INPUT_MAX_TOKEN}/${BUDGET}"
mkdir -p ${LOG_DIR}
echo "Start to predict..."

python -u ./pred.py \
    --prefill_method ${PREFILL_METHOD} \
    --attn_type ${ATTN_TYPE} \
    --save_dir ${RESULT_DIR} \
    --model ${MODEL} \
    --dtype bf16 \
    --device auto \
    --budget_ratio ${BUDGET_RATIO} \
    --estimate_ratio ${ESTIMATE_RATIO} \
    --budget ${BUDGET} \
    --input_max_token ${INPUT_MAX_TOKEN} \
    --ratio_or_fixed ${RATIO_OR_FIXED} \
    ${RECALL} \
    ${MEASURE_TIME} \
    --fixed_output_length ${FIXED_OUTPUT_LENGTH} \
    --num_samples ${NUM_EXAMPLES}  > ${LOG_DIR}/${INPUT_MAX_TOKEN}_${FIXED_OUTPUT_LENGTH}_${BUDGET}.log 2>&1

