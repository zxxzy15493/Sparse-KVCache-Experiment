if [ $# -ne 4 ]; then
    echo "Usage:  $0 <model> $1 <prefill_method> $2 <attn_type> $3 <budget>"
    exit 1
fi

MODEL_NAME=${1}
prefill_method=${2}
ATTN_TYPE=${3}
BUDGET=${4}

source ./config.sh

# Convert numeric flags to CLI arguments (handle both "0"/"1" and "" cases)
if [ "${RECALL}" = "1" ]; then
    RECALL_FLAG="--recall"
else
    RECALL_FLAG=""
fi
if [ "${MEASURE_TIME}" = "1" ]; then
    MEASURE_FLAG="--measure_time"
else
    MEASURE_FLAG=""
fi

PRED_DIR="./results/${MODEL_NAME}/${prefill_method}_${ATTN_TYPE}/${BUDGET}"
LOG_DIR="./log/${MODEL_NAME}/${prefill_method}_${ATTN_TYPE}/${BUDGET}"

mkdir -p ${PRED_DIR}
mkdir -p ${LOG_DIR}

python -u ./pred.py \
    --model ${MODEL_NAME} \
    --attn_type ${ATTN_TYPE} \
    --save_dir ${PRED_DIR} \
    --dtype ${DTYPE} \
    --device ${DEVICE} \
    --budget ${BUDGET} \
    --prefill_method ${prefill_method} \
    --budget_ratio ${BUDGET_RATIO} \
    --estimate_ratio ${ESTIMATE_RATIO} \
    --ratio_or_fixed ${RATIO_OR_FIXED} \
    --fixed_output_length ${FIXED_OUTPUT_LENGTH} \
    --cot \
    ${RECALL_FLAG} \
    ${MEASURE_FLAG} > "${LOG_DIR}/LongBenchV2.log" 2>&1


python -u ./result.py \
    --save_dir ${PRED_DIR}