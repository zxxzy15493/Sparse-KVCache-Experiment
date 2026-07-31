if [ $# -ne 10 ]; then
    echo "Usage: $0 <model> $1 <attn_type> $2 <budget_ratio> $3 <estimate_ratio> $4 <budget> $5 <ratio_or_fixed> $6{recall} $7{measure_time} $8{fixed_output_length} $9{prefill_method}"
    exit 1
fi

MODEL=${1}
ATTN_TYPE=${2}

BUDGET_RATIO=${3}
ESTIMATE_RATIO=${4}
BUDGET=${5}
RATIO_OR_FIXED=${6}
RECALL=${7}
MEASURE_TIME=${8}
FIXED_OUTPUT_LENGTH=${9}
PREFILL_METHOD=${10}
# -1 
NUM_EXAMPLES=1

RESULT_DIR="./results/efficencyOverview/${MODEL}/${ATTN_TYPE}/${PREFILL_METHOD}"

#  fix  mode
if [ "$RATIO_OR_FIXED" -eq 1 ]; then
    RESULT_DIR="${RESULT_DIR}/Fixed/${BUDGET}"
    MODE="Fixed"
elif [ "$RATIO_OR_FIXED" -eq 0 ]; then
    RESULT_DIR="${RESULT_DIR}/Ratio/${BUDGET_RATIO}_${ESTIMATE_RATIO}"
    MODE="Ratio"
else
    # ， mode 
    echo
    echo $RATIO_OR_FIXED"
fi

mkdir -p ${RESULT_DIR}

if [ "$RECALL" -eq 1 ]; then
    RECALL="--recall"
else
    RECALL=""
fi
if [ "$MEASURE_TIME" -eq 1 ]; then
    MEASURE_TIME="--measure_time"
else
    MEASURE_TIME=""
fi
LOG_DIR="./log/efficencyOverview/${MODEL}/${PREFILL_METHOD}_${ATTN_TYPE}/EfficiencyOverview/"
mkdir -p ${LOG_DIR}
echo "Parameters: ${MODEL} ${ATTN_TYPE} ${BUDGET_RATIO} ${ESTIMATE_RATIO} ${BUDGET} ${RATIO_OR_FIXED} ${RESULT_DIR}"
echo "Start to predict..."

INPUT_MAX_TOKENS=(
    4096
    # 8192 # 8k
    # 16384 #16k
    # 32768 # 32k
    65536 # 64k
    # 131072 # 128k
)
for INPUT_MAX_TOKEN in "${INPUT_MAX_TOKENS[@]}"; do
    CUDA_VISIBLE_DEVICES=0 \
    python -u ./pred.py \
        --save_dir ${RESULT_DIR} \
        --attn_type ${ATTN_TYPE} \
        --model ${MODEL} \
        --dtype bf16 \
        --device "cuda:0" \
        --benchmark "GSM8K" \
        --budget_ratio ${BUDGET_RATIO} \
        --estimate_ratio ${ESTIMATE_RATIO} \
        --budget ${BUDGET} \
        --input_max_token ${INPUT_MAX_TOKEN} \
        --ratio_or_fixed ${RATIO_OR_FIXED} \
        ${RECALL} \
        ${MEASURE_TIME} \
        --prefill_method ${PREFILL_METHOD} \
        --fixed_output_length ${FIXED_OUTPUT_LENGTH} \
        --num_samples ${NUM_EXAMPLES}  > ${LOG_DIR}${INPUT_MAX_TOKEN}_${FIXED_OUTPUT_LENGTH}_${BUDGET}.log 2>&1
done


