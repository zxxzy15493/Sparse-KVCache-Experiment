if [ $# -ne 6 ]; then
    echo "Usage: $0 <model> $1 {prefill_method} $2 <attn_type> $3 <budget> $4 <Benchmark> $5 <task>"
    exit 1
fi

MODEL=${1}
PREFILL_METHOD=${2}
ATTN_TYPE=${3}
BUDGET=${4}
BENCHMARK=${5}
task=${6}
FIXED_OUTPUT_LENGTH=0
BUDGET_RATIO=0.018
ESTIMATE_RATIO=0.23
RATIO_OR_FIXED=1
RECALL=1
MEASURE_TIME=0

# -1 
NUM_EXAMPLES=-1

RESULT_DIR="./results/RECALLOverview/${MODEL}/${PREFILL_METHOD}_${ATTN_TYPE}/${BENCHMARK}/${BUDGET}"
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
LOG_DIR="./log/RECALLOverview/${MODEL}/${PREFILL_METHOD}_${ATTN_TYPE}/${BENCHMARK}/${BUDGET}"
mkdir -p ${LOG_DIR}
echo "Parameters: ${MODEL} ${ATTN_TYPE} ${BUDGET_RATIO} ${ESTIMATE_RATIO} ${BUDGET} ${RATIO_OR_FIXED} ${RESULT_DIR}"
echo "Start to predict..."


python -u ./pred.py \
    --attn_type ${ATTN_TYPE} \
    --save_dir ${RESULT_DIR} \
    --model ${MODEL} \
    --dtype bf16 \
    --device auto \
    --task ${task} \
    --benchmark ${BENCHMARK} \
    --budget_ratio ${BUDGET_RATIO} \
    --estimate_ratio ${ESTIMATE_RATIO} \
    --budget ${BUDGET} \
    --ratio_or_fixed ${RATIO_OR_FIXED} \
    ${RECALL} \
    ${MEASURE_TIME} \
    --prefill_method ${PREFILL_METHOD} \
    --fixed_output_length ${FIXED_OUTPUT_LENGTH} \
    >> ${LOG_DIR}/${task}.log 2>&1

python -u ./evaluate.py \
    --data_dir ${RESULT_DIR} \
    --task ${task} \
    --budget ${BUDGET} 