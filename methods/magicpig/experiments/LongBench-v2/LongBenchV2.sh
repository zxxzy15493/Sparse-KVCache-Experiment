if [ $# -ne 4 ];  then
    echo "Usage: $0 <model_name> $1 <K> $2 <L> $3 <expect_budget>"
    exit 1
fi

MODEL_NAME=${1}
K=${2}
L=${3}
expect_budget=${4}

source ./config.sh

PRED_DIR="./results/${MODEL_NAME}/${expect_budget}/${KLS}/"
LOG_DIR="./log/${MODEL_NAME}/${expect_budget}/${KLS}"

mkdir -p ${PRED_DIR}
mkdir -p ${LOG_DIR}

CUDA_VISIBLE_DEVICES=0\
    OMP_NUM_THREADS=96 \
    torchrun --nproc_per_node=1 ./pred.py \
    --save_dir ${PRED_DIR} \
    --model ${MODEL_NAME} \
    --K ${K} \
    --L ${L} \
    --fixed_budget 0\
    --recall ${RECALL} \
    --fixed_output_length ${FIXED_OUTPUT_LENGTH}\
    --measure_time ${MEASURE_TIME}\
    --cot\
    ${STOP_WORDS} >> ${LOG_DIR}/LongBenchV2.log 2>&1

python -u ./result.py \
    --save_dir ${PRED_DIR}
