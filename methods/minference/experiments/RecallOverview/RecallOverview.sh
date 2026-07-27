MODEL_NAME=${1}
BENCHMARK=${2}
TASK=${3}

RESULTS_DIR="./results/RecallOverview/${MODEL_NAME}/${BENCHMARK}/${TASK}"
LOG_DIR="./log/RecallOverview/${MODEL_NAME}/${BENCHMARK}"

mkdir -p ${RESULTS_DIR}
mkdir -p ${LOG_DIR}


python ./pred.py \
    --save_dir ${RESULTS_DIR} \
    --model_name ${MODEL_NAME} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    > ${LOG_DIR}/${TASK}.log 2>&1
