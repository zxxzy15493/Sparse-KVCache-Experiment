
MODEL_NAME=${1}
BENCHMARK=${2}
TASK=${3}
NUM_EXAMPLES=${4:--1}
MAX_LEN=${5:-65536}

DATA_ROOT="../../../../benchmarks"

if [ "$BENCHMARK" = "LongBench" ]; then
    DATA_DIR="${DATA_ROOT}/LongBench"
elif [ "$BENCHMARK" = "Synthetic" ]; then
    case "$MODEL_NAME" in
        llama3.1*) MODEL_SHORT="llama3.1-8b" ;;
        qwen2.5*)  MODEL_SHORT="qwen2.5-7b" ;;
        glm*)      MODEL_SHORT="glm-4-9b" ;;
        *)         MODEL_SHORT="$MODEL_NAME" ;;
    esac
    DATA_DIR="${DATA_ROOT}/benchmark_root/${MODEL_SHORT}/synthetic/${MAX_LEN}/data/${TASK}"
else
    echo "Unknown benchmark: $BENCHMARK"
    exit 1
fi

RESULT_DIR="./results/pred/${MODEL_NAME}/${BENCHMARK}"
LOG_DIR="./log/pred/${MODEL_NAME}/${BENCHMARK}"

mkdir -p ${RESULT_DIR}
mkdir -p ${LOG_DIR}

MODEL_PATH=$(python -c "import json; print(json.load(open('config/model2path.json'))['${MODEL_NAME}'])")

echo "Parameters: ${MODEL_PATH} ${BENCHMARK} ${TASK} ${NUM_EXAMPLES} ${MAX_LEN}"
echo "DATA_DIR: ${DATA_DIR}"
echo "Start to predict..."

python -u pred.py \
    --model_name ${MODEL_PATH} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    --save_dir ${RESULT_DIR} \
    --dtype bf16 \
    --device auto \
    --num_samples ${NUM_EXAMPLES} \
    --data_dir ${DATA_DIR} \
    >${LOG_DIR}/${TASK}.log 2>&1

echo "Start to evaluate..."
python -u eval.py \
    --model ${MODEL_NAME} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    --save_dir ${RESULT_DIR} \
    --data_dir ${DATA_DIR}

echo "Results:"
cat "${RESULT_DIR}/result.json"
