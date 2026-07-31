MODEL_NAME=${1}
BENCHMARK=${2}
TASK=${3}
NUM_EXAMPLES=${4:--1}
MAX_LEN=${5:-65536}
K=${6:-9}
L=${7:-150}

DATA_ROOT="../../../../benchmarks"

if [ "$BENCHMARK" = "LongBench" ]; then
    DATA_DIR="${DATA_ROOT}/longbench"
elif [ "$BENCHMARK" = "Synthetic" ]; then
    case "$MODEL_NAME" in
        llama3.1*|llama-3.1*) MODEL_SHORT="llama-3.1-8b" ;;
        qwen2.5*|qwen-2.5*)   MODEL_SHORT="qwen-2.5-7b-1m" ;;
        glm*)                  MODEL_SHORT="glm-4-9b" ;;
        *)                     MODEL_SHORT="$MODEL_NAME" ;;
    esac
    DATA_DIR="${DATA_ROOT}/ruler/benchmark_root/${MODEL_SHORT}/synthetic/${MAX_LEN}/data/${TASK}"
else
    echo "Unknown benchmark: $BENCHMARK"
    exit 1
fi

KLS="${K}_${L}"
if [ "$BENCHMARK" = "Synthetic" ]; then
    RESULT_DIR="./results/pred/${MODEL_NAME}/${BENCHMARK}/${MAX_LEN}/${KLS}"
    LOG_DIR="./log/pred/${MODEL_NAME}/${BENCHMARK}/${MAX_LEN}/${KLS}"
else
    RESULT_DIR="./results/pred/${MODEL_NAME}/${BENCHMARK}/${KLS}"
    LOG_DIR="./log/pred/${MODEL_NAME}/${BENCHMARK}/${KLS}"
fi

mkdir -p ${RESULT_DIR}
mkdir -p ${LOG_DIR}

MODEL_PATH=$(python -c "import json; print(json.load(open('config/model2path.json'))['${MODEL_NAME}'])")

echo "Parameters: MODEL_PATH=${MODEL_PATH}, BENCHMARK=${BENCHMARK}, TASK=${TASK}, NUM_EXAMPLES=${NUM_EXAMPLES}, MAX_LEN=${MAX_LEN}, K=${K}, L=${L}"
echo "DATA_DIR: ${DATA_DIR}"
echo "Start to predict..."

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
OMP_NUM_THREADS=${OMP_NUM_THREADS:-96} \
torchrun --nproc_per_node=${NPROC_PER_NODE:-1} ./pred.py \
    --model_name ${MODEL_PATH} \
    --model_key ${MODEL_NAME} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    --save_dir ${RESULT_DIR} \
    --dtype bf16 \
    --num_samples ${NUM_EXAMPLES} \
    --data_dir ${DATA_DIR} \
    --max_seq_length ${MAX_LEN} \
    --K ${K} \
    --L ${L} \
    >${LOG_DIR}/${TASK}.log 2>&1

echo "Start to evaluate..."
if [ "$BENCHMARK" = "LongBench" ]; then
    python -u eval.py \
        --model ${MODEL_NAME} \
        --benchmark ${BENCHMARK} \
        --task ${TASK} \
        --save_dir ${RESULT_DIR} \
        --data_dir ${DATA_DIR}
    echo "Results:"
    cat "${RESULT_DIR}/result.json"
elif [ "$BENCHMARK" = "Synthetic" ]; then
    python -u evaluate.py \
        --data-dir ${RESULT_DIR} \
        --tasks ${TASK}
    echo "Results:"
    cat "${RESULT_DIR}/summary.csv"
fi
