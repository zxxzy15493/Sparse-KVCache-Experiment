
GPUS="1" # GPU size for tensor_parallel.
MODEL_DIR=" " # the path that contains individual model folders from HUggingface.
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.
BATCH_SIZE=1  # increase to improve GPU utilization

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
METHODS_ROOT=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)
REPO_ROOT=$(cd -- "$METHODS_ROOT/.." && pwd)
QUEST_ROOT="$METHODS_ROOT/quest"

CHUNK_SIZE=16
TOKEN_BUDGET=2048
MODEL_NAME="qwen-2.5-7b-1m"

cd "$QUEST_ROOT/evaluation/RULER/scripts"
export PYTHONPATH="$QUEST_ROOT:${PYTHONPATH:-}"
# Model and Tokenizer
source config_models.sh

MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi
#source config_tasks.sh
BENCHMARK="synthetic"

#TASKS="niah_single_1 niah_multiquery vt fwe"
TASKS="vt fwe"
#TASKS="niah_single_1"

ROOT_DIR="$REPO_ROOT/benchmarks/ruler/benchmark_root"

RES_DIR="$QUEST_ROOT/output/ruler"
# Start client (prepare data / call model API / obtain final metrics)
total_time=0
for MAX_SEQ_LENGTH in 131072; do
    
    data_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    pred_DIR="${RES_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${data_DIR}/data"
    PRED_DIR="${pred_DIR}/pred"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}
    
    for TASK in $TASKS; do
        start_time=$(date +%s)
        python pred/quest_call_api.py \
            --data_dir ${DATA_DIR} \
            --save_dir ${PRED_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --server_type ${MODEL_FRAMEWORK} \
            --model_name_or_path ${MODEL_PATH} \
            --temperature ${TEMPERATURE} \
            --top_k ${TOP_K} \
            --top_p ${TOP_P} \
            --batch_size ${BATCH_SIZE} \
            --chunk_size ${CHUNK_SIZE} \
            --token_budget ${TOKEN_BUDGET} \
            ${STOP_WORDS}
        end_time=$(date +%s)
        time_diff=$((end_time - start_time))
        total_time=$((total_time + time_diff))
    done
    
    python eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${BENCHMARK}
done
echo "Total time spent on call_api: $total_time seconds"
