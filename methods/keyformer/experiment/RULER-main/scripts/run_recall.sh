
export PYTHONPATH=$PYTHONPATH:$(pwd)/../../
if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> $1 <benchmark_name>"
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR="recall_results"
MODEL_DIR="../.." # the path that contains individual model folders from HUggingface.

TAU_INIT=1.0
TAU_DELTA=0.01

source config_models.sh
MODEL_NAME=${1}
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi

export OPENAI_API_KEY=${OPENAI_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}

source config_tasks.sh
BENCHMARK=${2}
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

if [ "$MODEL_FRAMEWORK" == "vllm" ]; then
    python pred/serve_vllm.py \
        --model=${MODEL_PATH} \
        --tensor-parallel-size=${GPUS} \
        --dtype bfloat16 \
        --disable-custom-all-reduce \
        &

elif [ "$MODEL_FRAMEWORK" == "trtllm" ]; then
    python pred/serve_trt.py \
        --model_path=${MODEL_PATH} \
        &

elif [ "$MODEL_FRAMEWORK" == "sglang" ]; then
    python -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --tp ${GPUS} \
        --port 5000 \
        --enable-flashinfer \
        &
fi

total_time=0
RECENT_SIZE=32

for BUDGET in 128 384 1024 4096; do
    KEY_SIZE=$((BUDGET - RECENT_SIZE))
    
    echo "STARTING EVALUATION: BUDGET=${BUDGET} (Recent=${RECENT_SIZE}, Key=${KEY_SIZE})"

    for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
        RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}/budget${BUDGET}"
        DATA_DIR="${SCRIPT_DIR}/../../../../../benchmarks/Ruler_recall/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}/data"
        PRED_DIR="${RESULTS_DIR}"
        mkdir -p ${DATA_DIR}
        mkdir -p ${PRED_DIR}
        
        for TASK in "${TASKS[@]}"; do
            
            start_time=$(date +%s)
            python pred/call_api.py \
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
                --key_size ${KEY_SIZE} \
                --recent_size ${RECENT_SIZE} \
                --tau_init ${TAU_INIT} \
                --tau_delta ${TAU_DELTA} \
                --enable_keyformer \
                --check_recall \
                ${STOP_WORDS}
            end_time=$(date +%s)
            time_diff=$((end_time - start_time))
            total_time=$((total_time + time_diff))
        done
        
        python eval/evaluate.py \
            --data_dir ${PRED_DIR} \
            --benchmark ${BENCHMARK}
    done
done

echo "Total time spent on call_api: $total_time seconds"

