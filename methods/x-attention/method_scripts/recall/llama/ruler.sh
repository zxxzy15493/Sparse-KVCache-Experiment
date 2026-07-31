# cd to algorithm root
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
BLOCK_SPARSE_BUILD="$(find ./Block-Sparse-Attention/build -maxdepth 1 -type d -name 'lib.*' -print -quit 2>/dev/null || true)"
export PYTHONPATH=".:./Block-Sparse-Attention:${BLOCK_SPARSE_BUILD:-}:${PYTHONPATH:-}"

# Root Directories

GPUS="1" # GPU size for tensor_parallel.
ROOT_DIR="benchmark_root" # the path that stores generated task samples and model predictions.
MODEL_DIR="../../benchmarks/models" # the path that contains individual model folders from Huggingface.
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.
BATCH_SIZE=1  # increase to improve GPU utilization

# Model and Tokenizer
source ./eval/recall/RULER/scripts/config_models.sh
MODEL_NAME="llama-3.1-8b"


MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"

echo "TOKENIZER_TYPE: ${TOKENIZER_TYPE}"

# Benchmark and Tasks
source ./eval/recall/RULER/scripts/config_tasks.sh
BENCHMARK="synthetic"
METRIC="xattn" # Default: xattn
STRIDE=16

# Parse additional arguments with defaults
PRINT_DETAIL=${PRINT_DETAIL:-""}

THRESHOLD=${THRESHOLD:-""}


MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

TASKS="niah_single_3 vt fwe"
total_time=0


ROOT_DIR="../../benchmarks/Ruler_recall"

RES_DIR="./efficiency"


total_time=0
# for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
for MAX_SEQ_LENGTH in 65536; do
    for p in 0.8 0.85 0.9 0.95; do
        data_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
        DATA_DIR="${data_DIR}/data"
        # mkdir -p ${DATA_DIR}
        
        for TASK in ${TASKS}; do
            start_time=$(date +%s)
            save_path="./efficiency/attn_score/ruler/llama/${p}"
            python ./eval/recall/RULER/scripts/budget_pred/call_api.py \
                --data_dir ${DATA_DIR} \
                --save_dir tmp/xattn_unused_pred \
                --benchmark ${BENCHMARK} \
                --task ${TASK} \
                --server_type ${MODEL_FRAMEWORK} \
                --model_name_or_path ${MODEL_PATH} \
                --temperature ${TEMPERATURE} \
                --top_k ${TOP_K} \
                --p ${p} \
                --save_path ${save_path} \
                --top_p ${TOP_P} \
                --batch_size ${BATCH_SIZE} \
                ${STOP_WORDS} \
                --metric ${METRIC} \
                ${THRESHOLD} \
                --stride ${STRIDE} \
                ${PRINT_DETAIL}
            end_time=$(date +%s)
            time_diff=$((end_time - start_time))
            total_time=$((total_time + time_diff))
        done
    done
done

echo "Total time spent on call_api: $total_time seconds"
