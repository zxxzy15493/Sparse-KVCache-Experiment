
# cd to algorithm root
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.."

GPUS="1" # GPU size for tensor_parallel.
ROOT_DIR="benchmark_root" # the path that stores generated task samples and model predictions.
MODEL_DIR="../../../../../../benchmarks/models" # the path that contains individual model folders from HUggingface.
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.
BATCH_SIZE=1  # increase to improve GPU utilization

cd ./experiments/recall/ruler/scripts

source config_models.sh

MODEL_NAME="llama-3.1-8b"

MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"


BENCHMARK="synthetic"

SEQ_LENGTHS1=(
    65536
)

NAME="ann"
SCORE="sparse_q"
REALLOCATE_TO_MEAN_VALUE=True
# Start client (prepare data / call model API / obtain final metrics)
total_time=0

TASKS="niah_single_3 vt fwe"


ROOT_DIR="../../../../../../benchmarks/Ruler_recall"

RES_DIR="../../../../output_budget/ruler/"

for MAX_SEQ_LENGTH in 65536; do
    data_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    pred_DIR="${RES_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"

    DATA_DIR="${data_DIR}/data"
    PRED_DIR="${pred_DIR}/pred"

    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}
    
    for TASK in $TASKS; do
        for K in 128 384 1024 4096; do
            PRED_DIR_final="${PRED_DIR}/${K}"
            save_path="../../../../efficiency/recall_attnscores/ruler/llama/${K}"
            for LOCAL_K in 32; do
                for RANK in 16; do
                    start_time=$(date +%s)
                    python pred/sparq_call_api.py \
                        --data_dir ${DATA_DIR} \
                        --save_dir ${PRED_DIR_final} \
                        --benchmark ${BENCHMARK} \
                        --task ${TASK} \
                        --server_type ${MODEL_FRAMEWORK} \
                        --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
                        --temperature ${TEMPERATURE} \
                        --top_k ${TOP_K} \
                        --top_p ${TOP_P} \
                        --batch_size ${BATCH_SIZE} \
                        --name ${NAME} \
                        --k ${K} \
                        --local_k ${LOCAL_K} \
                        --score ${SCORE} \
                        --rank ${RANK} \
                        --recall_save_path ${save_path} \
                        --reallocate_to_mean_value ${REALLOCATE_TO_MEAN_VALUE} \
                        ${STOP_WORDS}
                    end_time=$(date +%s)
                    time_diff=$((end_time - start_time))
                    total_time=$((total_time + time_diff))
                done
            done
        done
    done
done

echo "Total time spent on call_api: $total_time seconds"
