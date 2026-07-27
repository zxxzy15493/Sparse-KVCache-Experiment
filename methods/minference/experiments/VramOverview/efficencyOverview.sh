


MODEL_NAME=${1}
warmup=${2}
input_max_token=${3}
fixed_output_length=${4}

RESULTS_DIR="./results/EfficencyOverview/${MODEL_NAME}/${input_max_token}_${fixed_output_length}"
LOG_DIR="./log/EfficencyOverview/${MODEL_NAME}/"


# mkdir -p ${RESULTS_DIR}
mkdir -p ${LOG_DIR}

case $MODEL_NAME in
    llama3.1-8b-instruct)
        MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
        MODEL_TEMPLATE_TYPE="meta-chat-3"
        ;;
    qwen2.5-7b-instruct)
        MODEL_PATH="Qwen/Qwen2.5-7B-Instruct-1M"
        MODEL_TEMPLATE_TYPE="qwen2.5"
        ;;
    glm-4-9b-chat-1m)
        MODEL_PATH="THUDM/glm-4-9b-chat-1m"
        MODEL_TEMPLATE_TYPE="glm-4"
        ;;
esac


python ./pred.py \
    --save_dir ${RESULTS_DIR} \
    --model_name ${MODEL_PATH} \
    --warmup ${warmup} \
    --input_max_token ${input_max_token} \
    --fixed_output_length ${fixed_output_length} \
    >${LOG_DIR}/EfficencyOverview_${input_max_token}_${fixed_output_length}.log 2>&1
