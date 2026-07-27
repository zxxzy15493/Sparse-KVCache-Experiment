TEMPERATURE="0.0" # greedy
TOP_P="1.0"
TOP_K="32"

SEQ_LENGTHS=(
    65536
    32768
    16384
    8192
    4096
)

MODEL_SELECT() {
    MODEL_NAME=$1
    MODEL_DIR=$2
    ENGINE_DIR=$3
    
    case $MODEL_NAME in
        llama)
            MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
            MODEL_TEMPLATE_TYPE="meta-llama3"
            MODEL_FRAMEWORK="hf"
            ;;


        qwen-2.5-7b-1m)

            MODEL_PATH="Qwen/Qwen2.5-7B-Instruct-1M"


            # MODEL_TEMPLATE_TYPE="chatml"

            MODEL_TEMPLATE_TYPE="qwen-chat"


            MODEL_FRAMEWORK="hf"
            ;;

    esac


    if [ -z "${TOKENIZER_PATH}" ]; then
        if [ -f ${MODEL_PATH}/tokenizer.model ]; then
            TOKENIZER_PATH=${MODEL_PATH}/tokenizer.model
            TOKENIZER_TYPE="nemo"
        else
            TOKENIZER_PATH=${MODEL_PATH}
            TOKENIZER_TYPE="hf"
        fi
    fi


    echo "$MODEL_PATH:$MODEL_TEMPLATE_TYPE:$MODEL_FRAMEWORK:$TOKENIZER_PATH:$TOKENIZER_TYPE:$OPENAI_API_KEY:$GEMINI_API_KEY:$AZURE_ID:$AZURE_SECRET:$AZURE_ENDPOINT"
}
