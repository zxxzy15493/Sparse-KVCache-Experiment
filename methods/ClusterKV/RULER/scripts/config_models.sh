# Model identifiers used by the ClusterKV RULER scripts.
TEMPERATURE="0.0"

MODEL_SELECT() {
    local model_name=$1

    MODEL_PATH=""
    MODEL_TEMPLATE_TYPE=""
    MODEL_FRAMEWORK=""
    TOKENIZER_PATH=""
    TOKENIZER_TYPE=""
    OPENAI_API_KEY=""
    GEMINI_API_KEY=""
    AZURE_ID=""
    AZURE_SECRET=""
    AZURE_ENDPOINT=""

    case "$model_name" in
        llama-3.1-8b)
            MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
            MODEL_TEMPLATE_TYPE="meta-llama3"
            MODEL_FRAMEWORK="clusterkv"
            ;;
        qwen-2.5-7b-1m)
            MODEL_PATH="Qwen/Qwen2.5-7B-Instruct-1M"
            MODEL_TEMPLATE_TYPE="qwen-chat"
            MODEL_FRAMEWORK="clusterkv"
            ;;
        glm-4-9b-1m)
            MODEL_PATH="zai-org/glm-4-9b-chat-1m"
            MODEL_TEMPLATE_TYPE="chatglm-chat"
            MODEL_FRAMEWORK="clusterkv"
            ;;
        *)
            return 1
            ;;
    esac

    TOKENIZER_PATH="$MODEL_PATH"
    TOKENIZER_TYPE="hf"

    echo "$MODEL_PATH:$MODEL_TEMPLATE_TYPE:$MODEL_FRAMEWORK:$TOKENIZER_PATH:$TOKENIZER_TYPE:$OPENAI_API_KEY:$GEMINI_API_KEY:$AZURE_ID:$AZURE_SECRET:$AZURE_ENDPOINT"
}
