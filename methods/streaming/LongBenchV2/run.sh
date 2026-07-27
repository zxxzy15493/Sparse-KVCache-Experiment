
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct-1M"
SAVE_DIR="results"

mkdir -p "$SAVE_DIR"

python pred.py \
    --model_name_or_path "$MODEL_NAME" \
    --save_dir "$SAVE_DIR" \
    --enable_streaming \
    --cot

