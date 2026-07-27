
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct-1M"
SAVE_DIR="results"
COMPRESS_CONFIG="config/ablation_c4096_w32_k7_maxpool.json"

mkdir -p $SAVE_DIR
mkdir -p $LOG_DIR


python pred.py \
    --model "$MODEL_NAME" \
    --save_dir "$SAVE_DIR" \
    --compress_args_path "$COMPRESS_CONFIG" \
    --cot \

