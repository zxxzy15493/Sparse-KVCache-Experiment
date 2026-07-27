

# bash run.sh
    # "meta-llama/Llama-3.1-8B-Instruct"
	# "Qwen/Qwen2.5-7B-Instruct-1M"

METHOD="pyramidkv"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct-1M"
SAVE_DIR="results1"
MAX_CAPACITY_PROMPTS=4096
WINDOW_SIZE=32
PYRAM_BETA=10
KERNEL_SIZE=7
POOLING="maxpool"
DEVICE=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

mkdir -p $SAVE_DIR


python "${SCRIPT_DIR}/pred.py" \
    --method "$METHOD" \
    --model "$MODEL_NAME" \
    --save_dir "$SAVE_DIR" \
    --device $DEVICE \
    --max_capacity_prompts "$MAX_CAPACITY_PROMPTS" \
    --window_size "$WINDOW_SIZE" \
    --pyram_beta "$PYRAM_BETA" \
    --kernel_size "$KERNEL_SIZE" \
    --pooling "$POOLING" \
    --cot

