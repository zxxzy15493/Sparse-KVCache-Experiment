# bash run.sh


MODEL_NAME="Qwen/Qwen2.5-7B-Instruct-1M"
SAVE_DIR="results"
DEVICE=0
SINK_SIZE=64
RECENT_SIZE=256
SPARSITY=0.5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ATTN_LOAD_DIR="${PROJECT_ROOT}/attn_patterns/Qwen2.5-7B-Instruct"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

mkdir -p $SAVE_DIR

python "${SCRIPT_DIR}/pred.py" \
    --method "duo_attn" \
    --model "$MODEL_NAME" \
    --save_dir "$SAVE_DIR" \
    --device $DEVICE \
    --attn_load_dir "$ATTN_LOAD_DIR" \
    --sink_size "$SINK_SIZE" \
    --recent_size "$RECENT_SIZE" \
    --sparsity "$SPARSITY" \
    --cot
