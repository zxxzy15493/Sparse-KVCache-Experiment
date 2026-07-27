


MODEL_NAME="Qwen/Qwen2.5-7B-Instruct-1M"
SAVE_DIR="results1"
CACHE_SIZE=4096
WINDOW_SIZE=32
GAMMA=200

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

mkdir -p $SAVE_DIR


python "${SCRIPT_DIR}/pred.py" \
    --model "$MODEL_NAME" \
    --save_dir "$SAVE_DIR" \
    --compress \
    --cascading \
    --cache_size "$CACHE_SIZE" \
    --window_size "$WINDOW_SIZE" \
    --gamma "$GAMMA" \
    --cot

