



MODEL_NAME="Qwen/Qwen2.5-7B-Instruct-1M"
SAVE_DIR="results"
WINDOW_SIZE=32
DEVICE=0

# HeadKV-style parameters
MAX_CAPACITY_PROMPTS=4096
HEAD_CHOICE=1
BETA=1.5
TEMP=1.0
KERNEL_SIZE=7
SKIP=0
NORMALIZE=0
POOLING="maxpool"
FLOOR=0.2

# Parameters for calling HeadKV run_longbench
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct-1M"
ATTN_IMPL="flash_attention_2"
MAX_NUM_EXAMPLES=2000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

mkdir -p $SAVE_DIR

run_eval() {
    local method="$1"
    local head_choice="$2"
    local save_dir="${SAVE_DIR}/${method}_${head_choice}_base${MAX_CAPACITY_PROMPTS}_beta${BETA}_temp${TEMP}"

    echo "[RUN] method=${method}, head_choice=${head_choice}"
    python3 -u "${SCRIPT_DIR}/pred.py" \
        --model ${MODEL_PATH} \
        --save_dir ${save_dir} \
        --device ${DEVICE} \
        --window_size ${WINDOW_SIZE} \
        --max_capacity_prompts ${MAX_CAPACITY_PROMPTS} \
        --head_choice ${head_choice} \
        --beta ${BETA} \
        --temp ${TEMP} \
        --kernel_size ${KERNEL_SIZE} \
        --skip ${SKIP} \
        $( [ ${NORMALIZE} -eq 1 ] && echo "--normalize" || echo ) \
        --pooling ${POOLING} \
        --floor ${FLOOR} \
        --cot
}

# Run the two HeadKV method/head_choice combinations
run_eval "ReasonKV" "reason"



