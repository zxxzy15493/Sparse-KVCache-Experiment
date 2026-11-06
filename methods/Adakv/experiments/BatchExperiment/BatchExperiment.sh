#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: bash BatchExperiment.sh MODEL BENCHMARK TASK [BATCH_SIZE] [CONTEXT_LENGTH] [NUM_EXAMPLES]

MODEL: alias in config/model2path.json, HF model ID, or local model path
BENCHMARK: LongBench or RULER (case-insensitive; Synthetic also accepted)
BATCH_SIZE: defaults to 1
CONTEXT_LENGTH: RULER dataset length, defaults to 65536; use 0 for LongBench
NUM_EXAMPLES: defaults to -1 (all)
Environment (common): PYTHON=python, DEVICE=auto, DTYPE=bf16, DATA_DIR=<optional local task data>
                      RESULTS_ROOT=<optional results directory>, LOG_ROOT=<optional log directory>
Environment (HeadKV): METHOD=AdativeKV, HEAD_CHOICE=random, MAX_CAPACITY_PROMPTS=512,
                      BETA=1.5, TEMP=1.0, WINDOW_SIZE=32

Examples:
  bash BatchExperiment.sh qwen LongBench narrativeqa 4 0 10
  bash BatchExperiment.sh qwen RULER niah_single_1 4 4096 10
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if (( $# < 3 || $# > 6 )); then
    usage >&2
    exit 1
fi

MODEL_NAME=$1
BENCHMARK=$2
TASK=$3
BATCH_SIZE=${4:-1}
CONTEXT_LENGTH=${5:-0}
NUM_EXAMPLES=${6:--1}
PYTHON_BIN=${PYTHON:-python}
DEVICE=${DEVICE:-auto}
DTYPE=${DTYPE:-bf16}

METHOD=${METHOD:-AdativeKV}
HEAD_CHOICE=${HEAD_CHOICE:-random}
MAX_CAPACITY_PROMPTS=${MAX_CAPACITY_PROMPTS:-512}
BETA=${BETA:-1.5}
TEMP=${TEMP:-1.0}
WINDOW_SIZE=${WINDOW_SIZE:-32}

if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE must be a positive integer." >&2
    exit 1
fi
if [[ "$NUM_EXAMPLES" != "-1" && ! "$NUM_EXAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_EXAMPLES must be -1 or a positive integer." >&2
    exit 1
fi
if [[ ! "$TASK" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "TASK must be a task name without path separators." >&2
    exit 1
fi

EXTRA_ARGS=()
case "${BENCHMARK,,}" in
    longbench)
        BENCHMARK=LongBench
        if [[ "$CONTEXT_LENGTH" != "0" ]]; then
            echo "LongBench does not use context_length; set positional argument 5 to 0." >&2
            exit 1
        fi
        LENGTH_DIR=""
        ;;
    ruler|synthetic)
        BENCHMARK=RULER
        if [[ "$CONTEXT_LENGTH" == "0" ]]; then CONTEXT_LENGTH=65536; fi
        if [[ ! "$CONTEXT_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
            echo "CONTEXT_LENGTH must be a positive integer for RULER." >&2
            exit 1
        fi
        EXTRA_ARGS+=(--context_length "$CONTEXT_LENGTH")
        LENGTH_DIR="/${CONTEXT_LENGTH}"
        ;;
    *) echo "Unknown benchmark: $BENCHMARK" >&2; exit 1 ;;
esac
if [[ -n "${DATA_DIR:-}" ]]; then EXTRA_ARGS+=(--data_dir "$DATA_DIR"); fi

# One result namespace per batch size/context length; repeated invocations resume.
MODEL_TAG="${MODEL_NAME//\//--}"
RUN_DIR="${MODEL_TAG}/${BENCHMARK}${LENGTH_DIR}/batch-${BATCH_SIZE}"
RESULT_DIR="${RESULTS_ROOT:-${SCRIPT_DIR}/results/pred}/${RUN_DIR}"
LOG_DIR="${LOG_ROOT:-${SCRIPT_DIR}/log/pred}/${RUN_DIR}"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

HF_CACHE_ROOT="${SCRIPT_DIR}/.cache/huggingface/${BENCHMARK}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_CACHE_ROOT}/modules}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"

echo "Parameters: model=$MODEL_NAME benchmark=$BENCHMARK task=$TASK batch_size=$BATCH_SIZE context_length=$CONTEXT_LENGTH num_examples=$NUM_EXAMPLES"
echo "HeadKV: method=$METHOD head_choice=$HEAD_CHOICE max_capacity_prompts=$MAX_CAPACITY_PROMPTS beta=$BETA temp=$TEMP window_size=$WINDOW_SIZE"
echo "Prediction log: ${LOG_DIR}/${TASK}.log"
"$PYTHON_BIN" -u "${SCRIPT_DIR}/pred.py" \
    --model "$MODEL_NAME" --benchmark "$BENCHMARK" --task "$TASK" \
    --batch_size "$BATCH_SIZE" --num_samples "$NUM_EXAMPLES" \
    --save_dir "$RESULT_DIR" --dtype "$DTYPE" --device "$DEVICE" \
    --method "$METHOD" --head_choice "$HEAD_CHOICE" \
    --max_capacity_prompts "$MAX_CAPACITY_PROMPTS" --beta "$BETA" --temp "$TEMP" \
    --window_size "$WINDOW_SIZE" "${EXTRA_ARGS[@]}" \
    > "${LOG_DIR}/${TASK}.log" 2>&1

echo "Start to evaluate..."
if [[ "$BENCHMARK" == "LongBench" ]]; then
    "$PYTHON_BIN" -u "${SCRIPT_DIR}/eval.py" \
        --model "$MODEL_NAME" --benchmark LongBench --task "$TASK" --save_dir "$RESULT_DIR"
    cat "${RESULT_DIR}/result.json"
else
    "$PYTHON_BIN" -u "${SCRIPT_DIR}/evaluate.py" --data-dir "$RESULT_DIR"
    cat "${RESULT_DIR}/summary.csv"
fi
echo "Predictions: ${RESULT_DIR}/${TASK}.jsonl"
echo "Batch statistics: ${RESULT_DIR}/${TASK}.batch_stats.jsonl"
