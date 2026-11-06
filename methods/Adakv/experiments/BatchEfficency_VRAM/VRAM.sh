#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: bash VRAM.sh MODEL BENCHMARK TASK [BATCH_SIZE] [CONTEXT_LENGTH] [NUM_EXAMPLES] [BUDGET] [METHOD] [HEAD_CHOICE] [BETA] [TEMP] [MAX_LEN] [FIXED_OUTPUT_LENGTH]

MODEL: alias in config/model2path.json, HF model ID, or local model path
BENCHMARK: LongBench or RULER (ruler/Synthetic also accepted)
TASK: dataset task name, e.g. niah_single_1 or narrativeqa
BATCH_SIZE: defaults to 1; the final partial batch is kept
CONTEXT_LENGTH: RULER data length, defaults to 65536; use 0 for LongBench
NUM_EXAMPLES: defaults to -1 (all)
BUDGET: HeadKV cache budget (max_capacity_prompts), defaults to 1024
METHOD: HeadKV method (AdativeKV/ReasonKV/SnapKV/PyramidKV/fullkv), defaults to AdativeKV
HEAD_CHOICE: random/copy/reason, defaults to random
BETA: HeadKV beta, defaults to 1.5
TEMP: HeadKV temp, defaults to 1.0
MAX_LEN: prompt limit; 0 selects the model default (LongBench) or context_length (RULER)
FIXED_OUTPUT_LENGTH: defaults to 2 for VRAM; 0 uses the task default and per-sample EOS
Environment: PYTHON=python, DEVICE=auto, DTYPE=bf16, DATA_DIR=<optional local task data>
             WARMUP=2 (warmup calls excluded from results)
             WINDOW_SIZE/KERNEL_SIZE/POOLING/FLOOR/SKIP/NORMALIZE (optional HeadKV overrides)

Examples:
  bash VRAM.sh llama-3.1-8b RULER niah_single_1 2 4096 10 1024 AdativeKV random 1.5 1.0 0 2
  bash VRAM.sh qwen-2.5-7b-1m RULER niah_single_1 2 4096 -1 1024 AdativeKV random 1.5 1.0 0 2
  bash VRAM.sh llama-3.1-8b LongBench narrativeqa 2 0 -1 1024 AdativeKV random 1.5 1.0 0 2
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if (( $# < 3 || $# > 13 )); then
    usage >&2
    exit 1
fi

MODEL_NAME=$1
BENCHMARK=$2
TASK=$3
BATCH_SIZE=${4:-1}
CONTEXT_LENGTH=${5:-0}
NUM_EXAMPLES=${6:--1}
BUDGET=${7:-1024}
METHOD=${8:-AdativeKV}
HEAD_CHOICE=${9:-random}
BETA=${10:-1.5}
TEMP=${11:-1.0}
MAX_LEN=${12:-0}
FIXED_OUTPUT_LENGTH=${13:-2}
WARMUP_COUNT=${WARMUP:-2}
PYTHON_BIN=${PYTHON:-python}
DEVICE=${DEVICE:-auto}
DTYPE=${DTYPE:-bf16}

if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE must be a positive integer." >&2
    exit 1
fi
if [[ "$NUM_EXAMPLES" != "-1" && ! "$NUM_EXAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_EXAMPLES must be -1 or a positive integer." >&2
    exit 1
fi
if [[ ! "$FIXED_OUTPUT_LENGTH" =~ ^(0|[1-9][0-9]*)$ || ! "$WARMUP_COUNT" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "FIXED_OUTPUT_LENGTH and WARMUP must be non-negative integers." >&2
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
if [[ "$MAX_LEN" != "0" ]]; then EXTRA_ARGS+=(--max_len "$MAX_LEN"); fi
if [[ -n "${DATA_DIR:-}" ]]; then EXTRA_ARGS+=(--data_dir "$DATA_DIR"); fi
if [[ -n "${WINDOW_SIZE:-}" ]]; then EXTRA_ARGS+=(--window_size "$WINDOW_SIZE"); fi
if [[ -n "${KERNEL_SIZE:-}" ]]; then EXTRA_ARGS+=(--kernel_size "$KERNEL_SIZE"); fi
if [[ -n "${POOLING:-}" ]]; then EXTRA_ARGS+=(--pooling "$POOLING"); fi
if [[ -n "${FLOOR:-}" ]]; then EXTRA_ARGS+=(--floor "$FLOOR"); fi
if [[ -n "${SKIP:-}" ]]; then EXTRA_ARGS+=(--skip "$SKIP"); fi
if [[ -n "${NORMALIZE:-}" ]]; then EXTRA_ARGS+=(--normalize "$NORMALIZE"); fi

MODEL_TAG="${MODEL_NAME//\//--}"
PARAMS="base-${BUDGET}-method-${METHOD}-head-${HEAD_CHOICE}-beta-${BETA}-temp-${TEMP}"
RUN_DIR="${MODEL_TAG}/${BENCHMARK}${LENGTH_DIR}/batch-${BATCH_SIZE}/${PARAMS}/output-${FIXED_OUTPUT_LENGTH}"
RESULT_DIR="${RESULTS_ROOT:-${SCRIPT_DIR}/results/vram}/${RUN_DIR}"
LOG_DIR="${LOG_ROOT:-${SCRIPT_DIR}/log/vram}/${RUN_DIR}"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

echo "Parameters: model=$MODEL_NAME benchmark=$BENCHMARK task=$TASK batch_size=$BATCH_SIZE context_length=$CONTEXT_LENGTH num_examples=$NUM_EXAMPLES budget=$BUDGET method=$METHOD head_choice=$HEAD_CHOICE"
echo "Prediction log: ${LOG_DIR}/${TASK}.log"
"$PYTHON_BIN" -u "${SCRIPT_DIR}/pred_vram.py" \
    --model "$MODEL_NAME" --benchmark "$BENCHMARK" --task "$TASK" \
    --batch_size "$BATCH_SIZE" --num_samples "$NUM_EXAMPLES" \
    --max_capacity_prompts "$BUDGET" --method "$METHOD" --head_choice "$HEAD_CHOICE" \
    --beta "$BETA" --temp "$TEMP" --fixed_output_length "$FIXED_OUTPUT_LENGTH" \
    --warmup "$WARMUP_COUNT" --save_dir "$RESULT_DIR" --dtype "$DTYPE" --device "$DEVICE" \
    "${EXTRA_ARGS[@]}" > "${LOG_DIR}/${TASK}.log" 2>&1

echo "Measurements: ${RESULT_DIR}/${TASK}.vram.jsonl"
echo "Predictions: ${RESULT_DIR}/${TASK}.vram.pred.jsonl"
echo "Summary:"
cat "${RESULT_DIR}/${TASK}.vram.summary.json"
