#!/usr/bin/env bash
# One-click Qwen3-32B RULER grid. The shell owns method/length ordering and
# starts one Python worker per method/length; each worker loads one model and
# evaluates all 13 tasks before exiting.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNNER="$SCRIPT_DIR/qwen3_32b_all_methods.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B}"
DEVICE="${DEVICE:-cuda:0}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/benchmark_root/qwen3-32b/synthetic}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/benchmark_root_pred/synthetic/qwen3-32b-yarn2}"

LENGTHS=(4096 8192 16384 32768 65536)
TASKS=(
  niah_single_1 niah_single_2 niah_single_3
  niah_multikey_1 niah_multikey_2 niah_multikey_3
  niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2
)
# 每个方法只执行一次，保留当前实验顺序。
METHODS=(
  keyformer pqcache xattention headkv minference retroinfer
  # full_attention topk topp clusterkv
  # snapkv streaming quest sparq flexprefill adakv cakekv pyramidkv h2o
  # magicpig duo-attention
)

# Optional space-separated selectors allow a failed task/configuration to be
# resumed without traversing the whole grid. Defaults still run the full grid.
if [[ -n "${RULER_LENGTHS:-}" ]]; then
  read -r -a LENGTHS <<< "$RULER_LENGTHS"
fi
if [[ -n "${RULER_TASKS:-}" ]]; then
  read -r -a TASKS <<< "$RULER_TASKS"
fi
if [[ -n "${RULER_METHODS:-}" ]]; then
  read -r -a METHODS <<< "$RULER_METHODS"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1

echo "===== prepare RULER data: ${#LENGTHS[@]} lengths x ${#TASKS[@]} tasks x $NUM_SAMPLES samples ====="
if ! "$PYTHON_BIN" "$RUNNER" \
  --model "$MODEL_ID" \
  --lengths "${LENGTHS[@]}" \
  --tasks "${TASKS[@]}" \
  --num-samples "$NUM_SAMPLES" \
  --device "$DEVICE" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --prepare-only; then
  echo "RULER data preparation failed; stopping." >&2
  exit 1
fi

trap 'echo "Interrupted. Rerun this same .sh; completed samples will be skipped." >&2; exit 130' INT TERM

total=$((${#METHODS[@]} * ${#LENGTHS[@]}))
index=0
for method in "${METHODS[@]}"; do
  for length in "${LENGTHS[@]}"; do
    index=$((index + 1))
    echo "===== [$index/$total] run: $method / $length ====="
    if ! "$PYTHON_BIN" "$RUNNER" \
      --worker \
      --model "$MODEL_ID" \
      --methods "$method" \
      --lengths "$length" \
      --tasks "${TASKS[@]}" \
      --num-samples "$NUM_SAMPLES" \
      --device "$DEVICE" \
      --data-root "$DATA_ROOT" \
      --output-root "$OUTPUT_ROOT"; then
      echo "Worker failed: $method / $length; continuing. Rerunning this .sh will retry it." >&2
    fi
  done
done

echo "All method/length jobs have been attempted."
echo "Predictions: $OUTPUT_ROOT"
