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
STATUS_FILE="${STATUS_FILE:-$SCRIPT_DIR/qwen3_32b_yarn2_status.tsv}"
REPORT_FILE="${REPORT_FILE:-$SCRIPT_DIR/qwen3_32b_yarn2_report.md}"

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

mkdir -p "$(dirname -- "$STATUS_FILE")" "$(dirname -- "$REPORT_FILE")"

echo "===== prepare RULER data: ${#LENGTHS[@]} lengths x ${#TASKS[@]} tasks x $NUM_SAMPLES samples ====="
if ! "$PYTHON_BIN" "$RUNNER" \
  --model "$MODEL_ID" \
  --lengths "${LENGTHS[@]}" \
  --tasks "${TASKS[@]}" \
  --num-samples "$NUM_SAMPLES" \
  --device "$DEVICE" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --status-file "$STATUS_FILE" \
  --report-file "$REPORT_FILE" \
  --prepare-only; then
  echo "RULER data preparation failed; stopping." >&2
  exit 1
fi

if [[ ! -s "$STATUS_FILE" ]]; then
  printf 'method\tlength\tstatus\telapsed_seconds\tload_seconds\tinference_seconds\texamples\tscore_mean\tlow_score_tasks\tslow\tprediction_dir\terror\n' > "$STATUS_FILE"
fi

EXPECTED_EXAMPLES=$((NUM_SAMPLES * ${#TASKS[@]}))
is_complete() {
  local method=$1 seq_len=$2 pred_dir task pred_file count
  pred_dir=$(awk -F '\t' -v method="$method" -v seq_len="$seq_len" -v expected="$EXPECTED_EXAMPLES" '
    NR > 1 && $1 == method && $2 == seq_len {
      status = $3
      examples = $7
      prediction_dir = $11
    }
    END {
      if (status == "complete" && examples + 0 >= expected + 0) print prediction_dir
      else exit 1
    }
  ' "$STATUS_FILE") || return 1
  [[ -n "$pred_dir" ]] || return 1
  for task in "${TASKS[@]}"; do
    pred_file="$pred_dir/$task.jsonl"
    [[ -f "$pred_file" ]] || return 1
    count=$(wc -l < "$pred_file")
    (( count >= NUM_SAMPLES )) || return 1
  done
  return 0
}

trap 'echo "Interrupted. Rerun this same .sh; completed samples will be skipped." >&2; exit 130' INT TERM

total=$((${#METHODS[@]} * ${#LENGTHS[@]}))
index=0
for method in "${METHODS[@]}"; do
  for length in "${LENGTHS[@]}"; do
    index=$((index + 1))
    if is_complete "$method" "$length"; then
      echo "===== [$index/$total] skip complete: $method / $length ====="
      continue
    fi

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
      --output-root "$OUTPUT_ROOT" \
      --status-file "$STATUS_FILE" \
      --report-file "$REPORT_FILE"; then
      echo "Worker failed: $method / $length; continuing. Rerunning this .sh will retry it." >&2
    fi
  done
done

echo "All method/length jobs have been attempted."
echo "Status: $STATUS_FILE"
echo "Report: $REPORT_FILE"
