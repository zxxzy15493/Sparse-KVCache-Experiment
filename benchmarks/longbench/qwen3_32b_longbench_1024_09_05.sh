#!/usr/bin/env bash
# Run the first Qwen3-32B LongBench point only:
#   budget=1024, threshold=0.9, DuoAttention sparsity=0.5.
# The default list excludes methods that the Qwen3-32B smoke test established
# cannot run in this environment.  It is ordered from faster to slower.
set -uo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B}"
EXPERIMENT="${EXPERIMENT:-qwen3-32b-b1024-t090-s050}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
STATUS_FILE="${STATUS_FILE:-$(dirname "${BASH_SOURCE[0]}")/qwen3_32b_longbench_1024_09_05.status.tsv}"

BUDGET=1024
THRESHOLD=0.9
DUO_SPARSITY=0.5

# Measured speed order.  RetroInfer, MagicPig, MInference, and DuoAttention
# are intentionally omitted because they are unavailable for Qwen3-32B here.
METHODS=(
  # full_attention
  topk topp
  streaming h2o snapkv quest headkv keyformer
  # flexprefill adakv  xattention pyramidkv
  # cakekv clusterkv sparq pqcache
)
if [[ -n "${METHODS_OVERRIDE:-}" ]]; then
  read -r -a METHODS <<< "$METHODS_OVERRIDE"
fi

DEFAULT_DATASETS=(
  # narrativeqa qasper multifieldqa_en multifieldqa_zh hotpotqa 2wikimqa
  # musique dureader gov_report qmsum multi_news vcsum trec triviaqa samsum
  # lsht passage_count passage_retrieval_en passage_retrieval_zh lcc repobench-p
  narrativeqa qasper 2wikimqa musique gov_report multi_news triviaqa samsum passage_count passage_retrieval_en lcc repobench-p
)
read -r -a DATASETS <<< "${DATASETS:-${DEFAULT_DATASETS[*]}}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  DATASETS=("${SMOKE_DATASET:-qasper}")
  LIMIT="${LIMIT:-1}"
else
  LIMIT="${LIMIT:-}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PRED="$SCRIPT_DIR/longbench_pred.py"
EVAL="$SCRIPT_DIR/longbench_eval.py"

printf 'method\tparameter\tprediction\tevaluation\n' > "$STATUS_FILE"

common_pred=(--model "$MODEL_ID" --experiment "$EXPERIMENT" --datasets "${DATASETS[@]}")
common_eval=(--model "$MODEL_ID" --experiment "$EXPERIMENT" --datasets "${DATASETS[@]}")
if [[ -n "$OUTPUT_ROOT" ]]; then
  common_pred+=(--output-root "$OUTPUT_ROOT")
  common_eval+=(--output-root "$OUTPUT_ROOT")
fi
if [[ -n "$LIMIT" ]]; then
  common_pred+=(--limit "$LIMIT")
fi

run_one() {
  local method=$1
  local label=$2
  shift 2
  local -a method_args=(--method "$method" --budget "$BUDGET" "$@")
  local prediction=fail
  local evaluation=not-run

  echo "===== $method / $label / $MODEL_ID ====="
  if "$PYTHON_BIN" "$PRED" "${common_pred[@]}" "${method_args[@]}"; then
    prediction=pass
    if "$PYTHON_BIN" "$EVAL" "${common_eval[@]}" "${method_args[@]}"; then
      evaluation=pass
    else
      evaluation=fail
    fi
  fi
  printf '%s\t%s\t%s\t%s\n' "$method" "$label" "$prediction" "$evaluation" >> "$STATUS_FILE"
}

for method in "${METHODS[@]}"; do
  case "$method" in
    topp|flexprefill|xattention)
      run_one "$method" "budget-$BUDGET,threshold-$THRESHOLD" --set "fixthreshold=$THRESHOLD"
      ;;
    duo-attention)
      # Available only when a Qwen3-32B head-selection table is supplied.
      run_one "$method" "budget-$BUDGET,sparsity-$DUO_SPARSITY" --set "sparsity=$DUO_SPARSITY"
      ;;
    *)
      run_one "$method" "budget-$BUDGET"
      ;;
  esac
done

echo "Status written to $STATUS_FILE"
