#!/usr/bin/env bash
set -euo pipefail

# Official Hugging Face model ID is passed through to from_pretrained directly.
model="${MODEL_ID:-Qwen/Qwen3-8B}"
experiment="${EXPERIMENT:-overview}"
method_list="${METHODS:-full_attention topk topk32 topp topp32 pqcache clusterkv h2o keyformer snapkv streaming quest sparq xattention flexprefill pyramidkv adakv cakekv}"
dataset_list="${DATASETS:-narrativeqa qasper 2wikimqa musique gov_report multi_news triviaqa samsum passage_count passage_retrieval_en lcc repobench-p}"
read -r -a methods <<< "$method_list"
read -r -a datasets <<< "$dataset_list"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
longbench_dir=$(cd -- "$script_dir" && pwd)
python_bin="${PYTHON_BIN:-python}"
pred_common_args=(--model "$model" --experiment "$experiment" --datasets "${datasets[@]}")
eval_common_args=(--model "$model" --experiment "$experiment" --datasets "${datasets[@]}")
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
    pred_common_args+=(--output-root "$OUTPUT_ROOT")
    eval_common_args+=(--output-root "$OUTPUT_ROOT")
fi
if [[ -n "${LIMIT:-}" ]]; then
    pred_common_args+=(--limit "$LIMIT")
fi

for method in "${methods[@]}"; do
    method_args=(--method "$method")
    eval_args=(--method "$method")
    case "$method" in
        topp|topp32|flexprefill|xattention)
            method_args+=(--set fixthreshold=0.9)
            eval_args+=(--set fixthreshold=0.9)
            ;;
        duo-attention)
            method_args+=(--set sparsity=0.5)
            eval_args+=(--set sparsity=0.5)
            ;;
        *)
            method_args+=(--budget 1024)
            eval_args+=(--budget 1024)
            ;;
    esac
    echo "===== Qwen3 LongBench overview: $method / $model ====="
    "$python_bin" "$longbench_dir/longbench_pred.py" "${pred_common_args[@]}" "${method_args[@]}"
    "$python_bin" "$longbench_dir/longbench_eval.py" "${eval_common_args[@]}" "${eval_args[@]}"
done
