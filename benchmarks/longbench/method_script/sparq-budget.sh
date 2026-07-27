#!/usr/bin/env bash
set -euo pipefail

method=sparq
experiment=budget
models=(llama-3.1-8b qwen-2.5-7b qwen-2.5-7b-1m glm-4-9b-1m ds-qwen-1.5b)
budgets=(128 256 512 1024)
datasets=(narrativeqa qasper trec lcc)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
longbench_dir=$(cd -- "$script_dir/.." && pwd)

for model in "${models[@]}"; do
    for budget in "${budgets[@]}"; do
        python "$longbench_dir/longbench_pred.py" --method "$method" --model "$model" --budget "$budget" --datasets "${datasets[@]}" --experiment "$experiment"
        python "$longbench_dir/longbench_eval.py" --model "$model" --experiment "$experiment" --method "$method" --datasets "${datasets[@]}" --budget "$budget"
    done
done
