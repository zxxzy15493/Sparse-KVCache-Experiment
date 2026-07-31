#!/usr/bin/env bash
set -euo pipefail

method=duo-attention
experiment=budget
models=(llama-3.1-8b qwen-2.5-7b glm-4-9b-1m)
sparsities=(0.6 0.7 0.8 0.9)
datasets=(narrativeqa qasper trec lcc)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
longbench_dir=$(cd -- "$script_dir/.." && pwd)

for model in "${models[@]}"; do
    for sparsity in "${sparsities[@]}"; do
        python "$longbench_dir/longbench_pred.py" --method "$method" --model "$model" --datasets "${datasets[@]}" --experiment "$experiment" --set "sparsity=$sparsity"
        python "$longbench_dir/longbench_eval.py" --model "$model" --experiment "$experiment" --method "$method" --datasets "${datasets[@]}" --set "sparsity=$sparsity"
    done
done