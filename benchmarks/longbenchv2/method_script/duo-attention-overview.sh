#!/usr/bin/env bash
set -euo pipefail

method=duo-attention
models=(qwen-2.5-7b-1m)
sparsities=(0.5)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1
benchmark_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

for model in "${models[@]}"; do
    for sparsity in "${sparsities[@]}"; do
        python "$benchmark_dir/longbenchv2_pred.py" --method "$method" --model "$model" --set "sparsity=$sparsity" --cot
        python "$benchmark_dir/longbenchv2_eval.py" --model "$model" --method "$method" --set "sparsity=$sparsity"
    done
done
