#!/usr/bin/env bash
set -euo pipefail

method=duo-attention
models=(ds-qwen-1.5b)
sparsities=(0.5)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
benchmark_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

for model in "${models[@]}"; do
    for sparsity in "${sparsities[@]}"; do
        python "$benchmark_dir/gsm8k_pred.py" --method "$method" --model "$model" --set "sparsity=$sparsity"
        python "$benchmark_dir/gsm8k_eval.py" --model "$model" --method "$method" --set "sparsity=$sparsity"
    done
done
