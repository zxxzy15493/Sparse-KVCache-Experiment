#!/usr/bin/env bash
set -euo pipefail

method=topp
models=(ds-qwen-1.5b)
budgets=(360)
thresholds=(0.9)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
benchmark_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

for model in "${models[@]}"; do
    for budget in "${budgets[@]}"; do
        for threshold in "${thresholds[@]}"; do
            python "$benchmark_dir/gsm8k_pred.py" --method "$method" --model "$model" --budget "$budget" --set "fixthreshold=$threshold"
            python "$benchmark_dir/gsm8k_eval.py" --model "$model" --method "$method" --budget "$budget" --set "fixthreshold=$threshold"
        done
    done
done
