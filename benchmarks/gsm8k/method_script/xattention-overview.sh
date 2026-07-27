#!/usr/bin/env bash
set -euo pipefail

method=xattention
models=(ds-qwen-1.5b)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
benchmark_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

for model in "${models[@]}"; do
    python "$benchmark_dir/gsm8k_pred.py" --method "$method" --model "$model" --set "fixthreshold=0.8"
    python "$benchmark_dir/gsm8k_eval.py" --model "$model" --method "$method" --set "fixthreshold=0.8"
done
