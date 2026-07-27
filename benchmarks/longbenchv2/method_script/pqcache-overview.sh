#!/usr/bin/env bash
set -euo pipefail

method=pqcache
models=(qwen-2.5-7b-1m)
budgets=(4096)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1
benchmark_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

for model in "${models[@]}"; do
    for budget in "${budgets[@]}"; do
        python "$benchmark_dir/longbenchv2_pred.py" --method "$method" --model "$model" --budget "$budget"
        python "$benchmark_dir/longbenchv2_eval.py" --model "$model" --method "$method" --budget "$budget"
    done
done
