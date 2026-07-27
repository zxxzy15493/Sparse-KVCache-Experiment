#!/usr/bin/env bash
set -euo pipefail

method=pqcache
experiment=overview
models=(llama-3.1-8b qwen-2.5-7b qwen-2.5-7b-1m glm-4-9b-1m ds-qwen-1.5b)
budgets=(1024)
datasets=(narrativeqa qasper 2wikimqa musique gov_report multi_news triviaqa samsum passage_count passage_retrieval_en lcc repobench-p)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
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
