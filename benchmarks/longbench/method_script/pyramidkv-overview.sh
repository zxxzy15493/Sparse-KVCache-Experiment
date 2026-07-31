#!/usr/bin/env bash
set -euo pipefail

method=pyramidkv
experiment=overview
models=(llama-3.1-8b qwen-2.5-7b glm-4-9b-1m)
budgets=(1024)
datasets=(narrativeqa qasper 2wikimqa musique gov_report multi_news triviaqa samsum passage_count passage_retrieval_en lcc repobench-p)

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