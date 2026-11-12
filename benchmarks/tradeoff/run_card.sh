#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ruler_dir=$(cd -- "$script_dir/../ruler" && pwd)
logs_dir="$script_dir/logs"
mkdir -p "$logs_dir"

gpu=0
models=(llama-3.1-8b qwen-2.5-7b-1m)
tasks=(niah_single_3 vt fwe)
lengths=(65536)
budgets=(
    128
    256
    512
    1024
    2048
    3072
    4096
    5120
    6144
    7168
    8192
)

llama_methods=(
    # topk
    snapkv
    retroinfer
    cakekv
    clusterkv
    adakv
    magicpig
    h2o
)
qwen_methods=(
    streaming
    pyramidkv
    sparq
    pqcache
    headkv
    quest
    keyformer
)

export CUDA_VISIBLE_DEVICES="$gpu"
export RULER_BENCHMARK_ROOT="$script_dir/data"
export RULER_BENCHMARK_PRED_ROOT="$script_dir/pred"

for model in "${models[@]}"; do
    case "$model" in
        llama-3.1-8b)
            methods=("${llama_methods[@]}")
            ;;
        qwen-2.5-7b-1m)
            methods=("${qwen_methods[@]}")
            ;;
    esac

    for method in "${methods[@]}"; do
        log_file="$logs_dir/${model}_${method}.log"
        {
            printf '\n[%s] card 0 / GPU %s: %s %s\n' \
                "$(date '+%Y-%m-%d %H:%M:%S')" "$gpu" "$model" "$method"
            printf 'tasks: %s\n' "${tasks[*]}"
            printf 'lengths: %s\n' "${lengths[*]}"
            printf 'budgets: %s\n' "${budgets[*]}"
            bash "$ruler_dir/run_experiment.sh" "$method" \
                --models "$model" \
                --tasks "${tasks[@]}" \
                --lengths "${lengths[@]}" \
                --budgets "${budgets[@]}"
        } >>"$log_file" 2>&1
    done
done
