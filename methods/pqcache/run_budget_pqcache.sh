#!/bin/bash
set -euo pipefail
set -x

COMPRESSOR=pq_search
EXP_NAME=budget
SINK_SIZE=16
RECENT_SIZE=32

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

for BUDGET in 128 256 512 1024; do
    for MODEL in llama-3.1 qwen-2.5-7b; do
        export MODEL
        export COMPRESSOR
        export EXP_NAME
        export SINK_SIZE
        export RECENT_SIZE
        export BUDGET
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
        "$script_dir/run_budget_method.sh"
    done
done
