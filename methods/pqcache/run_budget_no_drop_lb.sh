#!/bin/bash
set -euo pipefail
set -x

COMPRESSOR=no_drop_lb
EXP_NAME=budget
SINK_SIZE=0
RECENT_SIZE=0

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
        "$script_dir/run_budget_method.sh"
    done
done
