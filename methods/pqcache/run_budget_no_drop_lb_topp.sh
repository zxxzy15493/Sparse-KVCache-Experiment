#!/bin/bash
set -euo pipefail
set -x

COMPRESSOR=no_drop_lb_topp
EXP_NAME=budget
SINK_SIZE=0
RECENT_SIZE=0
BUDGET=1024

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

for FIXTHRESHOLD in 0.8 0.85 0.9 0.95; do
    for MODEL in llama-3.1 qwen-2.5-7b; do
        export MODEL
        export COMPRESSOR
        export EXP_NAME
        export SINK_SIZE
        export RECENT_SIZE
        export BUDGET
        export FIXTHRESHOLD
        "$script_dir/run_budget_method.sh"
    done
done
