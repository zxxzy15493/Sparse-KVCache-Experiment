#!/bin/bash
set -euo pipefail
set -x

COMPRESSOR=no_drop_lb
EXP_NAME=overview
SINK_SIZE=0
RECENT_SIZE=0
BUDGET=1024

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

for MODEL in llama-3.1 qwen-2.5-7b glm-9b; do
    export MODEL
    export COMPRESSOR
    export EXP_NAME
    export SINK_SIZE
    export RECENT_SIZE
    export BUDGET
    "$script_dir/run_longbench.sh"
done
