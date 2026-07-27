#!/bin/bash
set -euo pipefail
set -x

COMPRESSOR=pq_search
EXP_NAME=overview
SINK_SIZE=16
RECENT_SIZE=32
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
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export OMP_NUM_THREADS
    MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export MKL_NUM_THREADS
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    export OPENBLAS_NUM_THREADS
    NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
    export NUMEXPR_NUM_THREADS
    VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
    export VECLIB_MAXIMUM_THREADS
    "$script_dir/run_longbench.sh"
done
