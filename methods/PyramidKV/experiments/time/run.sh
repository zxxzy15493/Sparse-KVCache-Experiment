#!/bin/bash
# PyramidKV timing test

cd "$(dirname "$0")"
REPO_ROOT=$(cd ../../../.. && pwd)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# 4k
python run_timing_test.py \
    --model llama3.1-8b-128k \
    --input_max_token 4096 \
    --max_capacity_prompts 1024

# 64k
python run_timing_test.py \
    --model llama3.1-8b-128k \
    --input_max_token 65536 \
    --max_capacity_prompts 1024
