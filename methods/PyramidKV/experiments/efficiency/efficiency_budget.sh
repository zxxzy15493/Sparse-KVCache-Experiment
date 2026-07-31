#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
METHODS_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${METHODS_DIR}:${PYTHONPATH:-}"

MODELS="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct"

# ====== 4k input, different budget ======
for budget in 128 256 512 1024; do
  python efficiency.py \
    --models ${MODELS} \
    --input_file ../../../../benchmarks/myinput.txt \
    --method pyramidkv \
    --max_capacity_prompts ${budget} \
    --input_max_tokens 4096 \
    --max_new_tokens 32 \
    --save_dir ./results
done

# ====== 64k input, different budget ======
for budget in 128 384 1024 4096; do
  python efficiency.py \
    --models ${MODELS} \
    --input_file ../../../../benchmarks/myinput.txt \
    --method pyramidkv \
    --max_capacity_prompts ${budget} \
    --input_max_tokens 65536 \
    --max_new_tokens 32 \
    --save_dir ./results
done