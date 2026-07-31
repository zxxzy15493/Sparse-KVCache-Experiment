#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

MODELS="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct-1M"

for budget in 64 512; do
  python memory.py \
    --models ${MODELS} \
    --input_file ../../../../benchmarks/myinput.txt \
    --method ReasonKV \
    --max_capacity_prompts ${budget} \
    --head_choice reason \
    --beta 1.5 --temp 1.0 \
    --input_max_tokens 1024 \
    --max_new_tokens 2 \
    --save_dir ./results

  python memory.py \
    --models ${MODELS} \
    --input_file ../../../../benchmarks/myinput.txt \
    --method ReasonKV \
    --max_capacity_prompts ${budget} \
    --head_choice reason \
    --beta 1.5 --temp 1.0 \
    --input_max_tokens 1024 \
    --max_new_tokens 4096 \
    --save_dir ./results
done

for budget in 512 8192; do
  python memory.py \
    --models ${MODELS} \
    --input_file ../../../../benchmarks/myinput.txt \
    --method ReasonKV \
    --max_capacity_prompts ${budget} \
    --head_choice reason \
    --beta 1.5 --temp 1.0 \
    --input_max_tokens 65536 \
    --max_new_tokens 2 \
    --save_dir ./results
done

python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method ReasonKV \
  --max_capacity_prompts 1024 \
  --head_choice reason \
  --beta 1.5 --temp 1.0 \
  --input_max_tokens 4096 8192 16384 32768 65536 131072 \
  --max_new_tokens 2 \
  --save_dir ./results