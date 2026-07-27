#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
MODELS="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct-1M"

python efficiency.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method AdativeKV \
  --max_capacity_prompts 1024 \
  --head_choice random \
  --input_max_tokens 4096 8192 16384 32768 65536 131072 \
  --max_new_tokens 32 \
  --save_dir ./results