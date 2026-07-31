#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
METHODS_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${METHODS_DIR}:${PYTHONPATH:-}"
MODELS="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct"

python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method pyramidkv \
  --max_capacity_prompts 64 \
  --input_max_tokens 1024 \
  --max_new_tokens 2 \
  --save_dir ./results

python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method pyramidkv \
  --max_capacity_prompts 512 \
  --input_max_tokens 1024 \
  --max_new_tokens 2 \
  --save_dir ./results

python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method pyramidkv \
  --max_capacity_prompts 64 \
  --input_max_tokens 1024 \
  --max_new_tokens 4096 \
  --save_dir ./results

python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method pyramidkv \
  --max_capacity_prompts 512 \
  --input_max_tokens 1024 \
  --max_new_tokens 4096 \
  --save_dir ./results

# ====== 64k input, budget 512/8192, output 2 ======
python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method pyramidkv \
  --max_capacity_prompts 512 \
  --input_max_tokens 65536 \
  --max_new_tokens 2 \
  --save_dir ./results

python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method pyramidkv \
  --max_capacity_prompts 8192 \
  --input_max_tokens 65536 \
  --max_new_tokens 2 \
  --save_dir ./results

python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method pyramidkv \
  --max_capacity_prompts 1024 \
  --input_max_tokens 4096 8192 16384 32768 65536 131072 \
  --max_new_tokens 2 \
  --save_dir ./results