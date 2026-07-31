#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

MODELS="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct-1M"

# ====== short budget: 1k input, budget 64/512, output 2 ======
python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method AdativeKV \
  --max_capacity_prompts 64 \
  --head_choice random \
  --input_max_tokens 1024 \
  --max_new_tokens 2 \
  --save_dir ./results

python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method AdativeKV \
  --max_capacity_prompts 512 \
  --head_choice random \
  --input_max_tokens 1024 \
  --max_new_tokens 2 \
  --save_dir ./results

# ====== short budget: 1k input, budget 64/512, output 4096 ======
python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method AdativeKV \
  --max_capacity_prompts 64 \
  --head_choice random \
  --input_max_tokens 1024 \
  --max_new_tokens 4096 \
  --save_dir ./results

python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method AdativeKV \
  --max_capacity_prompts 512 \
  --head_choice random \
  --input_max_tokens 1024 \
  --max_new_tokens 4096 \
  --save_dir ./results

# ====== 64k input, budget 512/8192, output 2 ======
python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method AdativeKV \
  --max_capacity_prompts 512 \
  --head_choice random \
  --input_max_tokens 65536 \
  --max_new_tokens 2 \
  --save_dir ./results

python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method AdativeKV \
  --max_capacity_prompts 8192 \
  --head_choice random \
  --input_max_tokens 65536 \
  --max_new_tokens 2 \
  --save_dir ./results

# ====== long budget: 4k-128k, budget 1024, output 2 ======
python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method AdativeKV \
  --max_capacity_prompts 1024 \
  --head_choice random \
  --input_max_tokens 4096 8192 16384 32768 65536 131072 \
  --max_new_tokens 2 \
  --save_dir ./results