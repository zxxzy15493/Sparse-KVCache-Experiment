#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

MODELS="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct-1M"

# ====== 4k input, different budget ======
python efficiency.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method ReasonKV \
  --max_capacity_prompts 128 \
  --head_choice reason \
  --beta 1.5 --temp 1.0 \
  --input_max_tokens 4096 \
  --max_new_tokens 32 \
  --save_dir ./results

python efficiency.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method ReasonKV \
  --max_capacity_prompts 256 \
  --head_choice reason \
  --beta 1.5 --temp 1.0 \
  --input_max_tokens 4096 \
  --max_new_tokens 32 \
  --save_dir ./results

python efficiency.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method ReasonKV \
  --max_capacity_prompts 512 \
  --head_choice reason \
  --beta 1.5 --temp 1.0 \
  --input_max_tokens 4096 \
  --max_new_tokens 32 \
  --save_dir ./results

python efficiency.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method ReasonKV \
  --max_capacity_prompts 1024 \
  --head_choice reason \
  --beta 1.5 --temp 1.0 \
  --input_max_tokens 4096 \
  --max_new_tokens 32 \
  --save_dir ./results

# ====== 64k input, different budget ======
python efficiency.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method ReasonKV \
  --max_capacity_prompts 128 \
  --head_choice reason \
  --beta 1.5 --temp 1.0 \
  --input_max_tokens 65536 \
  --max_new_tokens 32 \
  --save_dir ./results

python efficiency.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method ReasonKV \
  --max_capacity_prompts 384 \
  --head_choice reason \
  --beta 1.5 --temp 1.0 \
  --input_max_tokens 65536 \
  --max_new_tokens 32 \
  --save_dir ./results

python efficiency.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method ReasonKV \
  --max_capacity_prompts 1024 \
  --head_choice reason \
  --beta 1.5 --temp 1.0 \
  --input_max_tokens 65536 \
  --max_new_tokens 32 \
  --save_dir ./results

python efficiency.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method ReasonKV \
  --max_capacity_prompts 4096 \
  --head_choice reason \
  --beta 1.5 --temp 1.0 \
  --input_max_tokens 65536 \
  --max_new_tokens 32 \
  --save_dir ./results