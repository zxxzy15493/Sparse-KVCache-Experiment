#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
METHODS_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="${METHODS_DIR}:${PYTHONPATH:-}"
MODELS="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct"

# ====== short budget: 1k input, budget 64/512, output 2 ======
python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --cache_size 64 \
  --input_max_tokens 1024 \
  --max_new_tokens 2 \
  --save_dir ./results

python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --cache_size 512 \
  --input_max_tokens 1024 \
  --max_new_tokens 2 \
  --save_dir ./results

# ====== short budget: 1k input, budget 64/512, output 4096 ======
python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --cache_size 64 \
  --input_max_tokens 1024 \
  --max_new_tokens 4096 \
  --save_dir ./results

python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --cache_size 512 \
  --input_max_tokens 1024 \
  --max_new_tokens 4096 \
  --save_dir ./results

# ====== 64k input, budget 512/8192, output 2 ======
python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --cache_size 512 \
  --input_max_tokens 65536 \
  --max_new_tokens 2 \
  --save_dir ./results

python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --cache_size 8192 \
  --input_max_tokens 65536 \
  --max_new_tokens 2 \
  --save_dir ./results

# ====== long budget: 4k-128k, budget 1024, output 2 ======
python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --cache_size 1024 \
  --input_max_tokens 4096 8192 16384 32768 65536 131072 \
  --max_new_tokens 2 \
  --save_dir ./results