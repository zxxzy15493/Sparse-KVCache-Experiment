#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

ATTN_DIR="../../attn_patterns"
MODELS="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct-1M"

# ====== short budget: 1k input, sparsity 0.5, output 2 ======
python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method duo_attn \
  --attn_load_dir "${ATTN_DIR}" \
  --sparsities 0.5 \
  --input_max_tokens 1024 \
  --max_new_tokens 2 \
  --save_dir ./results

# ====== short budget: 1k input, sparsity 0.5, output 4096 ======
python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method duo_attn \
  --attn_load_dir "${ATTN_DIR}" \
  --sparsities 0.5 \
  --input_max_tokens 1024 \
  --max_new_tokens 4096 \
  --save_dir ./results

# ====== 64k input, sparsity 0.5, output 2 ======
python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method duo_attn \
  --attn_load_dir "${ATTN_DIR}" \
  --sparsities 0.5 \
  --input_max_tokens 65536 \
  --max_new_tokens 2 \
  --save_dir ./results

# ====== long budget: 4k-128k, sparsity 0.5, output 2 ======
python memory.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method duo_attn \
  --attn_load_dir "${ATTN_DIR}" \
  --sparsities 0.5 \
  --input_max_tokens 4096 8192 16384 32768 65536 131072 \
  --max_new_tokens 2 \
  --save_dir ./results