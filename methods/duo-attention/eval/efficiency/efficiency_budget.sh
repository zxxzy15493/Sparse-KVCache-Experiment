#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

ATTN_DIR="../../attn_patterns"
MODELS="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct-1M"

# ====== 4k input, differentsparsity ======
python efficiency.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method duo_attn \
  --attn_load_dir "${ATTN_DIR}" \
  --sparsities 0.6 0.7 0.8 0.9 \
  --input_max_tokens 4096 \
  --max_new_tokens 32 \
  --save_dir ./results

# ====== 64k input, differentsparsity ======
python efficiency.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --method duo_attn \
  --attn_load_dir "${ATTN_DIR}" \
  --sparsities 0.6 0.7 0.8 0.9 \
  --input_max_tokens 65536 \
  --max_new_tokens 32 \
  --save_dir ./results