#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
cd "$(dirname "$0")"
METHODS_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="${METHODS_DIR}:${PYTHONPATH:-}"
MODELS="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct"

python efficiency.py \
  --models ${MODELS} \
  --input_file ../../../../benchmarks/myinput.txt \
  --cache_size 1024 \
  --input_max_tokens 4096 8192 16384 32768 65536 131072 \
  --max_new_tokens 32 \
  --save_dir ./results