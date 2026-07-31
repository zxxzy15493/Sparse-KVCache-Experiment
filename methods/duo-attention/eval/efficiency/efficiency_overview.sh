#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
METHODS_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${METHODS_DIR}:${PYTHONPATH:-}"

ATTN_BASE="../../attn_patterns"
MODELS="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct-1M"

for model in ${MODELS}; do
    case "${model}" in
        meta-llama/Llama-*)
            attn_dir="${ATTN_BASE}/Meta-Llama-3.1-8B-Instruct"
            ;;
        Qwen/Qwen2.5-7B-Instruct*)
            attn_dir="${ATTN_BASE}/Qwen2.5-7B-Instruct"
            ;;
        *)
            echo "[ERROR] Unknown model: ${model}, no attn_patterns mapping"
            continue
            ;;
    esac

    python efficiency.py \
      --model "${model}" \
      --input_file ../../../../benchmarks/myinput.txt \
      --method duo_attn \
      --attn_load_dir "${attn_dir}" \
      --sparsities 0.5 \
      --input_max_tokens 4096 8192 16384 32768 65536 131072 \
      --max_new_tokens 32 \
      --save_dir ./results
done