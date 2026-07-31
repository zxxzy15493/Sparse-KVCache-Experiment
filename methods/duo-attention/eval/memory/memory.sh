#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
METHODS_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${METHODS_DIR}:${PYTHONPATH:-}"

ATTN_BASE="../../attn_patterns"
MODELS="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct-1M"

get_attn_dir() {
    case "$1" in
        meta-llama/Llama-*)
            echo "${ATTN_BASE}/Meta-Llama-3.1-8B-Instruct"
            ;;
        Qwen/Qwen2.5-7B-Instruct*)
            echo "${ATTN_BASE}/Qwen2.5-7B-Instruct"
            ;;
        *)
            echo ""
            ;;
    esac
}

for model in ${MODELS}; do
    attn_dir="$(get_attn_dir "${model}")"
    [ -z "${attn_dir}" ] && { echo "[ERROR] Unknown model: ${model}"; continue; }

    echo "========================================="
    echo " Model: ${model}"
    echo " Attn patterns: ${attn_dir}"
    echo "========================================="

    # ====== 1k input, output 2 ======
    python memory.py \
      --model "${model}" \
      --input_file ../../../../benchmarks/myinput.txt \
      --method duo_attn \
      --attn_load_dir "${attn_dir}" \
      --sparsity 0.5 \
      --input_max_tokens 1024 \
      --max_new_tokens 2 \
      --save_dir ./results

    # ====== 1k input, output 4096 (decode-heavy) ======
    python memory.py \
      --model "${model}" \
      --input_file ../../../../benchmarks/myinput.txt \
      --method duo_attn \
      --attn_load_dir "${attn_dir}" \
      --sparsity 0.5 \
      --input_max_tokens 1024 \
      --max_new_tokens 4096 \
      --save_dir ./results

    # ====== 64k input, output 2 ======
    python memory.py \
      --model "${model}" \
      --input_file ../../../../benchmarks/myinput.txt \
      --method duo_attn \
      --attn_load_dir "${attn_dir}" \
      --sparsity 0.5 \
      --input_max_tokens 65536 \
      --max_new_tokens 2 \
      --save_dir ./results

    # ====== multiple input lengths: 4k-128k, output 2 ======
    python memory.py \
      --model "${model}" \
      --input_file ../../../../benchmarks/myinput.txt \
      --method duo_attn \
      --attn_load_dir "${attn_dir}" \
      --sparsity 0.5 \
      --input_max_tokens 4096 8192 16384 32768 65536 131072 \
      --max_new_tokens 2 \
      --save_dir ./results
done
  --max_new_tokens 2 \
  --save_dir ./results