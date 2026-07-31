#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
METHODS_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${METHODS_DIR}:${PYTHONPATH:-}"

ATTN_BASE="../../attn_patterns"
MODELS="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct-1M"

run_efficiency() {
    local model=$1
    local attn_dir=$2
    local input_len=$3

    python efficiency.py \
      --model "${model}" \
      --input_file ../../../../benchmarks/myinput.txt \
      --method duo_attn \
      --attn_load_dir "${attn_dir}" \
      --sparsities 0.6 0.7 0.8 0.9 \
      --input_max_tokens "${input_len}" \
      --max_new_tokens 32 \
      --save_dir ./results
}

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

    # ====== 4k input, different sparsity ======
    run_efficiency "${model}" "${attn_dir}" 4096

    # ====== 64k input, different sparsity ======
    run_efficiency "${model}" "${attn_dir}" 65536
done