#!/usr/bin/env bash
# Compare Full Flash Attention and RetroInfer for Qwen3-32B at 4K--128K.
# Optional arguments: <budget> <fixed_output_length> <prefill_method>
set -uo pipefail

BUDGET="${1:-1024}"
FIXED_OUTPUT_LENGTH="${2:-2}"
PREFILL_METHOD="${3:-Full_Flash_Attn}"
LENGTHS=(4096 8192 16384 32768 65536 131072)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

for attn_type in Full_Flash_Attn RetroInfer; do
    for input_length in "${LENGTHS[@]}"; do
        if ! bash ./VRAMOverview.sh qwen3-32b "$PREFILL_METHOD" "$attn_type" \
            "$BUDGET" "$input_length" "$FIXED_OUTPUT_LENGTH"; then
            # An OOM at a long Full-Attention point is an expected measurement
            # outcome; continue the sweep so RetroInfer points still run.
            echo "[failed] ${attn_type} at ${input_length} tokens" >&2
        fi
    done
done
