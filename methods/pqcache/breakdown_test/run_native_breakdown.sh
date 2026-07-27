#!/bin/bash
#
# run_native_breakdown.sh
#
# Measure per-component GPU time for a NATIVE (uncompressed) Llama or Qwen2 model.
#
# Usage:
#   bash breakdown_test/run_native_breakdown.sh [MODEL] [INPUT_LENS] [MAX_NEW_TOKENS]
#
# Examples:
#   # Default: Llama-3.1-8B-Instruct, 4k..128k inputs, 32 output tokens
#   bash breakdown_test/run_native_breakdown.sh
#
#   # Llama-3.1-8B at 4k
#   bash breakdown_test/run_native_breakdown.sh "meta-llama/Llama-3.1-8B-Instruct" 4096
#
#   # Llama-3.1-8B at multiple lengths
#   bash breakdown_test/run_native_breakdown.sh "meta-llama/Llama-3.1-8B-Instruct" "4096,8192,16384"
#
#   # Qwen2.5-7B at 64k
#   bash breakdown_test/run_native_breakdown.sh "Qwen/Qwen2.5-7B-Instruct" 65536 32
#   bash run_native_breakdown.sh "Qwen/Qwen2.5-7B-Instruct" 
#
#   # Qwen2.5-7B with flash_attention_2 at 32k
#   ATTN_IMPL=flash_attention_2 bash breakdown_test/run_native_breakdown.sh "Qwen/Qwen2.5-7B-Instruct" 32768
#
#   # Qwen2.5-7B with eager attention at 4k (quick test)
#   ATTN_IMPL=eager bash breakdown_test/run_native_breakdown.sh "Qwen/Qwen2.5-7B-Instruct" 4096 16
#   
#   # TinyLlama (quick test) at 4k
#   bash breakdown_test/run_native_breakdown.sh "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 4096 16
#
#   # Llama-3.1-8B with flash_attention_2 at 32k
#   ATTN_IMPL=flash_attention_2 bash breakdown_test/run_native_breakdown.sh "meta-llama/Llama-3.1-8B-Instruct" 32768
#
#   # Override model type detection
#   MODEL_TYPE=qwen2 bash breakdown_test/run_native_breakdown.sh "Qwen/Qwen2.5-7B-Instruct" 4096
#

set -euo pipefail

LOG_DIR="${LOG_DIR:-./breakdown_test/log}"
mkdir -p "${LOG_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/native_breakdown_${RUN_TS}.log}"
exec > "${LOG_FILE}" 2>&1

# ---- configurable defaults ----
MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
INPUT_LENS="${2:-4096,8192,16384,32768,65536,131072}"  # comma-separated, no spaces
MAX_NEW_TOKENS="${3:-32}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"         # sdpa | flash_attention_2 | eager
MODEL_TYPE="${MODEL_TYPE:-auto}"     # auto | llama | qwen2
INPUT_FILE="${INPUT_FILE:-./myinput.txt}"
# Ensure myinput.txt exists at INPUT_FILE; copy from benchmarks/ via relative path if missing
if [ ! -f "${INPUT_FILE}" ]; then
    _SRC="$(dirname "$0")/../../../benchmarks/myinput.txt"
    if [ -f "${_SRC}" ]; then
        case "${INPUT_FILE}" in
            /*) _DEST="${INPUT_FILE}" ;;
            *)  _DEST="$(dirname "$0")/${INPUT_FILE#./}" ;;
        esac
        mkdir -p "$(dirname "${_DEST}")"
        cp "${_SRC}" "${_DEST}"
    fi
fi
unset _SRC _DEST
WARMUP_ROUNDS="${WARMUP_ROUNDS:-3}"
MEASURE_ROUNDS="${MEASURE_ROUNDS:-7}"
DEVICE="${DEVICE:-cuda:0}"
CSV_OUT="${CSV_OUT:-./breakdown_test/log/native_breakdown_results_${RUN_TS}.csv}"

# Make sure we can import from the project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "============================================================"
echo " Native Llama / Qwen2 Component Breakdown Test"
echo "============================================================"
echo " MODEL          = ${MODEL}"
echo " INPUT_LENS     = ${INPUT_LENS}"
echo " MAX_NEW_TOKENS = ${MAX_NEW_TOKENS}"
echo " ATTN_IMPL      = ${ATTN_IMPL}"
echo " MODEL_TYPE     = ${MODEL_TYPE}"
echo " INPUT_FILE     = ${INPUT_FILE}"
echo " WARMUP/MR      = ${WARMUP_ROUNDS} / ${MEASURE_ROUNDS}"
echo " CSV            = ${CSV_OUT}"
echo " DEVICE         = ${DEVICE}"
echo " LOG_FILE       = ${LOG_FILE}"
echo "------------------------------------------------------------"

python "${SCRIPT_DIR}/breakdown_native.py" \
    --model "${MODEL}" \
    --input-file "${INPUT_FILE}" \
    --input-lens ${INPUT_LENS} \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --attn-implementation "${ATTN_IMPL}" \
    --model-type "${MODEL_TYPE}" \
    --warmup-rounds "${WARMUP_ROUNDS}" \
    --measure-rounds "${MEASURE_ROUNDS}" \
    --csv "${CSV_OUT}" \
    --device "${DEVICE}"

echo ""
echo "Done. Results saved to: ${CSV_OUT}"
