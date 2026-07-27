#!/bin/bash
#
# run_myllama_breakdown.sh
#
# Measure per-component time for ClusterKV myllama Llama.
#
# Usage:
#   bash breakdown_test/run_myllama_breakdown.sh [MODEL] [INPUT_LENS] [MAX_NEW_TOKENS] [BUDGET]
#
# Examples:
#   bash breakdown_test/run_myllama_breakdown.sh
#   bash breakdown_test/run_myllama_breakdown.sh "meta-llama/Llama-3.1-8B-Instruct" 4096 32 512
#   OFFLOAD=1 bash breakdown_test/run_myllama_breakdown.sh "meta-llama/Llama-3.1-8B-Instruct" "4096,8192" 32 512
#

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,13p' "$0"
    exit 0
fi

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
INPUT_LENS="${2:-4096,8192,16384,32768,65536,131072}"
MAX_NEW_TOKENS="${3:-32}"
BUDGET="${4:-1024}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_FILE="${INPUT_FILE:-${SCRIPT_DIR}/myinput.txt}"
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
WARMUP_ROUNDS="${WARMUP_ROUNDS:-2}"
MEASURE_ROUNDS="${MEASURE_ROUNDS:-3}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"
METHOD="${METHOD:-clusterkv}"
NLIST="${NLIST:-200}"
NITER="${NITER:-20}"
SINK="${SINK:-16}"
WINDOW="${WINDOW:-320}"
WINDOW_NLIST="${WINDOW_NLIST:-8}"
OFFLOAD="${OFFLOAD:-1}"
CSV_OUT="${CSV_OUT:-./breakdown_test/log/myllama_breakdown_results.csv}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export SYNC_TEST_TIME="${SYNC_TEST_TIME:-1}"

echo "============================================================"
echo " MyLlama ClusterKV Component Breakdown Test"
echo "============================================================"
echo " MODEL          = ${MODEL}"
echo " INPUT_LENS     = ${INPUT_LENS}"
echo " MAX_NEW_TOKENS = ${MAX_NEW_TOKENS}"
echo " BUDGET         = ${BUDGET}"
echo " INPUT_FILE     = ${INPUT_FILE}"
echo " WARMUP/MR      = ${WARMUP_ROUNDS} / ${MEASURE_ROUNDS}"
echo " DEVICE         = ${DEVICE}"
echo " DTYPE          = ${DTYPE}"
echo " METHOD         = ${METHOD}"
echo " NLIST/NITER    = ${NLIST} / ${NITER}"
echo " WINDOW         = ${WINDOW}"
echo " WINDOW_NLIST   = ${WINDOW_NLIST}"
echo " OFFLOAD        = ${OFFLOAD}"
echo " CSV            = ${CSV_OUT}"
echo "------------------------------------------------------------"

cmd=(
    python "${SCRIPT_DIR}/breakdown_myllama.py"
    --model "${MODEL}"
    --input-file "${INPUT_FILE}"
    --input-lens "${INPUT_LENS}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --budget "${BUDGET}"
    --csv "${CSV_OUT}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
    --method "${METHOD}"
    --nlist "${NLIST}"
    --niter "${NITER}"
    --window "${WINDOW}"
    --window-nlist "${WINDOW_NLIST}"
    --warmup-rounds "${WARMUP_ROUNDS}"
    --measure-rounds "${MEASURE_ROUNDS}"
)

if [[ -n "${SINK}" ]]; then
    cmd+=(--sink "${SINK}")
fi

if [[ "${OFFLOAD}" == "1" ]]; then
    cmd+=(--offload)
fi

"${cmd[@]}"

echo ""
echo "Done. Results saved to: ${CSV_OUT}"
