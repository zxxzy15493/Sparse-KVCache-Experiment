#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XATTN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${XATTN_ROOT}"

if [[ -n "${P:-}" ]]; then
  P_VALUE="${P}"
else
  P_VALUE="${1:-0.9}"
  if [[ $# -gt 0 ]]; then
    shift
  fi
fi

case "${P_VALUE}" in
  0.8|.8) P_TAG="80" ;;
  0.85|.85) P_TAG="85" ;;
  0.9|.9) P_TAG="90" ;;
  0.95|.95) P_TAG="95" ;;
  *) P_TAG="$(printf '%s' "${P_VALUE}" | tr -d '.')" ;;
esac

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
TEXT_PATH="${TEXT_PATH:-threshold/text.json}"
OUTPUT_DIR="${OUTPUT_DIR:-xattn/threshold/threshold}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/llama_threshold_p${P_TAG}.py}"
CHUNK_TEXTS="${CHUNK_TEXTS:-0}"
BATCH_OUTPUT_DIR="${BATCH_OUTPUT_DIR:-}"

if [[ ! -f "${TEXT_PATH}" ]]; then
  echo "TEXT_PATH does not exist: ${TEXT_PATH}" >&2
  echo "Set TEXT_PATH=/path/to/text.json and rerun." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

chunk_args=()
if [[ "${CHUNK_TEXTS}" != "0" ]]; then
  chunk_args+=(--chunk_texts "${CHUNK_TEXTS}")
  if [[ -n "${BATCH_OUTPUT_DIR}" ]]; then
    chunk_args+=(--batch_output_dir "${BATCH_OUTPUT_DIR}")
  fi
fi

conda run -n xattn python threshold/train_scripts/train_threshold.py \
  --name_or_path "${MODEL_PATH}" \
  --model_type llama \
  --p "${P_VALUE}" \
  --text_path "${TEXT_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --output_format py \
  "${chunk_args[@]}" \
  "$@"
