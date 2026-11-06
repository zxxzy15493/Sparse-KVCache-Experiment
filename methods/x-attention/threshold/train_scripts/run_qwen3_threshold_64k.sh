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

MODEL_PATH="${MODEL_PATH:-/data/zc/models/Qwen3-32B}"
TEXT_PATH="${TEXT_PATH:-${XATTN_ROOT}/threshold/qwen3_64k.json}"
OUTPUT_DIR="${OUTPUT_DIR:-threshold/threshold}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/qwen3_32b_yarn2_64k_p${P_TAG}_s16.json}"
CHUNK_TEXTS="${CHUNK_TEXTS:-1}"
BATCH_OUTPUT_DIR="${BATCH_OUTPUT_DIR:-}"
MAX_TEXTS="${MAX_TEXTS:-0}"
RESUME_CHUNKS="${RESUME_CHUNKS:-true}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
DEVICE_MAP="${DEVICE_MAP:-balanced}"
export CUDA_VISIBLE_DEVICES

# Qwen recommends YaRN 2x when the typical context length is 65,536 tokens.
ROPE_SCALING_JSON='{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}'
MAX_POSITION_EMBEDDINGS=65536

if [[ ! -f "${TEXT_PATH}" ]]; then
  echo "TEXT_PATH does not exist: ${TEXT_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
echo "[qwen3-threshold] 64K devices=${CUDA_VISIBLE_DEVICES}, device_map=${DEVICE_MAP}"

memory_args=()
if [[ -n "${MAX_MEMORY:-}" ]]; then
  memory_args+=(--max_memory "${MAX_MEMORY}")
fi

chunk_args=()
if [[ "${CHUNK_TEXTS}" != "0" ]]; then
  chunk_args+=(--chunk_texts "${CHUNK_TEXTS}")
  if [[ -n "${BATCH_OUTPUT_DIR}" ]]; then
    chunk_args+=(--batch_output_dir "${BATCH_OUTPUT_DIR}")
  fi
fi

python threshold/train_scripts/train_threshold.py \
  --name_or_path "${MODEL_PATH}" \
  --model_type qwen3 \
  --device_map "${DEVICE_MAP}" \
  --reserve_memory_gib "${RESERVE_MEMORY_GIB:-8}" \
  --head_chunk_size "${HEAD_CHUNK_SIZE:-4}" \
  --estimate_chunk_size "${ESTIMATE_CHUNK_SIZE:-16384}" \
  --mlp_chunk_size "${MLP_CHUNK_SIZE:-4096}" \
  --p "${P_VALUE}" \
  --stride 16 \
  --block_size 128 \
  --torch_dtype bfloat16 \
  --attn_implementation flash_attention_2 \
  --text_path "${TEXT_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --output_format json \
  --max_texts "${MAX_TEXTS}" \
  --resume_chunks "${RESUME_CHUNKS}" \
  --rope_scaling_json "${ROPE_SCALING_JSON}" \
  --max_position_embeddings "${MAX_POSITION_EMBEDDINGS}" \
  "${memory_args[@]}" \
  "${chunk_args[@]}" \
  "$@"
