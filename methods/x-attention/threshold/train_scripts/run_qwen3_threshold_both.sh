#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XATTN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULT_DIR="${OUTPUT_DIR:-${XATTN_ROOT}/threshold/threshold}"
if [[ "${RESULT_DIR}" != /* ]]; then
  RESULT_DIR="${XATTN_ROOT}/${RESULT_DIR}"
fi

p_tag() {
  case "$1" in
    0.8|.8) printf '80' ;;
    0.85|.85) printf '85' ;;
    0.9|.9) printf '90' ;;
    0.95|.95) printf '95' ;;
    *) printf '%s' "$1" | tr -d '.' ;;
  esac
}

run_if_incomplete() {
  local context="$1"
  local p_value="$2"
  local tag
  local result_path
  local runner

  tag="$(p_tag "${p_value}")"
  case "${context}" in
    64K)
      result_path="${RESULT_DIR}/qwen3_32b_yarn2_64k_p${tag}_s16.json"
      runner="${SCRIPT_DIR}/run_qwen3_threshold_64k.sh"
      ;;
    128K)
      result_path="${RESULT_DIR}/qwen3_32b_yarn4_128k_p${tag}_s16.json"
      runner="${SCRIPT_DIR}/run_qwen3_threshold_128k.sh"
      ;;
    *)
      echo "Unknown context: ${context}" >&2
      return 1
      ;;
  esac

  if [[ -s "${result_path}" ]]; then
    echo "[qwen3-threshold] skipping completed ${context} calibration with p=${p_value}: ${result_path}"
    return
  fi

  echo "[qwen3-threshold] starting ${context} calibration with p=${p_value}"
  bash "${runner}" "${p_value}"
  echo "[qwen3-threshold] finished ${context} calibration with p=${p_value}"
}

if [[ $# -gt 0 ]]; then
  P_VALUES=("$@")
elif [[ -n "${P:-}" ]]; then
  P_VALUES=("${P}")
else
  P_VALUES=(0.8 0.85 0.9 0.95)
fi

echo "[qwen3-threshold] p values: ${P_VALUES[*]}"

for p_value in "${P_VALUES[@]}"; do
  run_if_incomplete 64K "${p_value}"
done

for p_value in "${P_VALUES[@]}"; do
  run_if_incomplete 128K "${p_value}"
done

echo "[qwen3-threshold] all calibrations completed"
