#!/usr/bin/env bash
#
# Run the MyLlama/MyQwen efficiency overview with CUDA 12.9 selected for
# FlashInfer on SM 12.x. The overview sweep itself remains in
# run_my_latency_overview.sh.
#
# Usage:
#   bash efficiency/run_my_latency_overview_cuda129.sh
#
# Example:
#   WARMUP_ROUNDS=1 MEASURE_ROUNDS=1 bash efficiency/run_my_latency_overview_cuda129.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CUDA_129_HOME="${CUDA_129_HOME:-/usr/local/cuda-12.9}"

if [[ ! -x "${CUDA_129_HOME}/bin/nvcc" ]]; then
    echo "CUDA 12.9 nvcc not found: ${CUDA_129_HOME}/bin/nvcc" >&2
    exit 1
fi

export CUDA_HOME="${CUDA_129_HOME}"
export CUDA_PATH="${CUDA_129_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export CSV_FILE="${CSV_FILE:-${SCRIPT_DIR}/log_latency/myllama_latency_overview_cuda129.csv}"

echo "============================================================"
echo " ClusterKV CUDA 12.9 efficiency wrapper"
echo "============================================================"
echo " CUDA_HOME        = ${CUDA_HOME}"
echo " PYTHONNOUSERSITE = ${PYTHONNOUSERSITE}"
echo " CSV_FILE         = ${CSV_FILE}"
echo "------------------------------------------------------------"
python - <<'PY'
import os
import sys

import flashinfer
import flashinfer.jit.cpp_ext as cpp_ext
import torch

print("python         =", sys.executable)
print("torch          =", torch.__version__, "cuda", torch.version.cuda)
print("flashinfer     =", getattr(flashinfer, "__version__", "unknown"), flashinfer.__file__)
print("flashinfer CUDA=", cpp_ext.get_cuda_path(), cpp_ext.get_cuda_version())

excluded_prefix = os.environ.get("EXCLUDED_PYTHON_PREFIX")
bad_paths = [path for path in sys.path if excluded_prefix and path.startswith(excluded_prefix)]
if bad_paths:
    raise SystemExit(f"unexpected kvcache paths in sys.path: {bad_paths}")

version_parts = tuple(int(part) for part in str(cpp_ext.get_cuda_version()).split(".")[:2])
if version_parts < (12, 9):
    raise SystemExit("FlashInfer does not see CUDA >= 12.9")
PY
echo "------------------------------------------------------------"

exec bash "${SCRIPT_DIR}/run_my_latency_overview.sh" "$@"
