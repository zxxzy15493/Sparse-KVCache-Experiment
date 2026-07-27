#!/usr/bin/env bash
#
# Run myllama breakdown with CUDA 12.9 selected for FlashInfer on SM 12.x.
#
# This is a wrapper only: it does not modify conda activation files or source
# code. It forwards all positional args to run_myllama_breakdown.sh.
#
# Usage:
#   bash breakdown_test/run_myllama_breakdown_cuda129.sh [MODEL] [INPUT_LENS] [MAX_NEW_TOKENS] [BUDGET]
#
# Examples:
#   WARMUP_ROUNDS=1 MEASURE_ROUNDS=1 bash breakdown_test/run_myllama_breakdown_cuda129.sh meta-llama/Llama-3.1-8B-Instruct 65536 32 1024
#   CSV_OUT=breakdown_test/log/myllama_breakdown_cuda129_results.csv bash breakdown_test/run_myllama_breakdown_cuda129.sh

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

# Avoid user-site .pth files injecting packages from other conda envs.
export PYTHONNOUSERSITE=1

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export CSV_OUT="${CSV_OUT:-${SCRIPT_DIR}/log/myllama_breakdown_cuda129_results.csv}"

echo "============================================================"
echo " ClusterKV CUDA 12.9 FlashInfer wrapper"
echo "============================================================"
echo " CUDA_HOME      = ${CUDA_HOME}"
echo " PYTHONNOUSERSITE = ${PYTHONNOUSERSITE}"
echo " CSV_OUT        = ${CSV_OUT}"
echo "------------------------------------------------------------"
python - <<'PY'
import os
import sys
import torch
import flashinfer
import flashinfer.jit.cpp_ext as cpp_ext

print("python         =", sys.executable)
print("torch          =", torch.__version__, "cuda", torch.version.cuda)
print("flashinfer     =", getattr(flashinfer, "__version__", "unknown"), flashinfer.__file__)
print("flashinfer CUDA=", cpp_ext.get_cuda_path(), cpp_ext.get_cuda_version())

excluded_prefix = os.environ.get("EXCLUDED_PYTHON_PREFIX")
bad_paths = [p for p in sys.path if excluded_prefix and p.startswith(excluded_prefix)]
if bad_paths:
    raise SystemExit(f"unexpected kvcache paths in sys.path: {bad_paths}")

version_parts = tuple(int(part) for part in str(cpp_ext.get_cuda_version()).split(".")[:2])
if version_parts < (12, 9):
    raise SystemExit("FlashInfer does not see CUDA >= 12.9")
PY
echo "------------------------------------------------------------"

exec bash "${SCRIPT_DIR}/run_myllama_breakdown.sh" "$@"
