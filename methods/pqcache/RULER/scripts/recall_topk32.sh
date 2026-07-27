#!/bin/bash
set -e

# Per-query-head counterpart of recall_topk.sh.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

cd "${SCRIPT_DIR}"
ROOT_DIR=recall_test_topk32 \
PQ_COMPRESSOR=no_drop_lb_32 \
bash "${SCRIPT_DIR}/recall_topk.sh" "$@"
