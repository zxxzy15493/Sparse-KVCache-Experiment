#!/bin/bash
set -e

# Sweep/analysis driver for per-query-head topk recall.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

cd "${SCRIPT_DIR}"
RECALL_SCRIPT=recall_topk32.sh \
RECALL_SUFFIX=no_drop_lb_32 \
bash recall_all_topk.sh "$@"
