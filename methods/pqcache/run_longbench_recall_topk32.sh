#!/bin/bash
set -e

# Per-query-head topk recall sweep. All datasets, models, and budgets stay in
# run_longbench_recall_topk.sh so both topk variants remain directly comparable.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

cd "${SCRIPT_DIR}"
COMPRESSOR=no_drop_lb_32
export COMPRESSOR
TOPK_VARIANT=topk32
export TOPK_VARIANT
EXP_NAME=recall_test_topk32
export EXP_NAME
bash "${SCRIPT_DIR}/run_longbench_recall_topk.sh" "$@"
