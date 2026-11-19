#!/usr/bin/env bash
# 两个模型，16 个方法（去除 minference/xattention/flexprefill），batch=4，64K 参数扫描。
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

METHODS=(
  full_attention
  h2o
  keyformer
  snapkv
  streaming
  pyramidkv
  headkv
  adakv
  cakekv
  duo-attention
  pqcache
  quest
  retroinfer
  sparq
  clusterkv
  magicpig
)

BATCH_SIZES=(4)
INPUT_LENGTHS=(65536)
BUDGETS=(128 256 512 1024 2048 3072 4096 5120 6144 7168 8192)
THRESHOLDS=(0.80 0.825 0.85 0.875 0.90 0.91 0.92 0.93 0.94 0.95 0.96)
SPARSITIES=(0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40)
WARMUP=1
MEASURED_ROUNDS=1
EXPERIMENT="trade_off_b4"

run_benchmark "$@"
