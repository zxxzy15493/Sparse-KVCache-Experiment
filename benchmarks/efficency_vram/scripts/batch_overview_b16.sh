#!/usr/bin/env bash
# 两个模型，19 个方法，batch=16，4K 至 64K，不预热、测一轮。
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
  flexprefill
  minference
  pqcache
  quest
  retroinfer
  sparq
  xattention
  clusterkv
  magicpig
)

BATCH_SIZES=(16)
INPUT_LENGTHS=(4096 8192 16384 32768 49152 65536)
WARMUP=0
MEASURED_ROUNDS=1
EXPERIMENT="batch_overview_b16"

run_benchmark "$@"
