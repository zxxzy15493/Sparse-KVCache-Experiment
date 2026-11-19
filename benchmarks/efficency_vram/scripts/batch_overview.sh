#!/usr/bin/env bash
# 两个模型，19 个方法，batch=1/4，全长度 overview。
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

BATCH_SIZES=(1 4)
INPUT_LENGTHS=(4096 8192 16384 32768 49152 65536 81920 98304 114688 131072)
WARMUP=1
MEASURED_ROUNDS=3
EXPERIMENT="batch_overview"

run_benchmark "$@"
