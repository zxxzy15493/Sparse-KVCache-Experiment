#!/usr/bin/env bash
# 两个模型，19 个方法，64K，batch 从 1 到 16。
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

BATCH_SIZES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
INPUT_LENGTHS=(65536)
WARMUP=1
MEASURED_ROUNDS=1
EXPERIMENT="mul_batch"

run_benchmark "$@"
