#!/usr/bin/env bash
# Qwen3-32B，18 个方法，batch=1，全长度 overview。
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

MODELS=(qwen3-32b)

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

BATCH_SIZES=(1)
INPUT_LENGTHS=(4096 8192 16384 32768 40960 49152 65536)
WARMUP=1
MEASURED_ROUNDS=3
EXPERIMENT="qwen3"

run_benchmark "$@"
