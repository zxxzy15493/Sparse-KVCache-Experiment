#!/usr/bin/env bash
# 3 条基线 + 18 个方法的单卡效率/显存矩阵；运行时确认不支持多 batch 后才跳过后续 batch。
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BENCHMARKS_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export MAX_CPU_IN_USE="${MAX_CPU_IN_USE:-16}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  # "Qwen/Qwen2.5-7B-Instruct-1M"
)

METHODS=(
  # full_attention
  topk
  # h2o
  # keyformer
  snapkv
  streaming
  pyramidkv
  # headkv
  # adakv
  # cakekv
  # duo-attention
  # flexprefill
  # minference
  # pqcache
  # quest
  retroinfer
  # sparq
  # xattention
  # clusterkv
  # magicpig
)

# BATCH_SIZES=(1 4)
# INPUT_LENGTHS=(4096 16384 32768 65536 131072)
BATCH_SIZES=(1 4)
# INPUT_LENGTHS=(4096 8192 16384 32768 49152 65536 81920 98304 114688 131072)
INPUT_LENGTHS=(65536)
BUDGETS=(128 256 512 1024 2048 3072 4096 5120 6144 7168 8192)
EXPERIMENT="tradeoff_1112"
DATA_FILE="$BENCHMARKS_DIR/ruler/benchmark_root/{model}/synthetic/4096/data/niah_single_1/validation.jsonl"
OUTPUT_ROOT="$SCRIPT_DIR/output/${EXPERIMENT}_efficiency_vram"
STATE_DIR="$OUTPUT_ROOT"

exec python -u "$SCRIPT_DIR/tools/run_grid.py" \
  --models "${MODELS[@]}" \
  --methods "${METHODS[@]}" \
  --batch-sizes "${BATCH_SIZES[@]}" \
  --input-lengths "${INPUT_LENGTHS[@]}" \
  --data-file "$DATA_FILE" \
  --output-root "$OUTPUT_ROOT" \
  --state-dir "$STATE_DIR" \
  --experiment "$EXPERIMENT" \
  --task niah_single_1 \
  --budgets "${BUDGETS[@]}" \
  --output-length 32 \
  --warmup 1 \
  --measured-rounds 3 \
  --timeout-seconds 600 \
  --efficiency-backend
