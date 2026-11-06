#!/usr/bin/env bash
# 单卡运行 Llama 3.1 与 Qwen2.5-7B-1M 的效率/显存组合测试。
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BENCHMARKS_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)


export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MAX_CPU_IN_USE="${MAX_CPU_IN_USE:-16}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Hugging Face 仓库 ID；不填写本机模型权重路径。
MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct-1M"
)

# 取自 qwen3_0.6b_smoke_gpu1_b2/report.md 的 passed 方法，并额外包含 full。
METHODS=(
  full_attention
  h2o
  keyformer
  snapkv
  streaming
  # Quest
  sparq
  xattention
  flexprefill
  retroinfer
  headkv
  adakv
  cakekv
  pyramidkv
)

BATCH_SIZES=(1 2 4 8 16)
INPUT_LENGTHS=(4096 8192 16384 32768 49152 65536 81920 98304 114688 131072)

EXPERIMENT="llama31_qwen25_7b1m_32tok"
DATA_FILE="$BENCHMARKS_DIR/ruler/benchmark_root/qwen3-32b/synthetic/4096/data/niah_single_1/validation.jsonl"
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
  --budget 1024 \
  --output-length 32 \
  --warmup 1 \
  --measured-rounds 3 \
  --timeout-seconds 1200 \
  --device cuda:0
