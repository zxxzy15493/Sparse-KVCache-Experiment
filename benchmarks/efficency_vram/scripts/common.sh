#!/usr/bin/env bash
# 六个矩阵入口共用的路径、默认模型和运行环境。

EFFICIENCY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BENCHMARKS_DIR=$(cd -- "$EFFICIENCY_DIR/.." && pwd)

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export MAX_CPU_IN_USE="${MAX_CPU_IN_USE:-16}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODELS=(llama-3.1-8b qwen-2.5-7b-1m)

BUDGETS=(1024)
THRESHOLDS=(0.90)
SPARSITIES=(0.50)
WARMUP=1
MEASURED_ROUNDS=3
TIMEOUT_SECONDS=1800
OUTPUT_LENGTH=32

run_benchmark() {
  local data_file="$BENCHMARKS_DIR/ruler/benchmark_root/{model}/synthetic/4096/data/niah_single_1/validation.jsonl"
  local output_root="${OUTPUT_ROOT:-$EFFICIENCY_DIR/output/${EXPERIMENT}_efficiency_vram}"

  exec python -u "$EFFICIENCY_DIR/pred.py" \
    --models "${MODELS[@]}" \
    --methods "${METHODS[@]}" \
    --batch-sizes "${BATCH_SIZES[@]}" \
    --input-lengths "${INPUT_LENGTHS[@]}" \
    --budgets "${BUDGETS[@]}" \
    --thresholds "${THRESHOLDS[@]}" \
    --sparsities "${SPARSITIES[@]}" \
    --data-file "$data_file" \
    --output-root "$output_root" \
    --experiment "$EXPERIMENT" \
    --task niah_single_1 \
    --output-length "$OUTPUT_LENGTH" \
    --warmup "$WARMUP" \
    --measured-rounds "$MEASURED_ROUNDS" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --efficiency-backend \
    "$@"
}
