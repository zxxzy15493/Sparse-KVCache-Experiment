#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/remain_short.py" \
  --tokenizer_name Qwen/Qwen2.5-7B-Instruct-1M \
  --dataset_name THUDM/LongBench-v2 \
  --split train \
  --min_len 65536 \
  --max_len 196608 \
  --output_path "$SCRIPT_DIR/filtered_longbench_v2_64k-192k.jsonl"
