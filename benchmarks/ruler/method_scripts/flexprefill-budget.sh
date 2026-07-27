#!/usr/bin/env bash
set -euo pipefail

models=(llama-3.1-8b qwen-2.5-7b-1m)
tasks=(niah_single_3 vt cwe fwe qa_1)
lengths=(65536)

thresholds=(0.8 0.85 0.9 0.95)

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec "$script_dir/run_experiment.sh" flexprefill \
  --models "${models[@]}" \
  --tasks "${tasks[@]}" \
  --lengths "${lengths[@]}" \
  --fixthreshold "${thresholds[@]}"
