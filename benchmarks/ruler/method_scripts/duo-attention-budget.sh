#!/usr/bin/env bash
set -euo pipefail

models=(llama-3.1-8b qwen-2.5-7b-1m)
tasks=(niah_single_3 vt cwe fwe qa_1)
lengths=(65536)
sparsities=(0.6 0.7 0.8 0.9)

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec "$script_dir/run_experiment.sh" duo-attention \
  --models "${models[@]}" \
  --tasks "${tasks[@]}" \
  --lengths "${lengths[@]}" \
  --fixthreshold "${sparsities[@]}"
