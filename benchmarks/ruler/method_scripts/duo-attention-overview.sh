#!/usr/bin/env bash
set -euo pipefail

models=(llama-3.1-8b qwen-2.5-7b-1m)
tasks=(niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2)
lengths=(4096 8192 16384 32768 65536)
sparsities=(0.5)

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec "$script_dir/run_experiment.sh" duo-attention \
  --models "${models[@]}" \
  --tasks "${tasks[@]}" \
  --lengths "${lengths[@]}" \
  --sparsities "${sparsities[@]}"
