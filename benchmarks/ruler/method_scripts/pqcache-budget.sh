#!/usr/bin/env bash
set -euo pipefail

models=(llama-3.1-8b qwen-2.5-7b-1m)
tasks=(niah_single_3 vt cwe fwe qa_1)
lengths=(65536)
budgets=(128 384 1024 4096)

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec "$script_dir/run_experiment.sh" pqcache \
  --models "${models[@]}" \
  --tasks "${tasks[@]}" \
  --lengths "${lengths[@]}" \
  --budgets "${budgets[@]}"
