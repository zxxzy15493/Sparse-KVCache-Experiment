model_name="Llama-3.1-8B-Instruct"


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
METHODS_ROOT=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)
FLEX_ROOT="$METHODS_ROOT/flexPrefill"
cd "$FLEX_ROOT"
export PYTHONPATH="$FLEX_ROOT:${PYTHONPATH:-}"

tasks="narrativeqa qasper trec lcc"
for p in 0.8 0.85 0.95 0.9;do
  for task in $tasks; do
      python "$FLEX_ROOT/experiments/benchmark/longbench/budget_pred.py" \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --model_name $model_name \
    --task $task \
    --config_path "$FLEX_ROOT/experiments/benchmark/longbench/config" \
    --output_dir "$FLEX_ROOT/experiments/benchmark/longbench/budget_pred" \
    --p $p
  done
done
