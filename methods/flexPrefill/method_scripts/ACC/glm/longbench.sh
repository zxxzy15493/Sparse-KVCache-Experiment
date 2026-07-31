model_name="glm-4-9b-chat-1m"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
METHODS_ROOT=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)
FLEX_ROOT="$METHODS_ROOT/flexPrefill"
cd "$FLEX_ROOT"
export PYTHONPATH="$FLEX_ROOT:${PYTHONPATH:-}"

tasks="narrativeqa triviaqa 2wikimqa musique gov_report multi_news passage_count passage_retrieval_en lcc repobench-p"

for task in $tasks; do
    python "$FLEX_ROOT/experiments/benchmark/longbench/pred.py" \
  --model_path zai-org/glm-4-9b-chat-1m \
  --model_name $model_name \
  --task $task \
  --config_path "$FLEX_ROOT/experiments/benchmark/longbench/config" \
  --output_dir "$FLEX_ROOT/experiments/benchmark/longbench/pred"
done
