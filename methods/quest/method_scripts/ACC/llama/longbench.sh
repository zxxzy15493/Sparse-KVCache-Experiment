SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
METHODS_ROOT=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)
QUEST_ROOT="$METHODS_ROOT/quest"

cd "$QUEST_ROOT/evaluation/LongBench"
export PYTHONPATH="$QUEST_ROOT:${PYTHONPATH:-}"



tasks="samsum narrativeqa qasper triviaqa 2wikimqa musique gov_report multi_news passage_count passage_retrieval_en lcc repobench-p"
budget=1024

model_path="meta-llama/Llama-3.1-8B-Instruct"
model_name="Llama-3.1-8B-Instruct"

output_dir="$QUEST_ROOT/evaluation/LongBench/pred"
config_path="$QUEST_ROOT/evaluation/LongBench/config"

for task in $tasks;do
    python -u pred.py \
    --model_name "$model_name" --task "$task" \
    --model_path "$model_path" --config_path "$config_path" \
    --quest --token_budget "$budget" --chunk_size 16 \
    --output_dir "$output_dir"
done
