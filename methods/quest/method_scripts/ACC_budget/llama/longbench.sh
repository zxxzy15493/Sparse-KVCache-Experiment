SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
METHODS_ROOT=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)
QUEST_ROOT="$METHODS_ROOT/quest"

cd "$QUEST_ROOT/evaluation/LongBench"
export PYTHONPATH="$QUEST_ROOT:${PYTHONPATH:-}"

tasks="narrativeqa qasper trec lcc"

model_path="meta-llama/Llama-3.1-8B-Instruct"
model_name="Llama-3.1-8B-Instruct"

output_dir="$QUEST_ROOT/evaluation/LongBench/pred"
config_path="$QUEST_ROOT/evaluation/LongBench/config"


for budget in 128 256 512 1024; do
    for task in $tasks;do
        python -u pred.py \
        --model_name "$model_name" --task "$task" \
        --model_path "$model_path" --config_path "$config_path" \
        --quest --token_budget "$budget" --chunk_size 16 \
        --output_dir "$output_dir"
    done
done
