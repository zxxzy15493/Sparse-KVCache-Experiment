#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SPARQ_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

cd "$SPARQ_ROOT/experiments/longbench"
export PYTHONPATH="$SPARQ_ROOT:${PYTHONPATH:-}"

model="Qwen2.5-7B-Instruct"
model_path="Qwen/Qwen2.5-7B-Instruct"
config_path="$SPARQ_ROOT/experiments/longbench/config"
output_dir="$SPARQ_ROOT/experiments/longbench/pred"

tasks="samsum narrativeqa qasper triviaqa 2wikimqa musique gov_report multi_news passage_count passage_retrieval_en lcc repobench-p"
KS="1024"
LOCAL_KS="32"
RANKS="16"
NAME="ann"
SCORE="sparse_q"
REALLOCATE_TO_MEAN_VALUE=True

for task in $tasks; do
    for K in $KS; do
        for LOCAL_K in $LOCAL_KS; do
            for RANK in $RANKS; do
                python -u pred.py \
                    --model_path "$model_path" \
                    --config_path "$config_path" \
                    --output_dir "$output_dir" \
                    --model_name "$model" \
                    --task "$task" \
                    --name "$NAME" \
                    --k "$K" \
                    --local_k "$LOCAL_K" \
                    --reallocate_to_mean_value "$REALLOCATE_TO_MEAN_VALUE" \
                    --score "$SCORE" \
                    --rank "$RANK"
            done
        done
    done
done
