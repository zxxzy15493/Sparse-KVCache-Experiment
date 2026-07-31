#!/bin/bash
LongBench=(
    "narrativeqa"
    "qasper"
    "2wikimqa"
    "musique"
    "gov_report"
    "multi_news"
    "triviaqa"
    "samsum"
    "passage_count"
    "passage_retrieval_en"
    "lcc"
    "repobench-p"
)

MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
RECENT_SIZE=32
TAU_INIT=1.0
TAU_DELTA=0.01
BUDGETS=(128 256 512 1024)


for BUDGET in "${BUDGETS[@]}"; do
    KEY_SIZE=$((BUDGET - RECENT_SIZE))

    for DATASET in "${LongBench[@]}"; do

        echo "Budget: $BUDGET | Dataset: $DATASET"

        python keyformer.py \
            --model_name_or_path "$MODEL_PATH" \
            --dataset_name "$DATASET" \
            --keyformer \
            --key_size "$KEY_SIZE" \
            --recent_size "$RECENT_SIZE" \
            --tau_init "$TAU_INIT" \
            --tau_delta "$TAU_DELTA" \
            --budget "$BUDGET"

        if [ $? -eq 0 ]; then
            echo -e "\n$DATASET (budget=$BUDGET) success"
        else
            echo -e "\n$DATASET (budget=$BUDGET) fail!"
        fi
    done
done
