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

Qwen="Qwen2.5-7B-Instruct"
Llama="Llama-3.1-8B-Instruct"
DATASET="LongBench"
BUDGETS=(128 256 512 1024)


for BUDGET in "${BUDGETS[@]}"; do
    COMPRESS_ARGS_PATH="./config/ablation_c${BUDGET}_w32_k7_maxpool.json"

    for DATASET_NAME in "${LongBench[@]}"; do

        echo "Budget: $BUDGET | Dataset: $DATASET_NAME"

        python pred_snap.py \
            --model "$Llama" \
            --dataset "$DATASET" \
            --dataset_name "$DATASET_NAME" \
            --compress_args_path "$COMPRESS_ARGS_PATH" \
            --budget "$BUDGET"

        if [ $? -eq 0 ]; then
            echo -e "\n$DATASET_NAME (budget=$BUDGET) success"
        else
            echo -e "\n$DATASET_NAME (budget=$BUDGET) fail!"
        fi
    done
done
