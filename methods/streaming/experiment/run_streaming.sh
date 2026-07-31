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


Llama="Llama-3.1-8B-Instruct"
Qwen="Qwen2.5-7B-Instruct"
Qwen_1M="Qwen2.5-7B-Instruct-1M"
START_SIZE=16
RECENT_SIZE=1008


for DATASET in "${LongBench[@]}"; do

    echo "$DATASET"

    python streaming.py \
        --model_name_or_path "$Llama" \
        --dataset_name "$DATASET" \
        --start_size "$START_SIZE" \
        --recent_size "$RECENT_SIZE" \
        --enable_streaming \

    if [ $? -eq 0 ]; then
        echo -e "\n$DATASET success"
    else
        echo -e "\n$DATASET fail!"
    fi
done

