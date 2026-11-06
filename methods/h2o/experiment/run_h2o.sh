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



Qwen="Qwen/Qwen2.5-7B-Instruct"
Qwen3="Qwen/Qwen3-32B"
HEAVY_HITTER_SIZE=992
RECENT_SIZE=32


for DATASET in "${LongBench[@]}"; do

    echo "$DATASET"

    python h2o.py \
        --model_name_or_path "$Qwen" \
        --dataset_name "$DATASET" \
        --enable_h2o_cache \
        --heavy_hitter_size "$HEAVY_HITTER_SIZE" \
        --recent_size "$RECENT_SIZE"


    if [ $? -eq 0 ]; then
        echo -e "\n$DATASET success"
    else
        echo -e "\n$DATASET fail！"
    fi
done

