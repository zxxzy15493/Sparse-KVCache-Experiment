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
DATA_ROOT="LongBench"
KEY_SIZE=992
RECENT_SIZE=32
TAU_INIT=1.0
TAU_DELTA=0.01


for DATASET in "${LongBench[@]}"; do

    echo "$DATASET"

    python keyformer.py \
        --model_name_or_path "$MODEL_PATH" \
        --data_root "$DATA_ROOT" \
        --dataset_name "$DATASET" \
        --keyformer \
        --key_size "$KEY_SIZE" \
        --recent_size "$RECENT_SIZE" \
        --tau_init "$TAU_INIT" \
        --tau_delta "$TAU_DELTA"

    if [ $? -eq 0 ]; then
        echo -e "\n$DATASET success"
    else
        echo -e "\n$DATASET fail!"
    fi
done
