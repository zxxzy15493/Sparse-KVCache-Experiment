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
COMPRESS_ARGS_PATH="./config/ablation_c1024_w32_k7_maxpool.json"


for DATASET_NAME in "${LongBench[@]}"; do

    echo "$DATASET_NAME"

    python pred_snap.py \
        --model "$Llama" \
        --dataset "$DATASET" \
        --dataset_name "$DATASET_NAME" \
        --compress_args_path "$COMPRESS_ARGS_PATH" \

    if [ $? -eq 0 ]; then
        echo -e "\n$DATASET success"
    else
        echo -e "\n$DATASET fail!"
    fi
done