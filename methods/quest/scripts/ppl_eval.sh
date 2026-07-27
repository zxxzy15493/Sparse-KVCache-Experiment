cd "$(dirname -- "${BASH_SOURCE[0]}")/../evaluation/pg19"


MODELPATH="lmsys/longchat-7b-v1.5-32k"
OUTPUT_DIR=results/ppl_eval/longchat
mkdir -p $OUTPUT_DIR

device=0
budget=4096

python -u ppl_eval.py \
    --model_name_or_path $MODELPATH \
    --output_dir $OUTPUT_DIR \
    --num_eval_tokens 30000 \
    --quest --token_budget $budget --chunk_size 16 
