cd "$(dirname -- "${BASH_SOURCE[0]}")"

python pred.py \
  --model_path Qwen/Qwen2.5-7B-Instruct-1M \
  --model_name Qwen2.5-7B-Instruct-1M \
  --max_context_len 204800 \
  --max_new_tokens 128 \
  --data_file ../../../benchmarks/longbenchv2/filtered_longbench_v2_64k-192k.jsonl \
  --save_dir res
