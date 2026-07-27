# Evaluate Qwen XAttention results
cd "$(dirname -- "${BASH_SOURCE[0]}")"

python eval.py \
  --input ../longbenchV2res/011011Qwen2.5-7B-Instruct-1M_xattn_s16_cot.jsonl \
  --output ../longbenchV2res/Qwen2.5-7B-Instruct-1M_xattn_s16_eval.jsonl

# Evaluate Llama XAttention results (uncomment to use)
# python eval.py \
#   --input ../longbenchV2res/Llama-3.1-8B-Instruct_xattn_s16.jsonl \
#   --output ../longbenchV2res/Llama-3.1-8B-Instruct_xattn_s16_eval.jsonl
