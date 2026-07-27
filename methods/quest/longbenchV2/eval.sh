cd "$(dirname -- "${BASH_SOURCE[0]}")"

python eval.py \
  --input ../longbenchV2res/Qwen2.5-7B-Instruct-1M_cot_50k.jsonl \
  --output Qwen2.5_64k-192k.jsonl



  
