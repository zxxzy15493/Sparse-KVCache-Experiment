# cd to algorithm root (method_scripts/../../)
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
budget=1024
for length in 4096 8192 16384 32768 65536 131072 262144; do

  python ./evaluation/VRAM/mypred.py \
    --model_path Qwen/Qwen2.5-7B-Instruct-1M \
    --dataset_path ../../benchmarks/myinput.txt \
    --output_dir ./efficiency/latency-results \
    --model_name Qwen2.5-7B-Instruct-1M \
    --quest \
    --token_budget ${budget} \
    --chunk_size 16 \
    --type LATENCY \
    --max_new_tokens 2 \
    --seqlen ${length}
done


length=1024
for budget in 64 512;do
  python ./evaluation/VRAM/mypred.py \
    --model_path Qwen/Qwen2.5-7B-Instruct-1M \
    --dataset_path ../../benchmarks/myinput.txt \
    --output_dir ./efficiency/latency-results \
    --model_name Qwen2.5-7B-Instruct-1M \
    --quest \
    --token_budget ${budget} \
    --chunk_size 16 \
    --type LATENCY \
    --max_new_tokens 2 \
    --seqlen ${length}
done


length=1024
for budget in 64 512;do
  python ./evaluation/VRAM/mypred.py \
    --model_path Qwen/Qwen2.5-7B-Instruct-1M \
    --dataset_path ../../benchmarks/myinput.txt \
    --output_dir ./efficiency/latency-results \
    --model_name Qwen2.5-7B-Instruct-1M \
    --quest \
    --token_budget ${budget} \
    --chunk_size 16 \
    --type LATENCY \
    --max_new_tokens 4096 \
    --seqlen ${length}
done