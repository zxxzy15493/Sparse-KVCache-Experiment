

# cd to algorithm root (method_scripts/../../)
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."

for length in 4096 8192 16384 32768 65536 131072; do
  python ./evaluation/efficiency/pred.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path ../../benchmarks/myinput.txt \
    --output_dir ./efficiency/latency-results \
    --model_name Llama-3.1-8B-Instruct \
    --quest \
    --token_budget 1024 \
    --chunk_size 16 \
    --type LATENCY \
    --max_new_tokens 32 \
    --seqlen ${length}
done
