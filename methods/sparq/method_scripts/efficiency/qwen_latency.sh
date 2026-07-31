# cd to algorithm root (method_scripts/../../)
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."

for length in 4096 8192 16384 32768 65536 131072; do


  python ./experiments/efficiency/pred.py \
    --model_path Qwen/Qwen2.5-7B-Instruct-1M \
    --dataset_path ../../benchmarks/myinput.txt \
    --output_dir ./efficiency/latency-results/qwen_remain \
    --model_name Qwen2.5-7B-Instruct-1M \
    --name "ann" \
    --k 1024 \
    --local_k 32 \
    --score "sparse_q" \
    --rank 16 \
    --reallocate_to_mean_value "TRUE" \
    --type LATENCY \
    --max_new_tokens 32 \
    --seqlen ${length}
done