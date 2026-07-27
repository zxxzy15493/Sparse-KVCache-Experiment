# cd to algorithm root (buchong_scripts/../../)
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."

for length in 4096 8192 16384 32768 65536 131072; do


  python ./experiments/benchmark/TTFT-qaper/mypred.py \
    --model_path Qwen/Qwen2.5-7B-Instruct-1M \
    --dataset_path ../../benchmarks/myinput.txt \
    --output_dir ./efficiency/latency-results \
    --model_name Qwen2.5-7B-Instruct-1M \
    --cfg block_size=128,flex_prefill_min_budget=512,flex_prefill_gamma=0.9,flex_prefill_tau=0.1\
    --tag flexprefill \
    --attention  flex_prefill\
    --type TTFT \
    --max_new_tokens 32 \
    --seqlen ${length}

done