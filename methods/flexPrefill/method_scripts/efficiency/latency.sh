

# cd to algorithm root (method_scripts/../../)
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."

for length in 4096 8192 16384 32768 65536 131072; do

  python ./experiments/benchmark/efficiency/pred.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path ../../benchmarks/myinput.txt \
    --output_dir ./efficiency/latency-results \
    --model_name Llama-3.1-8B-Instruct \
    --cfg block_size=128,flex_prefill_min_budget=512,flex_prefill_gamma=0.9,flex_prefill_tau=0.1\
    --tag flexprefill \
    --attention  flex_prefill\
    --type LAtency \
    --max_new_tokens 32 \
    --seqlen ${length}

done