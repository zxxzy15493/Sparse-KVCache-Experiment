# cd to algorithm root (buchong_scripts/../../)
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."

for length in 4096 8192 16384 32768 65536 131072; do

  python ./experiments/TTFT-qaper/mypred.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path ../../benchmarks/myinput.txt \
    --output_dir ./efficiency/latency-results \
    --model_name Llama-3.1-8B-Instruct \
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