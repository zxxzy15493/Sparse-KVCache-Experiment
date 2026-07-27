# cd to algorithm root (buchong_scripts/../../)
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."

# for p in 0.8 0.85 0.9 0.95;do

#     python ./experiments/benchmark/attn_rate_ruler.py \
#         --model meta-llama/Llama-3.1-8B-Instruct \
#         --chat \
#         --task ruler \
#         --attention flex_prefill \
#         --tag flex_prefill \
#         --p ${p}
# done


for p in 0.8 0.85 0.9 0.95;do
    python ./experiments/benchmark/attn_rate_ruler.py \
        --model Qwen/Qwen2.5-7B-Instruct-1M \
        --chat \
        --task ruler \
        --attention flex_prefill \
        --tag flex_prefill \
        --p ${p}
done