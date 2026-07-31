# cd to algorithm root (method_scripts/../../)
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."

budget=1024

for length in 4096 8192 16384 32768 65536 131072 262144; do
  python ./experiments/VRAM/pred.py \
    --model_path Qwen/Qwen2.5-7B-Instruct-1M \
    --dataset_path ../../benchmarks/myinput.txt \
    --output_dir ./efficiency/latency-results \
    --model_name Qwen2.5-7B-Instruct-1M \
    --name "ann" \
    --k ${budget} \
    --local_k 32 \
    --score "sparse_q" \
    --rank 16 \
    --reallocate_to_mean_value "TRUE" \
    --type LATENCY \
    --max_new_tokens 2 \
    --seqlen ${length}
done


length=1024

for budget in 64 512;do
  python ./experiments/VRAM/pred.py \
    --model_path Qwen/Qwen2.5-7B-Instruct-1M \
    --dataset_path ../../benchmarks/myinput.txt \
    --output_dir ./efficiency/latency-results \
    --model_name Qwen2.5-7B-Instruct-1M \
    --name "ann" \
    --k ${budget} \
    --local_k 4 \
    --score "sparse_q" \
    --rank 16 \
    --reallocate_to_mean_value "TRUE" \
    --type LATENCY \
    --max_new_tokens 2 \
    --seqlen ${length}
done

for budget in 64 512;do
  python ./experiments/VRAM/pred.py \
    --model_path Qwen/Qwen2.5-7B-Instruct-1M \
    --dataset_path ../../benchmarks/myinput.txt \
    --output_dir ./efficiency/latency-results \
    --model_name Qwen2.5-7B-Instruct-1M \
    --name "ann" \
    --k ${budget} \
    --local_k 32 \
    --score "sparse_q" \
    --rank 16 \
    --reallocate_to_mean_value "TRUE" \
    --type LATENCY \
    --max_new_tokens 4096 \
    --seqlen ${length}
done
