# cd to algorithm root (method_scripts/../../)
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."

for length in 4096; do
  for budget in 128 256 512 1024;do
    python ./evaluation/efficiency/pred.py \
      --model_path meta-llama/Llama-3.1-8B-Instruct \
      --dataset_path ../../benchmarks/myinput.txt \
      --output_dir ./efficiency/budget/latency-results \
      --model_name Llama-3.1-8B-Instruct \
      --quest \
      --token_budget ${budget} \
      --chunk_size 16 \
      --type LATENCY \
      --max_new_tokens 32 \
      --seqlen ${length}
  done
done


for length in 65536; do
  for budget in 128 384 1024 4096 16384;do
    python ./evaluation/efficiency/pred.py \
      --model_path meta-llama/Llama-3.1-8B-Instruct \
      --dataset_path ../../benchmarks/myinput.txt \
      --output_dir ./efficiency/budget/latency-results \
      --model_name Llama-3.1-8B-Instruct \
      --quest \
      --token_budget ${budget} \
      --chunk_size 16 \
      --type LATENCY \
      --max_new_tokens 32 \
      --seqlen ${length}
  done
done