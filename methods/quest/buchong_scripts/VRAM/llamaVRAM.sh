# cd to algorithm root (buchong_scripts/../../)
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."

for length in 4096 8192 16384 32768 65536 131072 262144; do

  python ./evaluation/VRAM/mypred.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path ../../benchmarks/myinput.txt \
    --output_dir ./efficiency/latency-results \
    --model_name Llama-3.1-8B-Instruct \
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
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path ../../benchmarks/myinput.txt \
    --output_dir ./efficiency/latency-results/022 \
    --model_name Llama-3.1-8B-Instruct \
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
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path ../../benchmarks/myinput.txt \
    --output_dir ./efficiency/latency-results/022 \
    --model_name Llama-3.1-8B-Instruct \
    --quest \
    --token_budget ${budget} \
    --chunk_size 16 \
    --type LATENCY \
    --max_new_tokens 4096 \
    --seqlen ${length}
done