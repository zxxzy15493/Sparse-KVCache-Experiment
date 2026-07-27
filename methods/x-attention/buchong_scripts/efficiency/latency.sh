# cd to algorithm root (buchong_scripts/../../)
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."

for length in 4096 8192 16384 32768 65536 131072 ;do

    python ./eval/TTFT-qasper/ttft_pred.py \
      --model_path meta-llama/Llama-3.1-8B-Instruct \
      --dataset_path ../../benchmarks/myinput.txt \
      --output_dir ./efficiency/latency-results \
      --model_name Llama-3.1-8B-Instruct \
      --method "xattn" --stride 16 \
      --type LATENCY \
      --max_new_tokens 32 \
      --seqlen ${length}
done


