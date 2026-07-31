# cd to algorithm root (method_scripts/../../)
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."

for length in 4096 65536;do

  for p in 0.8 0.85 0.9 0.95;do

    python ./experiments/benchmark/efficiency/pred.py \
      --model_path meta-llama/Llama-3.1-8B-Instruct \
      --dataset_path ../../benchmarks/myinput.txt \
      --output_dir ./efficiency/budget/latency-results \
      --model_name Llama-3.1-8B-Instruct \
      --tag flexprefill \
      --attention  flex_prefill\
      --type LAtency \
      --max_new_tokens 32 \
      --seqlen ${length} \
      --p ${p}

  done
done
