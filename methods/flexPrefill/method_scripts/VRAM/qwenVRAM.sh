# cd to algorithm root (method_scripts/../../)
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."

for length in 4096 8192 16384 32768 65536 131072 262144; do

  for p in 0.9;do
    python ./experiments/benchmark/VRAM/mypred.py \
      --model_path Qwen/Qwen2.5-7B-Instruct-1M \
      --dataset_path ../../benchmarks/myinput.txt \
      --output_dir ./efficiency/budget/latency-results \
      --model_name Qwen2.5-7B-Instruct-1M \
      --tag flexprefill \
      --attention  flex_prefill\
      --type LAtency \
      --max_new_tokens 2 \
      --seqlen ${length} \
      --p ${p}
  done
done

for length in 1024; do
  for p in 0.9;do
    python ./experiments/benchmark/VRAM/mypred.py \
      --model_path Qwen/Qwen2.5-7B-Instruct-1M \
      --dataset_path ../../benchmarks/myinput.txt \
      --output_dir ./efficiency/budget/latency-results \
      --model_name Qwen2.5-7B-Instruct-1M \
      --tag flexprefill \
      --attention  flex_prefill\
      --type LAtency \
      --max_new_tokens 4096 \
      --seqlen ${length} \
      --p ${p}
  done
done

for length in 1024; do
  for p in 0.9;do
    python ./experiments/benchmark/VRAM/mypred.py \
      --model_path Qwen/Qwen2.5-7B-Instruct-1M \
      --dataset_path ../../benchmarks/myinput.txt \
      --output_dir ./efficiency/budget/latency-results \
      --model_name Qwen2.5-7B-Instruct-1M \
      --tag flexprefill \
      --attention  flex_prefill\
      --type LAtency \
      --max_new_tokens 2 \
      --seqlen ${length} \
      --p ${p}
  done
done
