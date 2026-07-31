# cd to algorithm root (method_scripts/../../)
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."

for p in 0.8 0.85 0.9 0.95;do
  for task in narrativeqa qasper;do
    python ./experiments/benchmark/recall/longbench/budget_pred.py \
      --model_path meta-llama/Llama-3.1-8B-Instruct \
      --dataset_path ../../benchmarks/Longbench_recall \
      --config_path ./experiments/benchmark/recall/longbench/config \
      --output_dir ./efficiency/attn_rate-results \
      --model_name Llama-3.1-8B-Instruct \
      --task ${task} \
      --p ${p} \
      --type recall
  done
done



for p in 0.8 0.85 0.9 0.95;do
  for task in narrativeqa qasper;do
    python ./experiments/benchmark/recall/longbench/budget_pred.py \
      --model_path Qwen/Qwen2.5-7B-Instruct \
      --dataset_path ../../benchmarks/Longbench_recall \
      --config_path ./experiments/benchmark/recall/longbench/config \
      --output_dir ./efficiency/attn_rate-results \
      --model_name Qwen2.5-7B-Instruct \
      --task ${task} \
      --p ${p} \
      --type recall
  done
done