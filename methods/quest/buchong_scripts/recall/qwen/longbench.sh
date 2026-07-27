# cd to algorithm root (buchong_scripts/../../)
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.."

tasks="narrativeqa qasper"


config_path="./evaluation/recall/LongBench/config"

for task in $tasks;do
    for budget in 128 256 512 1024;do
        output_dir="./efficiency/budget_${budget}"
        save_path="./efficiency/recall_topkrate/longbench/${budget}/qwen"
        python -u ./evaluation/recall/LongBench/pred.py \
        --model_name Qwen2.5-7B-Instruct-1M --task $task \
        --model_path Qwen/Qwen2.5-7B-Instruct-1M --config_path $config_path \
        --quest --token_budget $budget --chunk_size 16\
        --dataset_path ../../benchmarks/Longbench_recall --output_dir $output_dir \
        --save_path ${save_path}
    done
done
