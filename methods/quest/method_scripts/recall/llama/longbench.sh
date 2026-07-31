# cd to algorithm root (method_scripts/../../)
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.."

tasks="narrativeqa qasper"

config_path="./evaluation/recall/LongBench/config"

for task in $tasks;do
    for budget in 128 256 512 1024;do
        output_dir="./efficiency/budget_${budget}"
        save_path="./efficiency/recall_topkrate/longbench/${budget}/llama"
        python -u ./evaluation/recall/LongBench/pred.py \
        --model_name Llama-3.1-8B-Instruct --task $task \
        --model_path meta-llama/Llama-3.1-8B-Instruct --config_path $config_path \
        --quest --token_budget $budget --chunk_size 16\
        --dataset_path ../../benchmarks/Longbench_recall --output_dir $output_dir \
        --save_path ${save_path}
    done
done
