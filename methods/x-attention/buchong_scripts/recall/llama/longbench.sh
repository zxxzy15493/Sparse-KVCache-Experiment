# cd to algorithm root (buchong_scripts/../../)
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.."

models="Llama-3.1-8B-Instruct"


methods="xattn"
tasks="qasper narrativeqa"





for p in 0.8 0.95 0.9 0.85;do
    for model in $models; do
        for task in $tasks; do
            for method in $methods; do
                save_path="./efficiency/attn_score/longbench/llama/${p}"
                python -u ./eval/recall/LongBench/budget_llama_pred.py \
                --model_path meta-llama/Llama-3.1-8B-Instruct \
                --longbench_dir ../../benchmarks/Longbench_recall \
                --model_name $model --task $task --method $method  --budget $p --save_path ${save_path}
            done
        done
    done
done