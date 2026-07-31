# cd to algorithm root (method_scripts/../../)
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.."

model="Llama-3.1-8B-Instruct"


KS="128 256 512 1024"
LOCAL_KS="32"
RANKS="16"
NAME="ann"
SCORE="sparse_q"
REALLOCATE_TO_MEAN_VALUE=True

tasks="qasper narrativeqa"

for task in $tasks;do
    for K in $KS; do
        for LOCAL_K in $LOCAL_KS; do
            for RANK in $RANKS; do
    save_path="./efficiency/recall_attnscores/longbench/llama/${K}"
    python ./experiments/recall/longbench/budget_pred.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --config_path ./experiments/recall/longbench/config \
    --dataset_path ../../benchmarks/Longbench_recall \
    --model_name $model --task $task \
    --name ${NAME} --k ${K}\
    --local_k ${LOCAL_K}  --reallocate_to_mean_value ${REALLOCATE_TO_MEAN_VALUE}\
    --score ${SCORE} --rank ${RANK} \
    --recall_save_path ${save_path}            
            done
        done
    done
done