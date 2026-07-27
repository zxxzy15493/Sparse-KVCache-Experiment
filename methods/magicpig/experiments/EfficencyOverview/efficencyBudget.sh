if [ $# -ne 6 ]; then
    echo "Usage: $0 <model_name> $1 <K> $2 <L> $3 <input_max_token> $4 <fixed_output_length> ${5} <except_budget>"
    exit 1
fi

MODEL=${1}
K=${2}
L=${3}
INPUT_MAX_TOKEN=${4}
fixed_output_length=${5}
except_budget=${6}

NUM_SAMPLE=-1
FIX_BUDGET=0
RECALL=0

if [ "$FIX_BUDGET" -gt 0 ]; then
    MODE="Fixed"
elif [ "$FIX_BUDGET" -eq 0 ]; then
    MODE="Ratio"
else
    echo "Error: FIX_BUDGET must be 0 or greater than 0"
    exit 1
fi

KLS="${K}-${L}"

DATA_DIR="../../../../benchmarks/longbench"
PRED_DIR="./results/efficencyBudget/${MODEL}/${MODE}/${INPUT_MAX_TOKEN}_${fixed_output_length}/${except_budget}"
LOG_DIR="./log/efficencyBudget/${MODEL}/${MODE}/${INPUT_MAX_TOKEN}_${fixed_output_length}/"
mkdir -p ${PRED_DIR}
mkdir -p ${LOG_DIR}



CUDA_VISIBLE_DEVICES=0 \
        OMP_NUM_THREADS=96 \
        torchrun --nproc_per_node=1 ./efficency.py \
        --data_dir ${DATA_DIR} \
        --save_dir ${PRED_DIR} \
        --model_name ${MODEL} \
        --K ${K} \
        --L ${L} \
        --num_sample ${NUM_SAMPLE} \
        --recall ${RECALL} \
        --input_max_token ${INPUT_MAX_TOKEN} \
        --fixed_output_length ${fixed_output_length} \
        --fixed_budget ${FIX_BUDGET}  >> ${LOG_DIR}/${except_budget}_${KLS}.log 2>&1
