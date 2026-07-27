if [ $# -ne 6 ]; then
    echo "Usage: $0 <model_name> $1 <K> $2 <L> $3 <except_budget> $4 <Benchmark> $5 <task>"
    exit 1
fi

MODEL=${1}
K=${2}
L=${3}
except_budget=${4}
BENCHMARK=${5}
TASK=${6}

fixed_output_length=0
NUM_SAMPLE=-1
FIX_BUDGET=0
RECALL=1

KLS="${K}_${L}"

DATA_DIR="../../../../benchmarks/longbench"
PRED_DIR="./results/RecallOverview/${MODEL}/${BENCHMARK}/${except_budget}/${KLS}"
LOG_DIR="./log/RecallOverview/${MODEL}/${BENCHMARK}/${except_budget}/${KLS}"
mkdir -p ${PRED_DIR}
mkdir -p ${LOG_DIR}



CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=96 \
    torchrun --nproc_per_node=1 ./pred.py \
    --save_dir ${PRED_DIR} \
    --model_name ${MODEL} \
    --K ${K} \
    --L ${L} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    --num_sample ${NUM_SAMPLE} \
    --recall ${RECALL} \
    --fixed_output_length ${fixed_output_length} \
    --fixed_budget ${FIX_BUDGET} \
    > ${LOG_DIR}/${TASK}.log 2>&1


python -u evaluate.py \
    --data_dir ${PRED_DIR} \
    --task ${TASK} \