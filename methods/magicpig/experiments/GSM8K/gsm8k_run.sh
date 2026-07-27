if [ $# -ne 3 ]; then
    echo "Usage: $0 <model_name> $1 <K> $2 <L>"
    exit 1
fi

MODEL=${1}
K=${2}
L=${3}

fixed_output_length=0
NUM_SAMPLE=-1
RECALL=0

KLS="${K}_${L}"

PRED_DIR="./results/${MODEL}/${KLS}"
LOG_DIR="./log/${MODEL}/${KLS}"
mkdir -p ${PRED_DIR}
mkdir -p ${LOG_DIR}



CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=96 \
    torchrun --nproc_per_node=1 ./pred.py \
    --save_dir ${PRED_DIR} \
    --model_name ${MODEL} \
    --K ${K} \
    --L ${L} \
    --num_sample ${NUM_SAMPLE} \
    --recall ${RECALL} \
    --fixed_output_length ${fixed_output_length} \
    > ${LOG_DIR}/gsm8k_run_1000.log 2>&1


python -u ./evaluate.py \
    --input ${PRED_DIR}/gsm8k.jsonl \
    --output ${PRED_DIR}/gsm8k_eval.jsonl \
    --force

python -u ./tool/data_infos.py \
    --data-dir ${PRED_DIR} \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --task gsm8k \