#!/bin/bash

# ==========================================
#   Group A - LongBench: qasper/narrativeqa x llama3.1-8b/qwen2.5-7b x 128/256/512/1024
#   Group B - RULER:     fwe/vt/niah_single_3 x llama3.1-8b/qwen2.5-7b-1m x 128/384/1024/4096
# ==========================================
export CUDA_VISIBLE_DEVICES=0
PYTHON=${PYTHON:-python}

# ==========================================
# ==========================================
WINDOW=32

declare -A MODEL_PATHS
MODEL_PATHS[llama3.1-8b-128k]="meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATHS[qwen2.5-7b-instruct]="Qwen/Qwen2.5-7B-Instruct"
MODEL_PATHS[qwen2.5-7b-instruct-1m]="Qwen/Qwen2.5-7B-Instruct-1M"

# ==========================================
# Group A: LongBench recall datasets
# ==========================================
echo "=============================================="
echo " GROUP A: LongBench (qasper, narrativeqa)"
echo " Models:  llama3.1-8b-128k, qwen2.5-7b-instruct"
echo " Budgets: 128 256 512 1024"
echo "=============================================="
echo ""

LB_DATA="../../../../benchmarks/Longbench_recall"
LB_DATASETS=("qasper" "narrativeqa")
LB_BUDGETS=(128 256 512 1024)
LB_MODELS=(llama3.1-8b-128k qwen2.5-7b-instruct)

for model_key in "${LB_MODELS[@]}"; do
    model_path="${MODEL_PATHS[$model_key]}"
    for budget in "${LB_BUDGETS[@]}"; do
        for ds in "${LB_DATASETS[@]}"; do
            config_name="${model_key}_c${budget}_w${WINDOW}.json"
            echo "============================================================"
            echo "  [A] Model: $model_key | Dataset: $ds | Budget: $budget"
            echo "============================================================"

            python recall.py \
                --model "$model_path" \
                --dataset "$LB_DATA" \
                --dataset_name "$ds" \
                --compress_args_path "$config_name"

            if [ $? -eq 0 ]; then
                echo "  [OK] $model_key / $ds / budget=$budget done."
            else
                echo "  [FAIL] $model_key / $ds / budget=$budget failed!"
            fi
            echo ""
        done
    done
done

# ==========================================
# Group B: RULER synthetic datasets
# ==========================================
echo "=============================================="
echo " GROUP B: RULER (fwe, vt, niah_single_3)"
echo " Models:  llama3.1-8b-128k, qwen2.5-7b-instruct-1m"
echo " Budgets: 128 384 1024 4096"
echo "=============================================="
echo ""

RULER_ROOT="../../../../benchmarks/Ruler_recall"
declare -A RULER_MODEL_DIRS
RULER_MODEL_DIRS[llama3.1-8b-128k]="llama-3.1-8b"
RULER_MODEL_DIRS[qwen2.5-7b-instruct-1m]="qwen-2.5-7b-1m"

RULER_TASKS=("fwe" "vt" "niah_single_3")
RULER_BUDGETS=(128 384 1024 4096)
RULER_MODELS=(llama3.1-8b-128k qwen2.5-7b-instruct-1m)
SEQ_LEN=65536

for model_key in "${RULER_MODELS[@]}"; do
    model_path="${MODEL_PATHS[$model_key]}"
    data_dir="${RULER_MODEL_DIRS[$model_key]}"
    dataset_path="${RULER_ROOT}/${data_dir}/synthetic/${SEQ_LEN}/data"
    for budget in "${RULER_BUDGETS[@]}"; do
        for task in "${RULER_TASKS[@]}"; do
            config_name="${model_key}_c${budget}_w${WINDOW}.json"
            echo "============================================================"
            echo "  [B] Model: $model_key | Task: $task | Budget: $budget"
            echo "============================================================"

            python recall.py \
                --model "$model_path" \
                --dataset "$dataset_path" \
                --task "$task" \
                --compress_args_path "$config_name"

            if [ $? -eq 0 ]; then
                echo "  [OK] $model_key / $task / budget=$budget done."
            else
                echo "  [FAIL] $model_key / $task / budget=$budget failed!"
            fi
            echo ""
        done
    done
done

echo ""
echo "=============================================="
echo " All CakeKV recall experiments finished!"
echo "=============================================="
