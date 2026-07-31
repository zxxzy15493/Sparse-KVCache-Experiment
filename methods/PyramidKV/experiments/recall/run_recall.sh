#!/bin/bash
# PyramidKV Recall Evaluation Script
# Usage: bash run_recall.sh [group]
#   group: A (LongBench), B (RULER), or empty (all)

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON=${PYTHON:-python}

WINDOW=32

# Model paths
declare -A MODEL_PATHS
MODEL_PATHS[llama3.1-8b-128k]="meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATHS[qwen2.5-7b-instruct]="Qwen/Qwen2.5-7B-Instruct"
MODEL_PATHS[qwen2.5-7b-instruct-1m]="Qwen/Qwen2.5-7B-Instruct-1M"

GROUP=${1:-all}

# ==========================================
# Group A: LongBench (qasper, narrativeqa)
# ==========================================
run_group_a() {
    echo "=============================================="
    echo " GROUP A: LongBench (qasper, narrativeqa)"
    echo " Models:  llama3.1-8b-128k, qwen2.5-7b-instruct"
    echo " Budgets: 128 256 512 1024"
    echo "=============================================="

    LB_DATA="../../../../benchmarks/Longbench_recall"
    LB_DATASETS=("qasper" "narrativeqa")
    LB_BUDGETS=(128 256 512 1024)
    LB_MODELS=(llama3.1-8b-128k qwen2.5-7b-instruct)
    for model_key in "${LB_MODELS[@]}"; do
        model_path="${MODEL_PATHS[$model_key]}"
        for budget in "${LB_BUDGETS[@]}"; do
            for ds in "${LB_DATASETS[@]}"; do
                config_path="config/${model_key}_c${budget}_w${WINDOW}.json"
                echo "============================================================"
                echo "  [A] Model: $model_key | Dataset: $ds | Budget: $budget"
                echo "============================================================"
                $PYTHON recall.py \
                    --model "$model_path" \
                    --dataset "$LB_DATA" \
                    --dataset_name "$ds" \
                    --compress_args_path "$config_path"
                if [ $? -eq 0 ]; then
                    echo "  [OK] $model_key / $ds / budget=$budget done."
                else
                    echo "  [FAIL] $model_key / $ds / budget=$budget failed!"
                fi
                echo ""
            done
        done
    done
}

# ==========================================
# Group B: RULER (fwe, vt, niah_single_3)
# ==========================================
run_group_b() {
    echo "=============================================="
    echo " GROUP B: RULER (fwe, vt, niah_single_3)"
    echo " Models:  llama3.1-8b-128k, qwen2.5-7b-instruct-1m"
    echo " Budgets: 128 384 1024 4096"
    echo "=============================================="

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
                config_path="config/${model_key}_c${budget}_w${WINDOW}.json"
                echo "============================================================"
                echo "  [B] Model: $model_key | Task: $task | Budget: $budget"
                echo "============================================================"
                $PYTHON recall.py \
                    --model "$model_path" \
                    --dataset "$dataset_path" \
                    --task "$task" \
                    --compress_args_path "$config_path"
                if [ $? -eq 0 ]; then
                    echo "  [OK] $model_key / $task / budget=$budget done."
                else
                    echo "  [FAIL] $model_key / $task / budget=$budget failed!"
                fi
                echo ""
            done
        done
    done
}

# ==========================================
# Main
# ==========================================
case "$GROUP" in
    A|a)
        run_group_a
        ;;
    B|b)
        run_group_b
        ;;
    all|ALL)
        run_group_a
        run_group_b
        ;;
    *)
        echo "Usage: $0 [A|B|all]"
        echo "  A   - LongBench (qasper, narrativeqa)"
        echo "  B   - RULER (fwe, vt, niah_single_3)"
        echo "  all - both groups"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo " All PyramidKV recall runs completed!"
echo " Results saved to ./recall_results/"
echo "=============================================="
