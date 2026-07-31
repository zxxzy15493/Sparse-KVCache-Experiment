#!/bin/bash
# DuoAttention Recall Evaluation Script
#
# Two groups matching CakeKV layout:
#   Group A - LongBench: qasper/narrativeqa x Llama-3.1-8B/Qwen2.5-7B x sparsity 0.5
#   Group B - RULER:     fwe/vt/niah_single_3 x Llama-3.1-8B/Qwen2.5-7B(-1M) x sparsity 0.5
#
# Usage:
#   bash run_recall.sh                          # Run all groups
#   bash run_recall.sh llama3.1-8b-128k         # One model, all datasets
#   bash run_recall.sh all qasper               # All models, one dataset/task

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON=python

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ATTN_PATTERN_DIR="${ATTN_PATTERN_DIR:-../../attn_patterns}"

# ==========================================
# ==========================================
MODEL_CONFIG="${SCRIPT_DIR}/config/model2path.json"
if [ ! -f "$MODEL_CONFIG" ]; then
    get_model_path() {
        case "$1" in
            llama3.1-8b-128k)        echo "meta-llama/Llama-3.1-8B-Instruct" ;;
            qwen2.5-7b-instruct)     echo "Qwen/Qwen2.5-7B-Instruct" ;;
            qwen2.5-7b-instruct-1m)  echo "Qwen/Qwen2.5-7B-Instruct-1M" ;;
            *) echo "" ;;
        esac
    }
    get_model_short() {
        echo "$1" | awk -F'/' '{print $NF}'
    }
else
    get_model_path() {
        python -c "import json; print(json.load(open('$MODEL_CONFIG')).get('$1', ''))"
    }
    get_model_short() {
        echo "$1" | awk -F'/' '{print $NF}'
    }
fi

# ==========================================
# Sparsity values to loop through
# ==========================================
SPARSITY_VALUES=("0.5")

for sparsity in "${SPARSITY_VALUES[@]}"; do

echo ""
echo "=============================================="
echo "  Current sparsity: $sparsity"
echo "=============================================="

# ==========================================
# Group A: LongBench
# ==========================================
echo "=============================================="
echo " Group A: LongBench (qasper, narrativeqa)"
echo " Models:  Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct"
echo " Sparsity: $sparsity"
echo "=============================================="
echo ""

LB_DATA="../../../../benchmarks/Longbench_recall"
LB_DATASETS=("qasper" "narrativeqa")
LB_MODELS=("llama3.1-8b-128k" "qwen2.5-7b-instruct")

for model_key in "${LB_MODELS[@]}"; do
    model_name="$(get_model_path "$model_key")"
    [ -z "$model_name" ] && { echo "[ERROR] Unknown model: $model_key"; continue; }

    # Resolve attention pattern directory
    # Try multiple candidate names: short name, Meta- prefix, model key
    attn_dir=""
    model_short="$(get_model_short "$model_name")"
    for candidate in "$model_short" "Meta-${model_short}" "$model_key"; do
        if [ -d "${ATTN_PATTERN_DIR}/${candidate}" ]; then
            attn_dir="${ATTN_PATTERN_DIR}/${candidate}"
            if [ "$candidate" != "$model_short" ]; then
                echo "  [Info] Using pattern from $candidate (model_short=$model_short)"
            fi
            break
        fi
    done
    if [ -z "$attn_dir" ]; then
        echo "[WARNING] No attention pattern found for $model_name, skipping..."
        continue
    fi

    # Auto-detect subdirectory structure (e.g. Llama pattern in subdir like "lr=0.02-...")
    if [ ! -f "${attn_dir}/full_attention_heads.tsv" ]; then
        subdir=$(find "${attn_dir}" -maxdepth 1 -type d -exec test -f "{}/full_attention_heads.tsv" \; -print 2>/dev/null | head -1)
        if [ -n "$subdir" ]; then
            echo "  [Info] Using subdirectory: $(basename "$subdir")"
            attn_dir="$subdir"
        fi
    fi

    echo "Model: $model_name ($model_key)"
    echo "  Pattern: $attn_dir"

    for ds in "${LB_DATASETS[@]}"; do
        echo ""
        echo "============================================================"
        echo "  [A] $model_key | $ds | sparsity $sparsity"
        echo "============================================================"

        cd "$SCRIPT_DIR"
        $PYTHON recall.py \
            --model "$model_name" \
            --dataset "$LB_DATA" \
            --dataset_name "$ds" \
            --attn_load_dir "$attn_dir" \
            --sparsity $sparsity \
            --sink_size 128 \
            --recent_size 256

        if [ $? -eq 0 ]; then
            echo "  [OK] $model_key / $ds done (sparsity=$sparsity)."
        else
            echo "  [FAIL] $model_key / $ds failed (sparsity=$sparsity)!"
        fi
    done
done

# ==========================================
# Group B: RULER synthetic datasets
# ==========================================
echo ""
echo "=============================================="
echo " Group B: RULER (fwe, vt, niah_single_3)"
echo " Models:  Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct-1M"
echo " Seq_len: 65536 | Sparsity: $sparsity"
echo "=============================================="
echo ""

RULER_ROOT="../../../../benchmarks/Ruler_recall"

get_ruler_data_dir() {
    case "$1" in
        llama3.1-8b-128k)       echo "llama-3.1-8b" ;;
        qwen2.5-7b-instruct-1m) echo "qwen-2.5-7b-1m" ;;
        *) echo "" ;;
    esac
}
RULER_TASKS=("fwe" "vt" "niah_single_3")
RULER_MODELS=("llama3.1-8b-128k" "qwen2.5-7b-instruct-1m")
SEQ_LEN=65536

for model_key in "${RULER_MODELS[@]}"; do
    model_name="$(get_model_path "$model_key")"
    [ -z "$model_name" ] && { echo "[ERROR] Unknown model: $model_key"; continue; }

    # Resolve attention pattern directory
    attn_dir=""
    model_short="$(get_model_short "$model_name")"
    for candidate in "$model_short" "Meta-${model_short}" "${model_short%-1M}" "Meta-${model_short%-1M}" "$model_key"; do
        if [ -d "${ATTN_PATTERN_DIR}/${candidate}" ]; then
            attn_dir="${ATTN_PATTERN_DIR}/${candidate}"
            if [ "$candidate" != "$model_short" ]; then
                echo "  [Info] $model_short reusing pattern from $candidate"
            fi
            break
        fi
    done
    if [ -z "$attn_dir" ]; then
        echo "[WARNING] No attention pattern found for $model_name, skipping RULER..."
        continue
    fi

    # Auto-detect subdirectory structure (same as Group A)
    if [ ! -f "${attn_dir}/full_attention_heads.tsv" ]; then
        subdir=$(find "${attn_dir}" -maxdepth 1 -type d -exec test -f "{}/full_attention_heads.tsv" \; -print 2>/dev/null | head -1)
        if [ -n "$subdir" ]; then
            echo "  [Info] Using subdirectory: $(basename "$subdir")"
            attn_dir="$subdir"
        fi
    fi

    # RULER data path for this model
    data_dir="$(get_ruler_data_dir "$model_key")"
    [ -z "$data_dir" ] && { echo "[ERROR] Unknown RULER data dir for $model_key"; continue; }
    dataset_path="${RULER_ROOT}/${data_dir}/synthetic/${SEQ_LEN}/data"

    if [ ! -d "$dataset_path" ]; then
        echo "[WARNING] RULER data not found: $dataset_path, skipping..."
        continue
    fi

    echo "Model: $model_name ($model_key)"
    echo "  Pattern: $attn_dir"
    echo "  RULER data: $dataset_path"

    for task in "${RULER_TASKS[@]}"; do
        echo ""
        echo "============================================================"
        echo "  [B] $model_key | $task | sparsity $sparsity"
        echo "============================================================"

        cd "$SCRIPT_DIR"
        $PYTHON recall.py \
            --model "$model_name" \
            --dataset "$dataset_path" \
            --task "$task" \
            --attn_load_dir "$attn_dir" \
            --sparsity $sparsity \
            --sink_size 128 \
            --recent_size 256

        if [ $? -eq 0 ]; then
            echo "  [OK] $model_key / $task done (sparsity=$sparsity)."
        else
            echo "  [FAIL] $model_key / $task failed (sparsity=$sparsity)!"
        fi
    done
done

done

echo ""
echo "=============================================="
echo " All DuoAttention recall runs completed!"
echo " Results saved to ./recall_results/ (subdirectories named by sparsity)"
echo "=============================================="