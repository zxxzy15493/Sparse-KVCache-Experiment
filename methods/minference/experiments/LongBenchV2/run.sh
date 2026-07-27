#!/bin/bash
# Run LongBenchV2 evaluation with MInference
#
# Usage:
#   bash run.sh <MODEL> [--no-cot] [--no-context] [--num-samples N]
#
#   MODEL: one of Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-7B-Instruct-1M,
#          GLM-4-9B-Chat-1M, DeepSeek-R1-Distill-Qwen-1.5B
#
# Examples:
#   bash run.sh Qwen2.5-7B-Instruct-1M
#   bash run.sh Llama-3.1-8B-Instruct
#   bash run.sh Qwen2.5-7B-Instruct-1M --no-cot

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${1:-Qwen2.5-7B-Instruct-1M}"
shift || true

# Default flags
COT_FLAG="--cot"
NO_CONTEXT_FLAG=""
NUM_SAMPLES_FLAG=""
EXTRA_ARGS=()

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cot)
            COT_FLAG=""
            shift
            ;;
        --no-context)
            NO_CONTEXT_FLAG="--no_context"
            shift
            ;;
        --num-samples)
            NUM_SAMPLES_FLAG="--num_samples $2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "========================================="
echo "  LongBenchV2 with MInference"
echo "  Model: $MODEL"
echo "  CoT: ${COT_FLAG:-disabled}"
echo "========================================="

python pred.py \
    --model "$MODEL" \
    $COT_FLAG \
    $NO_CONTEXT_FLAG \
    $NUM_SAMPLES_FLAG \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Prediction done. Evaluating..."
echo ""

python result.py "results/$MODEL"
