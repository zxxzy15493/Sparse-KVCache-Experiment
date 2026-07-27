#!/usr/bin/env bash
set -euo pipefail

method=${1:?Usage: $0 <method> --models ... --tasks ... --lengths ... --budgets ...}
shift

case "$method" in
    adakv|cakekv|clusterkv|duo-attention|flexprefill|full_attention|h2o|headkv|keyformer|magicpig|minference|pqcache|pyramidkv|quest|retroinfer|snapkv|sparq|streaming|topk|topk32|topp|topp32|xattention) ;;
    *) echo "Unsupported method: $method" >&2; exit 2 ;;
esac

num_samples=50
models=()
tasks=()
lengths=()
budgets=()
fixthreshold=()

read_values() {
    local -n values=$1
    shift
    values=()
    while [[ $# -gt 0 && $1 != --* ]]; do
        values+=("$1")
        shift
    done
    REPLY=$#
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            shift
            read_values models "$@"
            shift "$(( $# - REPLY ))"
            ;;
        --tasks)
            shift
            read_values tasks "$@"
            shift "$(( $# - REPLY ))"
            ;;
        --lengths)
            shift
            read_values lengths "$@"
            shift "$(( $# - REPLY ))"
            ;;
        --budgets)
            shift
            read_values budgets "$@"
            shift "$(( $# - REPLY ))"
            ;;
        --fixthreshold)
            shift
            read_values fixthreshold "$@"
            shift "$(( $# - REPLY ))"
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

if (( ${#models[@]} == 0 || ${#tasks[@]} == 0 || ${#lengths[@]} == 0 )); then
    echo "--models, --tasks, and --lengths must all be provided" >&2
    exit 2
fi
if (( ${#budgets[@]} == 0 && ${#fixthreshold[@]} == 0 )); then
    echo "Either --budgets or --fixthreshold must be provided" >&2
    exit 2
fi
# If only fixthreshold is provided, use a default budget.
if (( ${#budgets[@]} == 0 )); then
    budgets=(1024)
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
if [[ "$method" == "pqcache" ]]; then
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
    export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
benchmark_root=${RULER_BENCHMARK_ROOT:-"$script_dir/benchmark_root"}
prediction_root=${RULER_BENCHMARK_PRED_ROOT:-"$script_dir/benchmark_root_pred"}

model_config() {
    case "$1" in
        llama-3.1-8b) printf '%s:%s\n' 'meta-llama/Llama-3.1-8B-Instruct' 'meta-llama3' ;;
        qwen-2.5-7b-1m) printf '%s:%s\n' 'Qwen/Qwen2.5-7B-Instruct-1M' 'qwen-chat' ;;
    esac
}

_fixthreshold=("${fixthreshold[@]:-}")
if (( ${#_fixthreshold[@]} == 0 )); then
    _fixthreshold=("")
fi

for model in "${models[@]}"; do
    IFS=: read -r tokenizer_path template_type <<< "$(model_config "$model")"
    for budget in "${budgets[@]}"; do
        for threshold in "${_fixthreshold[@]}"; do
            # Build the output-directory signature and threshold override.
            if [[ -n "$threshold" ]]; then
                parameter_signature="budget-${budget}__fixthreshold-${threshold}"
                extra_args=(--set "fixthreshold=$threshold")
            else
                parameter_signature="budget-${budget}"
                extra_args=()
            fi
            for max_seq_length in "${lengths[@]}"; do
                data_dir="$benchmark_root/$model/synthetic/$max_seq_length/data"
                pred_dir="$prediction_root/synthetic/$model/$method/$parameter_signature/$max_seq_length"
                mkdir -p "$data_dir" "$pred_dir"

                for task in "${tasks[@]}"; do
                    python "$script_dir/data/prepare.py" \
                        --save_dir "$data_dir" \
                        --benchmark synthetic \
                        --task "$task" \
                        --tokenizer_path "$tokenizer_path" \
                        --tokenizer_type hf \
                        --max_seq_length "$max_seq_length" \
                        --model_template_type "$template_type" \
                        --num_samples "$num_samples"
                    python "$script_dir/call_api.py" \
                        --data-dir "$data_dir" \
                        --save-dir "$pred_dir" \
                        --task "$task" \
                        --method "$method" \
                        --model "$model" \
                        --max_seq_length "$max_seq_length" \
                        --budget "$budget" \
                        "${extra_args[@]}"
                done

                python "$script_dir/evaluate.py" --data-dir "$pred_dir" --tasks "${tasks[@]}"
            done
        done
    done
done
