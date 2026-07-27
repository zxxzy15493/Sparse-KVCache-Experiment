#!/usr/bin/env bash
set -euo pipefail

# Generate the RULER synthetic data shared by the unified benchmark and the
# method-local RULER runners. Run this script from any working directory.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
data_root="$script_dir/benchmark_root"

models=(llama-3.1-8b qwen-2.5-7b-1m)
lengths=(4096 8192 16384 32768 65536 131072 196608)
tasks=(niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2)
num_samples=50

usage() {
    cat <<'EOF'
Usage: bash benchmarks/ruler/generate_datasets.sh [options]

Generate RULER synthetic data from the sole source assets in
benchmarks/ruler/data/synthetic/json/.

Options:
  --models MODEL [MODEL ...]     Models to prepare (default: both supported models)
  --lengths LENGTH [LENGTH ...]  Context lengths in tokens (default: 4K--192K)
  --tasks TASK [TASK ...]        RULER synthetic tasks (default: all paper tasks)
  --num-samples N                Samples per task (default: 50)
  -h, --help                     Show this help message

Supported models: llama-3.1-8b, qwen-2.5-7b-1m
EOF
}

read_values() {
    local -n destination=$1
    shift
    destination=()
    while [[ $# -gt 0 && $1 != --* ]]; do
        destination+=("$1")
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
        --lengths)
            shift
            read_values lengths "$@"
            shift "$(( $# - REPLY ))"
            ;;
        --tasks)
            shift
            read_values tasks "$@"
            shift "$(( $# - REPLY ))"
            ;;
        --num-samples)
            num_samples=${2:?--num-samples requires an integer}
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if (( ${#models[@]} == 0 || ${#lengths[@]} == 0 || ${#tasks[@]} == 0 )); then
    echo "--models, --lengths, and --tasks must not be empty." >&2
    exit 2
fi

if ! [[ $num_samples =~ ^[1-9][0-9]*$ ]]; then
    echo "--num-samples must be a positive integer." >&2
    exit 2
fi

model_config() {
    case "$1" in
        llama-3.1-8b)
            printf '%s:%s\n' 'meta-llama/Llama-3.1-8B-Instruct' 'meta-llama3'
            ;;
        qwen-2.5-7b-1m)
            printf '%s:%s\n' 'Qwen/Qwen2.5-7B-Instruct-1M' 'qwen-chat'
            ;;
        *)
            echo "Unsupported model: $1" >&2
            exit 2
            ;;
    esac
}

mkdir -p "$data_root"
for model in "${models[@]}"; do
    IFS=: read -r tokenizer_path template_type <<< "$(model_config "$model")"

    for length in "${lengths[@]}"; do
        if ! [[ $length =~ ^[1-9][0-9]*$ ]]; then
            echo "Invalid context length: $length" >&2
            exit 2
        fi
        save_dir="$data_root/$model/synthetic/$length/data"
        mkdir -p "$save_dir"

        for task in "${tasks[@]}"; do
            python "$script_dir/data/prepare.py" \
                --save_dir "$save_dir" \
                --benchmark synthetic \
                --task "$task" \
                --tokenizer_path "$tokenizer_path" \
                --tokenizer_type hf \
                --max_seq_length "$length" \
                --model_template_type "$template_type" \
                --num_samples "$num_samples"
        done
    done
done
