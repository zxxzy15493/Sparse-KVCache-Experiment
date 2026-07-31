SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FLEX_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$FLEX_ROOT"
pwd
export PYTHONPATH="$FLEX_ROOT:${PYTHONPATH:-}"

DEFAULT_OUTPUT_DIR="$FLEX_ROOT/breakdown_results"
OUTPUT_TARGET="${1:-}"
if [[ -z "$OUTPUT_TARGET" ]]; then
    OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
    OUTPUT_FOLDER_NAME="${2:-}"
elif [[ "$OUTPUT_TARGET" == */* ]]; then
    OUTPUT_DIR="$OUTPUT_TARGET"
    OUTPUT_FOLDER_NAME="${2:-}"
else
    OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
    OUTPUT_FOLDER_NAME="$OUTPUT_TARGET"
fi
for input_length in 4096 65536; do
# for input_length in 4096 8192 16384 32768 65536 131072; do
    python "$FLEX_ROOT/breakdown/run_flexprefill_breakdown.py" \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset "$REPO_ROOT/benchmarks/myinput.txt" \
        --output_dir "$OUTPUT_DIR" \
        --output_folder_name "$OUTPUT_FOLDER_NAME" \
        --input_lengths $input_length \
        --num_runs 4 \
        --max_new_tokens 32 \
        --gamma 0.9 \
        --tau 0.1 \
        --block_size 128 \
        --min_budget 512
done
