cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
DEFAULT_OUTPUT_DIR="./breakdown_results"
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

for input_length in 4096 65536;do
    python breakdown/run_xattention_breakdown.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --model_family llama \
        --dataset benchmarks/myinput.txt \
        --output_dir "$OUTPUT_DIR" \
        --output_folder_name "$OUTPUT_FOLDER_NAME" \
        --input_lengths $input_length \
        --num_runs 4 \
        --max_new_tokens 32 \
        --stride 16
done