cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
DEFAULT_OUTPUT_DIR="./breakdown_no_repeatkv_core_attn_results174"
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

for input_length in 4096 8192 16384 32768 65536 131072; do
    python breakdown/run_xattention_no_repeatkv_breakdown.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --model_family llama \
        --dataset benchmarks/myinput.txt \
        --output_dir "$OUTPUT_DIR" \
        --output_folder_name "$OUTPUT_FOLDER_NAME" \
        --input_lengths "$input_length" \
        --num_runs 4 \
        --max_new_tokens 32 \
        --stride 16
done

for input_length in 4096 8192 16384 32768 65536 131072; do
    python breakdown/run_xattention_no_repeatkv_breakdown.py \
        --model Qwen/Qwen2.5-7B-Instruct-1M \
        --model_family qwen \
        --dataset benchmarks/myinput.txt \
        --output_dir "$OUTPUT_DIR" \
        --output_folder_name "$OUTPUT_FOLDER_NAME" \
        --input_lengths "$input_length" \
        --num_runs 4 \
        --max_new_tokens 32 \
        --stride 16
done
