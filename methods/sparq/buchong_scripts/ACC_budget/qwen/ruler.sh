#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SPARQ_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

GPUS="${GPUS:-1}"
MODEL_DIR="${MODEL_DIR:-$SPARQ_ROOT/models}"
ENGINE_DIR="${ENGINE_DIR:-.}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DATA_ROOT="${DATA_ROOT:-$SPARQ_ROOT/../../benchmarks/ruler/benchmark_root}"
RES_DIR="${RES_DIR:-$SPARQ_ROOT/output_budget/ruler}"

cd "$SPARQ_ROOT/experiments/ruler/scripts"
export PYTHONPATH="$SPARQ_ROOT:${PYTHONPATH:-}"
source config_models.sh

MODEL_NAME="qwen-2.5-7b-1m"
MODEL_CONFIG=$(MODEL_SELECT "$MODEL_NAME" "$MODEL_DIR" "$ENGINE_DIR")
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"

BENCHMARK="synthetic"
SEQ_LENGTHS1=(
    65536
)

NAME="ann"
SCORE="sparse_q"
REALLOCATE_TO_MEAN_VALUE=True
total_time=0
TASKS="niah_single_3 vt cwe fwe qa_1"

for MAX_SEQ_LENGTH in 65536; do
    data_DIR="${DATA_ROOT}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    pred_DIR="${RES_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${data_DIR}/data"
    PRED_DIR="${pred_DIR}/pred"

    mkdir -p "$DATA_DIR" "$PRED_DIR"

    for TASK in $TASKS; do
        for K in 128 384 1024 4096; do
            PRED_DIR_final="${PRED_DIR}/${K}"
            for LOCAL_K in 32; do
                for RANK in 16; do
                    start_time=$(date +%s)
                    python pred/sparq_call_api.py \
                        --data_dir "$DATA_DIR" \
                        --save_dir "$PRED_DIR_final" \
                        --benchmark "$BENCHMARK" \
                        --task "$TASK" \
                        --server_type "$MODEL_FRAMEWORK" \
                        --model_name_or_path "Qwen/Qwen2.5-7B-Instruct-1M" \
                        --temperature "$TEMPERATURE" \
                        --top_k "$TOP_K" \
                        --top_p "$TOP_P" \
                        --batch_size "$BATCH_SIZE" \
                        --name "$NAME" \
                        --k "$K" \
                        --local_k "$LOCAL_K" \
                        --score "$SCORE" \
                        --rank "$RANK" \
                        --reallocate_to_mean_value "$REALLOCATE_TO_MEAN_VALUE" \
                        ${STOP_WORDS}
                    end_time=$(date +%s)
                    time_diff=$((end_time - start_time))
                    total_time=$((total_time + time_diff))
                done
            done
        done
    done

    python eval/evaluate.py \
        --data_dir "$PRED_DIR" \
        --benchmark "$BENCHMARK"
done

echo "Total time spent on call_api: $total_time seconds"
