
set -euo pipefail

# ======================== Paths ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${EVAL_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/..:${PROJECT_ROOT}:${EVAL_DIR}:${PYTHONPATH:-}"

# ======================== Default Configuration ========================
DATASET=""
MODEL=""
SPARSITIES=()           # space-separated list, e.g. "0.5 0.6 0.7 0.8 0.9"
MAX_SAMPLES=500
SINK_SIZE=128
RECENT_SIZE=256
DEVICE=0

# --- LongBench specific ---
DEFAULT_LONGBENCH_TASKS=("narrativeqa" "qasper" "lcc" "trec")
LONGBENCH_TASKS=("${DEFAULT_LONGBENCH_TASKS[@]}")

# --- RULER specific ---
RULER_BENCHMARK="synthetic"
RULER_SEQ_LENGTHS=(4096)
RULER_NUM_SAMPLES=50

# --- GSM8K specific ---
GSM8K_NUM_SHOTS=8
GSM8K_MAX_NEW_TOKENS=10000

# ======================== Subcommand detection ========================
SUBCOMMAND=""
if [[ $# -gt 0 && "$1" != --* ]]; then
    case "$1" in
        overview|budget|full)
            SUBCOMMAND="$1"
            shift
            ;;
    esac
fi

RULER_TASK_FILTER=()   # empty = all tasks; set to subset e.g. ("niah_single_3" "vt")

# ======================== Argument Parsing ========================
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h|help)
            echo "Usage: $0 <subcommand|--dataset ...> [options]"
            echo ""
            echo "Subcommands (preset experiment suites):"
            echo "  overview       LongBench overview (sparsity=0.5, 3 models) + RULER overview (4k-64k, sparsity=0.5, 2 models)"
            echo "  budget         LongBench budget (sparsities 0.6/0.7/0.8/0.9, 2 models) + RULER budget (64k sparsities 0.6/0.7/0.8/0.9, 2 models)"
            echo "  full           overview + budget, all at once"
            echo ""
            echo "Or use --dataset directly:"
            echo "  Required:"
            echo "    --dataset         gsm8k | longbench | longbench-v2 | ruler | all"
            echo "    --model           model name (llama/qwen/glm/deepseek) or HF path"
            echo "    --sparsities      space-separated sparsity ratios, e.g. \"0.5 0.7 0.9\""
            echo ""
            echo "  Optional (common):"
            echo "    --max_samples     max samples per dataset      (default: 500)"
            echo "    --sink_size       sink token count             (default: 128)"
            echo "    --recent_size     recent/window token count    (default: 256)"
            echo "    --device          CUDA device id               (default: 0)"
            echo ""
            echo "  Optional (LongBench):"
            echo "    --longbench_tasks space-separated task names (default: narrativeqa qasper lcc trec)"
            echo ""
            echo "  Optional (RULER):"
            echo "    --ruler_benchmark     benchmark name          (default: synthetic)"
            echo "    --ruler_seq_lengths   space-separated lengths  (default: \"4096\")"
            echo "    --ruler_num_samples   samples per task        (default: 50)"
            echo ""
            echo "  Optional (GSM8K):"
            echo "    --num_shots       few-shot count              (default: 8)"
            echo ""
            echo "  Allowed dataset-model combinations:"
            echo "    longbench:   llama, qwen, glm"
            echo "    ruler:       llama, qwen"
            echo "    gsm8k:       deepseek"
            echo "    longbench-v2: qwen"
            echo ""
            echo "  Raw HuggingFace paths (e.g. \"org/model-name\") are also accepted."
            echo ""
            echo "Examples:"
            echo "  $0 full"
            echo "  $0 --dataset gsm8k --model deepseek --sparsities \"0.5\""
            echo "  $0 --dataset longbench --model qwen --sparsities \"0.5 0.7 0.9\""
            exit 0
            ;;
        --dataset)
            DATASET="$2"; shift 2 ;;
        --model)
            MODEL="$2"; shift 2 ;;
        --sparsities)
            IFS=' ' read -ra SPARSITIES <<< "$2"; shift 2 ;;
        --max_samples)
            MAX_SAMPLES="$2"; shift 2 ;;
        --sink_size)
            SINK_SIZE="$2"; shift 2 ;;
        --recent_size)
            RECENT_SIZE="$2"; shift 2 ;;
        --device)
            DEVICE="$2"; shift 2 ;;
        --longbench_tasks)
            IFS=' ' read -ra LONGBENCH_TASKS <<< "$2"; shift 2 ;;
        --ruler_benchmark)
            RULER_BENCHMARK="$2"; shift 2 ;;
        --ruler_seq_lengths)
            IFS=' ' read -ra RULER_SEQ_LENGTHS <<< "$2"; shift 2 ;;
        --ruler_num_samples)
            RULER_NUM_SAMPLES="$2"; shift 2 ;;
        --num_shots)
            GSM8K_NUM_SHOTS="$2"; shift 2 ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Use --help for usage information."
            exit 1 ;;
    esac
done

# ======================== Validation ========================
if [[ -n "$SUBCOMMAND" ]]; then
    # Subcommands (overview/budget/full) skip --dataset/--model/--sparsities validation
    :
elif [[ -z "$DATASET" ]]; then
    echo "Error: --dataset is required (gsm8k | longbench | longbench-v2 | ruler | all)"
    exit 1
elif [[ -z "$MODEL" ]]; then
    echo "Error: --model is required"
    exit 1
elif [[ ${#SPARSITIES[@]} -eq 0 ]]; then
    echo "Error: --sparsities is required (space-separated list, e.g. '0.5 0.6 0.7 0.8 0.9')"
    exit 1
fi

if [[ -z "$SUBCOMMAND" ]]; then
    VALID_DATASETS=("gsm8k" "longbench" "longbench-v2" "ruler" "all")
    if [[ ! " ${VALID_DATASETS[*]} " =~ " ${DATASET} " ]]; then
        echo "Error: --dataset must be one of: gsm8k, longbench, longbench-v2, ruler, all"
        exit 1
    fi
fi

# ======================== Model mapping ========================
# Each benchmark needs a different form of the model name:
#   LongBench / LongBench-v2:  key in config/model2path.json (pred.py resolves internally)
#   GSM8K:                     HuggingFace path
#   RULER:                     short name for config_models.sh

declare -A MODEL_MAP
# LongBench (model2path.json keys)
MODEL_MAP["longbench:llama"]="Meta-Llama-3.1-8B-Instruct"
MODEL_MAP["longbench:qwen"]="Qwen2.5-7B-Instruct"
MODEL_MAP["longbench:glm"]="glm-4-9b-chat-1m"
# RULER (config_models.sh short names)
MODEL_MAP["ruler:llama"]="llama-3.1-8b"
MODEL_MAP["ruler:qwen"]="qwen-2.5-7b-1m"
# GSM8K (HF path)
MODEL_MAP["gsm8k:deepseek"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# LongBench-v2 (model2path.json keys)
MODEL_MAP["longbench-v2:qwen"]="Qwen2.5-7B-Instruct-1M"

# Attention pattern directories per benchmark+model
declare -A ATTN_PATTERN_MAP
ATTN_PATTERN_MAP["gsm8k:deepseek"]="${PROJECT_ROOT}/attn_patterns/deepseek-r1-distill-qwen-1.5b"
ATTN_PATTERN_MAP["longbench:llama"]="${PROJECT_ROOT}/attn_patterns/Meta-Llama-3.1-8B-Instruct"
ATTN_PATTERN_MAP["longbench:qwen"]="${PROJECT_ROOT}/attn_patterns/Qwen2.5-7B-Instruct"
ATTN_PATTERN_MAP["longbench:glm"]="${PROJECT_ROOT}/attn_patterns/glm-4-9b-chat-1m"
ATTN_PATTERN_MAP["longbench-v2:qwen"]="${PROJECT_ROOT}/attn_patterns/Qwen2.5-7B-Instruct"
ATTN_PATTERN_MAP["ruler:llama"]="${PROJECT_ROOT}/attn_patterns/Meta-Llama-3.1-8B-Instruct"
ATTN_PATTERN_MAP["ruler:qwen"]="${PROJECT_ROOT}/attn_patterns/Qwen2.5-7B-Instruct"

resolve_model_name() {
    local ds=$1
    local canonical=$2
    local key="${ds}:${canonical}"
    local result="${MODEL_MAP[$key]:-}"
    if [[ -z "$result" ]]; then
        if [[ "$canonical" == */* ]]; then result="$canonical"
        else
            echo "ERROR: unsupported dataset-model combination: dataset='$ds', model='$canonical'" >&2
            echo "Allowed: longbench: llama,qwen,glm | ruler: llama,qwen | gsm8k: deepseek | longbench-v2: qwen" >&2
            echo "Or pass a raw HuggingFace path (e.g. 'org/model-name')" >&2
            exit 1
        fi
    fi
    echo "$result"
}

resolve_attn_pattern() {
    local ds=$1
    local canonical=$2
    local key="${ds}:${canonical}"
    echo "${ATTN_PATTERN_MAP[$key]:-}"
}

check_model_compat() {
    local ds=$1
    local canonical=$2
    local key="${ds}:${canonical}"
    [[ -n "${MODEL_MAP[$key]:-}" ]] && return 0
    [[ "$canonical" == */* ]] && return 0
    return 1
}

# ======================== GSM8K ========================
run_gsm8k() {
    local sp=$1
    local gsm8k_model=$(resolve_model_name "gsm8k" "$MODEL")
    local attn_dir=$(resolve_attn_pattern "gsm8k" "$MODEL")
    local model_tag=$(echo "$gsm8k_model" | tr '/' '_')
    local log_dir="${EVAL_DIR}/GSM8K/log/DuoAttention/${model_tag}/sp${sp}"
    local save_dir="${EVAL_DIR}/GSM8K/results/DuoAttention/${model_tag}/sp${sp}"

    echo "============================================"
    echo "[GSM8K] canonical=$MODEL  resolved=$gsm8k_model  sparsity=$sp"
    echo "============================================"

    mkdir -p "$log_dir" "$save_dir"

    cd "${EVAL_DIR}/GSM8K"

    python -u pred_duo.py \
        --model "$gsm8k_model" \
        --save_dir "$save_dir" \
        --attn_load_dir "$attn_dir" \
        --sparsity "$sp" \
        --sink_size "$SINK_SIZE" \
        --recent_size "$RECENT_SIZE" \
        --num_shots "$GSM8K_NUM_SHOTS" \
        --cot_type gsm8k-cot \
        --max_new_tokens "$GSM8K_MAX_NEW_TOKENS" \
        > "${log_dir}/gsm8k.log" 2>&1

    python -u evaluate.py \
        --input "${save_dir}/gsm8k.jsonl" \
        --output "${save_dir}/gsm8k_eval.jsonl" \
        --force

    python -u ./tool/data_infos.py \
        --data-dir "$save_dir" \
        --model "$gsm8k_model" \
        --task gsm8k
}

# ======================== LongBench ========================
run_longbench() {
    local sp=$1
    local lb_model_key=$(resolve_model_name "longbench" "$MODEL")
    local attn_dir=$(resolve_attn_pattern "longbench" "$MODEL")

    echo "============================================"
    echo "[LongBench] canonical=$MODEL  resolved=$lb_model_key  sparsity=$sp"
    echo "============================================"

    cd "${EVAL_DIR}/LongBench"

    for task in "${LONGBENCH_TASKS[@]}"; do
        echo "[LongBench] task=$task  sparsity=$sp"
        python -u pred.py \
            --model "$lb_model_key" \
            --task "$task" \
            --method duo_attn \
            --attn_load_dir "$attn_dir" \
            --sparsity "$sp" \
            --sink_size "$SINK_SIZE" \
            --recent_size "$RECENT_SIZE" \
            --max_num_examples "$MAX_SAMPLES"
    done

    # Evaluate (eval.py reads from pred/ directory)
    # if [[ -f eval.py ]]; then
    #     python -u eval.py
    # else
    #     echo "[LongBench] WARNING: eval.py not found, skipping evaluation."
    # fi
}

# ======================== LongBench-v2 ========================
run_longbench_v2() {
    local sp=$1
    local lbv2_model_key=$(resolve_model_name "longbench-v2" "$MODEL")
    local attn_dir=$(resolve_attn_pattern "longbench-v2" "$MODEL")

    echo "============================================"
    echo "[LongBench-v2] canonical=$MODEL  resolved=$lbv2_model_key  sparsity=$sp"
    echo "============================================"

    cd "${EVAL_DIR}/LongBench-v2"

    local save_dir="results/duo_attn_sp${sp}"

    python -u pred.py \
        --model "$lbv2_model_key" \
        --method duo_attn \
        --attn_load_dir "$attn_dir" \
        --sparsity "$sp" \
        --sink_size "$SINK_SIZE" \
        --recent_size "$RECENT_SIZE" \
        --cot \
        --save_dir "$save_dir" \
        --device "$DEVICE"

    # Evaluate
    local result_file="${save_dir}/${lbv2_model_key}.jsonl"
    if [[ -f "$result_file" ]]; then
        mkdir -p results
        cp "$result_file" "results/sp${sp}_${lbv2_model_key}.jsonl"
        python result.py
    else
        echo "[LongBench-v2] WARNING: result file not found: $result_file, skipping eval."
    fi
}

# ======================== RULER ========================
run_ruler() {
    local sp=$1
    local ruler_model=$(resolve_model_name "ruler" "$MODEL")
    local attn_dir=$(resolve_attn_pattern "ruler" "$MODEL")

    echo "============================================"
    echo "[RULER] canonical=$MODEL  resolved=$ruler_model  sparsity=$sp"
    echo "============================================"

    cd "${EVAL_DIR}/RULER"

    source "${EVAL_DIR}/RULER/config_models.sh"
    source "${EVAL_DIR}/RULER/config_tasks.sh"

    local model_short="${ruler_model}"
    local model_config=$(MODEL_SELECT "${model_short}" "" "")
    IFS=":" read -r MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE <<< "$model_config"
    if [[ -z "$MODEL_PATH" ]]; then
        echo "[RULER] ERROR: Model '${model_short}' is not supported in config_models.sh"
        return 1
    fi

    local benchmark="${RULER_BENCHMARK}"
    declare -n TASKS=$benchmark
    if [[ -z "${TASKS:-}" ]]; then
        echo "[RULER] ERROR: Benchmark '${benchmark}' is not supported in config_tasks.sh"
        return 1
    fi

    # Determine which tasks to run
    local -a TASK_LIST=()
    if [[ ${#RULER_TASK_FILTER[@]} -gt 0 ]]; then
        TASK_LIST=("${RULER_TASK_FILTER[@]}")
    else
        TASK_LIST=("${TASKS[@]}")
    fi

    # Map short model name to data directory name
    local data_short="${model_short}"
    case "${MODEL}" in
        llama) data_short="llama-3.1-8b" ;;
        qwen)  data_short="qwen-2.5-7b-1m"  ;;
    esac

    local max_examples="$MAX_SAMPLES"
    if [[ "$MAX_SAMPLES" -lt "$RULER_NUM_SAMPLES" ]]; then
        max_examples="$MAX_SAMPLES"
    else
        max_examples="$RULER_NUM_SAMPLES"
    fi

    for MAX_SEQ_LENGTH in "${RULER_SEQ_LENGTHS[@]}"; do
        local results_dir="./ruler_eval_result/${model_short}/${benchmark}/${MAX_SEQ_LENGTH}"
        local data_dir="../../../../benchmarks/ruler/benchmark_root/${data_short}/${benchmark}/${MAX_SEQ_LENGTH}/data"
        local pred_dir="${results_dir}/pred/sp${sp}"
        mkdir -p "$data_dir" "$pred_dir"

        for TASK in "${TASK_LIST[@]}"; do
            local task_data="${data_dir}/${TASK}/validation.jsonl"
            if [[ ! -f "$task_data" ]]; then
                echo "[RULER] WARNING: task data not found: $task_data, skipping."
                continue
            fi

            echo "[RULER] seq_len=$MAX_SEQ_LENGTH  sparsity=$sp  task=$TASK"
            python run_ruler.py \
                --task "$TASK" \
                --task_data "$task_data" \
                --save_dir "$pred_dir" \
                --benchmark "$benchmark" \
                --model_name "$model_short" \
                --model_path "$MODEL_PATH" \
                --method duo_attn \
                --attn_load_dir "$attn_dir" \
                --sparsity "$sp" \
                --sink_size "$SINK_SIZE" \
                --recent_size "$RECENT_SIZE" \
                --max_examples "$max_examples" \
                --seed 42 \
                --use_cache
        done

        # python eval/evaluate.py \
        #     --data_dir "$pred_dir" \
        #     --benchmark "$benchmark"
    done
}

# ======================== Task definitions ========================
LONGBENCH_ALL_TASKS=("narrativeqa" "qasper" "2wikimqa" "musique" "gov_report" "multi_news" "triviaqa" "samsum" "passage_count" "passage_retrieval_en" "lcc" "repobench-p")
LONGBENCH_BUDGET_TASKS=("narrativeqa" "qasper" "trec" "lcc")

RULER_OVERVIEW_TASKS=()  # empty means all tasks in config_tasks.sh

declare -A RULER_BUDGET_TASKS
RULER_BUDGET_TASKS["llama"]="niah_single_3 vt fwe qa_1"
RULER_BUDGET_TASKS["qwen"]="niah_single_3 vt fwe qa_1 cwe"

declare -A RULER_LONG_TASKS
RULER_LONG_TASKS["128k"]="niah_single_1 niah_multiquery vt fwe"
RULER_LONG_TASKS["192k"]="niah_single_1 niah_multiquery vt fwe"

# ======================== Print functions ========================
print_overview_table() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║                           Accuracy: Overview                             ║"
    echo "╠════════════════════════════════════════════════════════════════════════════╣"
    echo "║ LongBench: sparsity=0.5, models=llama/qwen/glm                           ║"
    echo "║   All 12 tasks: narrativeqa, qasper, 2wikimqa, musique, gov_report,      ║"
    echo "║     multi_news, triviaqa, samsum, passage_count, passage_retrieval_en,   ║"
    echo "║     lcc, repobench-p                                                     ║"
    echo "║                                                                          ║"
    echo "║ RULER (4k/8k/16k/32k/64k): sparsity=0.5, models=llama/qwen              ║"
    echo "║   All tasks in config_tasks.sh                                           ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
}

print_budget_table() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║                           Accuracy: Budget                               ║"
    echo "╠════════════════════════════════════════════════════════════════════════════╣"
    echo "║ LongBench: sparsities=0.6/0.7/0.8/0.9, models=llama/qwen                 ║"
    echo "║   Tasks (4): narrativeqa, qasper, trec, lcc                              ║"
    echo "║                                                                          ║"
    echo "║ RULER (64k): sparsities=0.6/0.7/0.8/0.9, models=llama/qwen              ║"
    echo "║   Tasks (4): niah_single_3, vt, fwe, qa_1                               ║"
    echo "║   Qwen extra: + cwe                                                      ║"
    echo "║                                                                          ║"
    echo "║ RULER (128k): sparsity=0.5, budget=2048, model=qwen only                 ║"
    echo "║   Tasks (4): niah_single_1, niah_multiquery, vt, fwe                     ║"
    echo "║ RULER (192k): sparsity=0.5, budget=4096, model=qwen only                 ║"
    echo "║   Tasks (4): niah_single_1, niah_multiquery, vt, fwe                     ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
}

# ======================== Subcommand helpers (sparsity-based) ========================
run_longbench_overview() {
    local saved_model="$MODEL"
    for m in llama qwen glm; do
        MODEL="$m"; export MODEL
        local sp="0.5"
        echo ""
        echo "========== [LongBench Overview] model=$MODEL  sparsity=$sp =========="
        LONGBENCH_TASKS=("${LONGBENCH_ALL_TASKS[@]}")
        run_longbench "$sp"
    done
    MODEL="$saved_model"
}

run_ruler_overview() {
    local saved_model="$MODEL"
    for m in llama qwen; do
        MODEL="$m"
        local sp="0.5"
        RULER_TASK_FILTER=()
        for seq in 4096 8192 16384 32768 65536; do
            RULER_SEQ_LENGTHS=($seq)
            echo ""
            echo "========== [RULER Overview] model=$MODEL  sparsity=$sp  seq=${seq} =========="
            run_ruler "$sp"
        done
    done
    MODEL="$saved_model"
    RULER_SEQ_LENGTHS=(4096 8192 16384 32768 65536)
}

run_longbench_budget() {
    local saved_model="$MODEL"
    for m in llama qwen; do
        MODEL="$m"
        LONGBENCH_TASKS=("${LONGBENCH_BUDGET_TASKS[@]}")
        for sp in 0.6 0.7 0.8 0.9; do
            echo ""
            echo "========== [LongBench Budget] model=$MODEL  sparsity=$sp =========="
            run_longbench "$sp"
        done
    done
    MODEL="$saved_model"
}

run_ruler_budget_64k() {
    local saved_model="$MODEL"
    for m in llama qwen; do
        MODEL="$m"
        local tasks_str="${RULER_BUDGET_TASKS[$m]:-$RULER_BUDGET_TASKS[llama]}"
        IFS=' ' read -ra RULER_TASK_FILTER <<< "$tasks_str"
        RULER_SEQ_LENGTHS=(65536)
        for sp in 0.6 0.7 0.8 0.9; do
            echo ""
            echo "========== [RULER Budget 64k] model=$MODEL  sparsity=$sp =========="
            run_ruler "$sp"
        done
    done
    MODEL="$saved_model"
}

run_ruler_budget_long() {
    local saved_model="$MODEL"
    MODEL="qwen"
    for seq_len in 128k 192k; do
        local tasks_str="${RULER_LONG_TASKS[$seq_len]}"
        IFS=' ' read -ra RULER_TASK_FILTER <<< "$tasks_str"
        local sp="0.5"
        case $seq_len in
            128k) RULER_SEQ_LENGTHS=(131072) ;;
            192k) RULER_SEQ_LENGTHS=(196608) ;;
        esac
        echo ""
        echo "========== [RULER Budget ${seq_len}] model=$MODEL  sparsity=$sp =========="
        run_ruler "$sp"
    done
    MODEL="$saved_model"
    RULER_SEQ_LENGTHS=(4096 8192 16384 32768 65536)
}

# ======================== Main dispatch ========================
run_dataset() {
    local ds=$1
    case $ds in
        gsm8k)
            for SP in "${SPARSITIES[@]}"; do run_gsm8k "$SP"; done ;;
        longbench)
            for SP in "${SPARSITIES[@]}"; do run_longbench "$SP"; done ;;
        longbench-v2)
            for SP in "${SPARSITIES[@]}"; do run_longbench_v2 "$SP"; done ;;
        ruler)
            for SP in "${SPARSITIES[@]}"; do run_ruler "$SP"; done ;;
    esac
}

overview_run() {
    print_overview_table
    run_longbench_overview
    run_ruler_overview
}

budget_run() {
    print_budget_table
    run_longbench_budget
    run_ruler_budget_64k
    run_ruler_budget_long
}

full_run() {
    overview_run
    budget_run
}

if [[ -n "$SUBCOMMAND" ]]; then
    case "$SUBCOMMAND" in
        overview) overview_run ;;
        budget)   budget_run ;;
        ruler-budget) print_budget_table; run_ruler_budget_64k; run_ruler_budget_long ;;
        full)     full_run ;;
    esac
else
    echo "============================================================"
    echo "DuoAttention Accuracy Runner"
    echo "  Dataset:       $DATASET"
    echo "  Model:         $MODEL"
    echo "  Sparsities:    ${SPARSITIES[*]}"
    echo "  Max samples:   $MAX_SAMPLES"
    echo "  Sink size:     $SINK_SIZE"
    echo "  Recent size:   $RECENT_SIZE"
    echo "============================================================"

    if [[ "$DATASET" == "all" ]]; then
        for ds in "gsm8k" "longbench" "longbench-v2" "ruler"; do
            echo ""
            if ! check_model_compat "$ds" "$MODEL"; then
                echo "########## Skipping $ds (model '$MODEL' not valid for this dataset) ##########"
                continue
            fi
            echo "########## Running $ds ##########"
            run_dataset "$ds"
        done
    else
        run_dataset "$DATASET"
    fi
fi

echo ""
echo "All done."