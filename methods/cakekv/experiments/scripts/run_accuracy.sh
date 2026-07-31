

set -euo pipefail

# ======================== Paths ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${EXPERIMENTS_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/..:${PROJECT_ROOT}:${EXPERIMENTS_DIR}:${PYTHONPATH:-}"

# ======================== Default Configuration ========================
# --- Common parameters ---
DATASET=""          # gsm8k | longbench | longbench-v2 | ruler | all
MODEL=""            # model name/path (see each benchmark's config for supported names)
CACHE_SIZES=()      # space-separated list, e.g. "128 256 512 1024"
MAX_SAMPLES=500    # max samples per dataset (LongBench / RULER)
WINDOW_SIZE=32
GAMMA=200.0
TAU1=1.0
TAU2=1.0
COMPRESS=true
CASCADING=true
DEVICE=0

# --- RULER specific ---
RULER_BENCHMARK="synthetic"
RULER_SEQ_LENGTHS=(4096)
RULER_NUM_SAMPLES=50
RULER_GEN_DATA=false    # whether to (re-)generate RULER data before prediction

# --- GSM8K specific ---
GSM8K_NUM_SHOTS=8
GSM8K_COT_TYPE="gsm8k-cot"
GSM8K_MAX_NEW_TOKENS=10000

# --- LongBench specific ---
LONGBENCH_PRED_NAME="pred"   # output subfolder name under pred_result

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
        --help|-h)
            DATASET="__help__"
            shift
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --cache_sizes)
            IFS=' ' read -ra CACHE_SIZES <<< "$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --window_size)
            WINDOW_SIZE="$2"
            shift 2
            ;;
        --gamma)
            GAMMA="$2"
            shift 2
            ;;
        --tau1)
            TAU1="$2"
            shift 2
            ;;
        --tau2)
            TAU2="$2"
            shift 2
            ;;
        --compress)
            COMPRESS="$2"
            shift 2
            ;;
        --cascading)
            CASCADING="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --pred_name)
            LONGBENCH_PRED_NAME="$2"
            shift 2
            ;;
        --ruler_benchmark)
            RULER_BENCHMARK="$2"
            shift 2
            ;;
        --ruler_seq_lengths)
            IFS=' ' read -ra RULER_SEQ_LENGTHS <<< "$2"
            shift 2
            ;;
        --ruler_num_samples)
            RULER_NUM_SAMPLES="$2"
            shift 2
            ;;
        --ruler_gen_data)
            RULER_GEN_DATA="$2"
            shift 2
            ;;
        --num_shots)
            GSM8K_NUM_SHOTS="$2"
            shift 2
            ;;
        --cot_type)
            GSM8K_COT_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 <subcommand|--dataset ...>"
            echo ""
            echo "Subcommands (preset experiment suites):"
            echo "  overview       LongBench overview (budget 1024, 3 models) + RULER overview (4k-64k, 2 models)"
            echo "  budget         LongBench budget sweep (128-1024, 2 models) + RULER budget (64k budgets 128-4096, 2 models)"
            echo "  full           overview + budget, all at once"
            echo ""
            echo "Or use --dataset directly:"
            echo "  Required:"
            echo "    --dataset         gsm8k | longbench | longbench-v2 | ruler | all"
            echo "    --model           model name/path"
            echo "    --cache_sizes     space-separated cache sizes, e.g. \"128 256 512 1024\""
            echo ""
            echo "  Optional (common):"
            echo "    --max_samples     max samples per dataset    (default: 500)"
            echo "    --window_size     CakeKV window size         (default: 32)"
            echo "    --gamma           CakeKV gamma               (default: 200.0)"
            echo "    --tau1            CakeKV tau1                (default: 1.0)"
            echo "    --tau2            CakeKV tau2                (default: 1.0)"
            echo "    --compress        enable CAKE compression    (default: true)"
            echo "    --cascading       enable cascading cache     (default: true)"
            echo "    --device          CUDA device id             (default: 0)"
            echo ""
            echo "  Optional (RULER):"
            echo "    --ruler_benchmark     benchmark name         (default: synthetic)"
            echo "    --ruler_seq_lengths   space-separated lengths  (default: \"4096\")"
            echo "    --ruler_num_samples   samples per task       (default: 50)"
            echo "    --ruler_gen_data      generate data first    (default: false)"
            echo ""
            echo "  Optional (GSM8K):"
            echo "    --num_shots       few-shot count             (default: 8)"
            echo "    --cot_type        CoT prompt type             (default: gsm8k-cot)"
            echo ""
            echo "  Optional (LongBench):"
            echo "    --pred_name       pred output subfolder      (default: pred)"
            echo ""
            echo "Examples:"
            echo "  $0 full"
            echo "  $0 --dataset gsm8k --model deepseek --cache_sizes \"360\""
            echo "  $0 --dataset longbench --model qwen --cache_sizes \"128 256 512\" --max_samples 200"
            exit 1
            ;;
    esac
done

# ======================== Validation ========================
if [[ "$DATASET" == "__help__" ]]; then
    echo "Usage: $0 <subcommand|--dataset ...>"
    echo ""
    echo "Subcommands (preset experiment suites):"
    echo "  overview       LongBench overview (budget 1024, 3 models) + RULER overview (4k-64k, 2 models)"
    echo "  budget         LongBench budget sweep (128-1024, 2 models) + RULER budget (64k, budgets 128-4096, 2 models)"
    echo "  full           overview + budget, all at once"
    echo ""
    echo "Or use --dataset directly:"
    echo "  Required:"
    echo "    --dataset         gsm8k | longbench | longbench-v2 | ruler | all"
    echo "    --model           model name/path"
    echo "    --cache_sizes     space-separated cache sizes, e.g. \"128 256 512 1024\""
    echo ""
    echo "  Optional (common):"
    echo "    --max_samples     max samples per dataset    (default: 500)"
    echo "    --window_size     CakeKV window size         (default: 32)"
    echo "    --gamma           CakeKV gamma               (default: 200.0)"
    echo "    --tau1            CakeKV tau1                (default: 1.0)"
    echo "    --tau2            CakeKV tau2                (default: 1.0)"
    echo "    --compress        enable CAKE compression    (default: true)"
    echo "    --cascading       enable cascading cache     (default: true)"
    echo "    --device          CUDA device id             (default: 0)"
    echo ""
    echo "  Optional (RULER):"
    echo "    --ruler_benchmark     benchmark name         (default: synthetic)"
    echo "    --ruler_seq_lengths   space-separated lengths  (default: \"4096\")"
    echo "    --ruler_num_samples   samples per task       (default: 50)"
    echo "    --ruler_gen_data      generate data first    (default: false)"
    echo ""
    echo "  Optional (GSM8K):"
    echo "    --num_shots       few-shot count             (default: 8)"
    echo "    --cot_type        CoT prompt type             (default: gsm8k-cot)"
    echo ""
    echo "  Optional (LongBench):"
    echo "    --pred_name       pred output subfolder      (default: pred)"
    echo ""
    echo "  Allowed dataset-model combinations:"
    echo "    longbench:   llama, qwen, glm"
    echo "    ruler:       llama, qwen"
    echo "    gsm8k:       deepseek"
    echo "    longbench-v2: qwen"
    echo ""
    echo "  Raw HuggingFace paths (e.g. \"org/model-name\") are also accepted."
    echo ""
    echo "  Examples:"
    echo "    $0 full"
    echo "    $0 --dataset gsm8k --model deepseek --cache_sizes \"360\""
    echo "    $0 --dataset longbench --model qwen --cache_sizes \"128 256 512\" --max_samples 200"
    exit 0
fi

if [[ -n "$SUBCOMMAND" ]]; then
    # Subcommands skip --dataset/--model/--cache_sizes validation
    :
elif [[ -z "$DATASET" ]]; then
    echo "Error: --dataset is required (gsm8k | longbench | longbench-v2 | ruler | all)"
    exit 1
elif [[ -z "$MODEL" ]]; then
    echo "Error: --model is required"
    exit 1
elif [[ ${#CACHE_SIZES[@]} -eq 0 ]]; then
    echo "Error: --cache_sizes is required (space-separated list, e.g. '128 256 512 1024')"
    exit 1
fi

# Validate dataset name (skip if using subcommand)
if [[ -z "$SUBCOMMAND" ]]; then
    VALID_DATASETS=("gsm8k" "longbench" "longbench-v2" "ruler" "all")
    if [[ ! " ${VALID_DATASETS[*]} " =~ " ${DATASET} " ]]; then
        echo "Error: --dataset must be one of: gsm8k, longbench, longbench-v2, ruler, all"
        exit 1
    fi
fi

# ======================== Unified model name mapping ========================
# User passes a simple canonical name (llama / qwen / glm / deepseek).
# Each benchmark picks the correct variant automatically.
#
# Allowed dataset-model combinations:
#   longbench:   llama, qwen (普通版本), glm
#   ruler:       llama, qwen (1m版本)
#   gsm8k:       deepseek
#   longbench-v2: qwen (1m版本)
#
# For unsupported combos, the script will error out with a clear message.

declare -A MODEL_MAP
# Format: MODEL_MAP["dataset:canonical"]="benchmark-internal-name"
# LongBench: llama, qwen(普通), glm
MODEL_MAP["longbench:llama"]="llama3.1-8b-128k"
MODEL_MAP["longbench:qwen"]="qwen2.5-7b-instruct"
MODEL_MAP["longbench:glm"]="glm-4-9b-chat-1m"
# RULER: llama, qwen(1m)
MODEL_MAP["ruler:llama"]="llama-3.1-8b"
MODEL_MAP["ruler:qwen"]="qwen-2.5-7b-1m"
# GSM8K: deepseek
MODEL_MAP["gsm8k:deepseek"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# LongBench-v2: qwen(1m)
MODEL_MAP["longbench-v2:qwen"]="Qwen2.5-7B-Instruct-1M"

# Also accept raw HuggingFace paths for all datasets (passthrough)
# e.g. --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" works on any dataset,
# and --model "Qwen/Qwen2.5-7B-Instruct-1M" works as-is.

resolve_model_name() {
    local ds=$1
    local canonical=$2
    local key="${ds}:${canonical}"
    local result="${MODEL_MAP[$key]:-}"

    if [[ -z "$result" ]]; then
        # Check if it looks like a raw path (contains '/')
        if [[ "$canonical" == */* ]]; then
            result="$canonical"
        else
            echo "ERROR: unsupported dataset-model combination: dataset='$ds', model='$canonical'" >&2
            echo "" >&2
            echo "Allowed combinations:" >&2
            echo "  longbench:   llama, qwen, glm" >&2
            echo "  ruler:       llama, qwen" >&2
            echo "  gsm8k:       deepseek" >&2
            echo "  longbench-v2: qwen" >&2
            echo "" >&2
            echo "Or pass a raw HuggingFace path (e.g. 'org/model-name')" >&2
            exit 1
        fi
    fi
    echo "$result"
}

# ======================== Helper: build compress flags ========================
build_compress_flags() {
    local flags=""
    if [[ "$COMPRESS" == "true" ]]; then
        flags="$flags --compress"
    fi
    if [[ "$CASCADING" == "true" ]]; then
        flags="$flags --cascading"
    fi
    echo "$flags"
}

# ======================== GSM8K ========================
run_gsm8k() {
    local cache_size=$1
    local gsm8k_model=$(resolve_model_name "gsm8k" "$MODEL")
    local model_tag=$(echo "$gsm8k_model" | tr '/' '_')
    local log_dir="${EXPERIMENTS_DIR}/GSM8K/log/CakeKV/${model_tag}/cache${cache_size}"
    local save_dir="${EXPERIMENTS_DIR}/GSM8K/results/CakeKV/${model_tag}/cache${cache_size}"

    echo "============================================"
    echo "[GSM8K] canonical=$MODEL  resolved=$gsm8k_model  cache_size=$cache_size"
    echo "============================================"

    mkdir -p "$log_dir" "$save_dir"

    cd "${EXPERIMENTS_DIR}/GSM8K"

    python -u pred_cake.py \
        --model "$gsm8k_model" \
        --save_dir "$save_dir" \
        --cache_size "$cache_size" \
        --window_size "$WINDOW_SIZE" \
        --gamma "$GAMMA" \
        $(build_compress_flags) \
        --num_shots "$GSM8K_NUM_SHOTS" \
        --cot_type "$GSM8K_COT_TYPE" \
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
    local cache_size=$1
    local longbench_tasks=${2:-}
    local lb_model=$(resolve_model_name "longbench" "$MODEL")

    echo "============================================"
    echo "[LongBench] canonical=$MODEL  resolved=$lb_model  cache_size=$cache_size  max_samples=$MAX_SAMPLES"
    echo "============================================"

    cd "${EXPERIMENTS_DIR}/LongBench"

    if [ -n "$longbench_tasks" ]; then
        python pred_cake.py \
            --model "$lb_model" \
            --pred_name "$LONGBENCH_PRED_NAME" \
            --device "$DEVICE" \
            --cache_size "$cache_size" \
            --window_size "$WINDOW_SIZE" \
            --tau1 "$TAU1" \
            --tau2 "$TAU2" \
            --gamma "$GAMMA" \
            --max_samples "$MAX_SAMPLES" \
            --tasks "$longbench_tasks" \
            $(build_compress_flags)
    else
        python pred_cake.py \
            --model "$lb_model" \
            --pred_name "$LONGBENCH_PRED_NAME" \
            --device "$DEVICE" \
            --cache_size "$cache_size" \
            --window_size "$WINDOW_SIZE" \
            --tau1 "$TAU1" \
            --tau2 "$TAU2" \
            --gamma "$GAMMA" \
            --max_samples "$MAX_SAMPLES" \
            $(build_compress_flags)
    fi

    # Evaluate all dataset results for this (cache_size, model) combination
    local pred_dir="pred_result/cache${cache_size}/${LONGBENCH_PRED_NAME}/${lb_model}"
    # if [[ -d "$pred_dir" ]]; then
    #     python eval.py --dir_path "$pred_dir"
    # else
    #     echo "[LongBench] WARNING: prediction dir not found: $pred_dir, skipping eval."
    # fi
}

# ======================== LongBench-v2 ========================
run_longbench_v2() {
    local cache_size=$1
    local lbv2_model=$(resolve_model_name "longbench-v2" "$MODEL")

    echo "============================================"
    echo "[LongBench-v2] canonical=$MODEL  resolved=$lbv2_model  cache_size=$cache_size"
    echo "============================================"

    cd "${EXPERIMENTS_DIR}/LongBench-v2"

    local save_dir="results/cache${cache_size}"

    python pred.py \
        --model "$lbv2_model" \
        --save_dir "$save_dir" \
        --device "$DEVICE" \
        --cache_size "$cache_size" \
        --window_size "$WINDOW_SIZE" \
        --tau1 "$TAU1" \
        --tau2 "$TAU2" \
        --gamma "$GAMMA" \
        $(build_compress_flags) \
        --cot

    # Evaluate (result.py reads from results/ dir by default)
    # Copy result file to a recognizable name
    local model_key=$(python -c "
import json, sys
sys.path.insert(0, '${EXPERIMENTS_DIR}/LongBench-v2')
from pred import _resolve_model_key
print(_resolve_model_key('${lbv2_model}'))
" 2>/dev/null || echo "$lbv2_model")

    local result_file="${save_dir}/${model_key}.jsonl"
    if [[ -f "$result_file" ]]; then
        # result.py looks at results/ dir; symlink or copy to there
        mkdir -p results
        cp "$result_file" "results/cache${cache_size}_${model_key}.jsonl"
        python result.py
    else
        echo "[LongBench-v2] WARNING: result file not found: $result_file, skipping eval."
    fi
}

# ======================== RULER ========================
run_ruler() {
    local cache_size=$1
    local ruler_model=$(resolve_model_name "ruler" "$MODEL")
    local local_method="${METHOD:-cake}"

    echo "============================================"
    echo "[RULER] canonical=$MODEL  resolved=$ruler_model  cache_size=$cache_size"
    echo "============================================"

    cd "${EXPERIMENTS_DIR}/RULER"

    # Source RULER config scripts (use subshell sourcing to avoid pollution)
    source "${EXPERIMENTS_DIR}/RULER/ruler_config_models.sh"
    source "${EXPERIMENTS_DIR}/RULER/ruler_config_tasks.sh"

    # Resolve the model short name -> full config
    local model_short="${ruler_model}"
    local model_config=$(MODEL_SELECT "${model_short}")
    IFS=":" read -r MODEL_ID MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE <<< "$model_config"
    if [[ -z "$MODEL_ID" ]]; then
        echo "[RULER] ERROR: Model '${model_short}' is not supported in ruler_config_models.sh"
        return 1
    fi

    # Resolve tasks for the benchmark
    local benchmark="${RULER_BENCHMARK}"
    declare -n TASKS=$benchmark
    if [[ -z "${TASKS:-}" ]]; then
        echo "[RULER] ERROR: Benchmark '${benchmark}' is not supported in ruler_config_tasks.sh"
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
        qwen)  data_short="qwen-2.5-7b-1m" ;;
    esac

    for MAX_SEQ_LENGTH in "${RULER_SEQ_LENGTHS[@]}"; do
        local clw="${cache_size}_${WINDOW_SIZE}"
        local results_dir="./ruler_eval_result/${local_method}/${model_short}/${benchmark}/${MAX_SEQ_LENGTH}"
        local data_dir="../../../../benchmarks/ruler/benchmark_root/${data_short}/${benchmark}/${MAX_SEQ_LENGTH}/data"
        local pred_dir="${results_dir}/pred/${clw}"

        mkdir -p "$data_dir" "$pred_dir"

        # Optionally generate data
        if [[ "$RULER_GEN_DATA" == "true" ]]; then
            echo "[RULER] Generating data for seq_len=$MAX_SEQ_LENGTH ..."
            for TASK in "${TASKS[@]}"; do
                python prepare.py \
                    --save_dir "$data_dir" \
                    --benchmark "$benchmark" \
                    --task "$TASK" \
                    --tokenizer_path "$TOKENIZER_PATH" \
                    --tokenizer_type "$TOKENIZER_TYPE" \
                    --max_seq_length "$MAX_SEQ_LENGTH" \
                    --model_template_type "$MODEL_TEMPLATE_TYPE" \
                    --num_samples "$RULER_NUM_SAMPLES"
            done
        fi

        # Run prediction for each task
        for TASK in "${TASK_LIST[@]}"; do
            echo "[RULER] seq_len=$MAX_SEQ_LENGTH  task=$TASK"
            python pred/cake_ruler.py \
                --model "$MODEL_ID" \
                --compress --cascading \
                --pred_name "pred_result" \
                --device "$DEVICE" \
                --cache_size "$cache_size" \
                --window_size "$WINDOW_SIZE" \
                --task "$TASK" \
                --data_dir "$data_dir" \
                --save_dir "$pred_dir" \
                --benchmark "$benchmark" \
                --server_type "$MODEL_FRAMEWORK" \
                --synthetic_len "$MAX_SEQ_LENGTH" \
                --limit "$RULER_NUM_SAMPLES"
        done

        # Evaluate
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
    echo "║ LongBench: budget=1024, models=llama/qwen/glm                            ║"
    echo "║   All 12 tasks: narrativeqa, qasper, 2wikimqa, musique, gov_report,      ║"
    echo "║     multi_news, triviaqa, samsum, passage_count, passage_retrieval_en,   ║"
    echo "║     lcc, repobench-p                                                     ║"
    echo "║                                                                          ║"
    echo "║ RULER (4k/8k/16k/32k/64k): budget=1024, models=llama/qwen               ║"
    echo "║   All tasks in config_tasks.sh                                           ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
}

print_budget_table() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║                           Accuracy: Budget                               ║"
    echo "╠════════════════════════════════════════════════════════════════════════════╣"
    echo "║ LongBench: budgets=128/256/512/1024, models=llama/qwen                   ║"
    echo "║   Tasks (4): narrativeqa, qasper, trec, lcc                              ║"
    echo "║                                                                          ║"
    echo "║ RULER (64k): budgets=128/384/1024/4096, models=llama/qwen                ║"
    echo "║   Tasks (4): niah_single_3, vt, fwe, qa_1                               ║"
    echo "║   Qwen extra: + cwe                                                      ║"
    echo "║                                                                          ║"
    echo "║ RULER (128k): budget=2048, model=qwen only                               ║"
    echo "║   Tasks (4): niah_single_1, niah_multiquery, vt, fwe                     ║"
    echo "║ RULER (192k): budget=4096, model=qwen only                               ║"
    echo "║   Tasks (4): niah_single_1, niah_multiquery, vt, fwe                     ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
}

# ======================== Subcommand helpers ========================
run_longbench_overview() {
    local saved_model="$MODEL"
    for m in llama qwen glm; do
        MODEL="$m"; export MODEL
        local sp="1024"
        echo ""
        echo "========== [LongBench Overview] model=$MODEL  budget=$sp =========="
        run_longbench "$sp"
    done
    MODEL="$saved_model"
}

run_ruler_overview() {
    local saved_model="$MODEL"
    for m in llama qwen; do
        MODEL="$m"
        local sp="1024"
        RULER_TASK_FILTER=()
        for seq in 4096 8192 16384 32768 65536; do
            RULER_SEQ_LENGTHS=($seq)
            echo ""
            echo "========== [RULER Overview] model=$MODEL  budget=$sp  seq=${seq} =========="
            run_ruler "$sp"
        done
    done
    MODEL="$saved_model"
    RULER_SEQ_LENGTHS=(4096 8192 16384 32768 65536)
}

run_longbench_budget() {
    local saved_model="$MODEL"
    local budget_tasks="narrativeqa,qasper,trec,lcc"
    for m in llama qwen; do
        MODEL="$m"
        for sp in 128 256 512 1024; do
            echo ""
            echo "========== [LongBench Budget] model=$MODEL  budget=$sp =========="
            run_longbench "$sp" "$budget_tasks"
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
        for sp in 128 384 1024 4096; do
            echo ""
            echo "========== [RULER Budget 64k] model=$MODEL  budget=$sp =========="
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
        case $seq_len in
            128k) RULER_SEQ_LENGTHS=(131072); local sp=2048 ;;
            192k) RULER_SEQ_LENGTHS=(196608); local sp=4096 ;;
        esac
        echo ""
        echo "========== [RULER Budget ${seq_len}] model=$MODEL  budget=$sp =========="
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
            for CS in "${CACHE_SIZES[@]}"; do
                run_gsm8k "$CS"
            done
            ;;
        longbench)
            for CS in "${CACHE_SIZES[@]}"; do
                run_longbench "$CS"
            done
            ;;
        longbench-v2)
            for CS in "${CACHE_SIZES[@]}"; do
                run_longbench_v2 "$CS"
            done
            ;;
        ruler)
            for CS in "${CACHE_SIZES[@]}"; do
                run_ruler "$CS"
            done
            ;;
    esac
}

# Helper: check if model is valid for a given dataset (exits silently, returns 0=ok 1=bad)
check_model_compat() {
    local ds=$1
    local canonical=$2
    local key="${ds}:${canonical}"
    if [[ -n "${MODEL_MAP[$key]:-}" ]]; then
        return 0
    fi
    if [[ "$canonical" == */* ]]; then
        return 0
    fi
    return 1
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
    echo "CakeKV Accuracy Runner"
    echo "  Dataset:      $DATASET"
    echo "  Model:        $MODEL"
    echo "  Cache sizes:  ${CACHE_SIZES[*]}"
    echo "  Max samples:  $MAX_SAMPLES"
    echo "  Window size:  $WINDOW_SIZE"
    echo "  Gamma:        $GAMMA"
    echo "  Compress:     $COMPRESS"
    echo "  Cascading:    $CASCADING"
    echo "  Device:       $DEVICE"
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