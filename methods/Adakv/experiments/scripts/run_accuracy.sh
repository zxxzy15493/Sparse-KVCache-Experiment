

set -euo pipefail
cd "$(dirname "$0")"
METHODS_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="${METHODS_DIR}:${PYTHONPATH:-}"
# ======================== Paths ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${EXPERIMENTS_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/..:${PROJECT_ROOT}:${EXPERIMENTS_DIR}:${PYTHONPATH:-}"

# ======================== Default Configuration ========================
# --- Common parameters ---
DATASET=""
MODEL=""
CACHE_SIZES=()      # space-separated list, e.g. "128 256 512 1024" (maps to max_capacity_prompts)
MAX_SAMPLES=500     # max samples per dataset (longbench / ruler)
WINDOW_SIZE=32
DEVICE=0

# --- HeadKV method parameters ---
METHOD="AdativeKV"           # HeadKV method (ReasonKV / AdativeKV) - auto-set by dataset
HEAD_CHOICE="random"      # head_choice (reason / random) - auto-set by method
BETA=1.5
TEMP=1.0
KERNEL_SIZE=7
SKIP=0
NORMALIZE=false
POOLING="maxpool"
FLOOR=0.2

# --- RULER specific ---
RULER_BENCHMARK="synthetic"
RULER_SEQ_LENGTHS=(4096)
RULER_NUM_SAMPLES=50
RULER_TASK_FILTER=()       # empty = run all tasks; set to array of task names to filter

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

# --- GSM8K specific ---
GSM8K_NUM_SHOTS=8
GSM8K_MAX_NEW_TOKENS=10000

# ======================== Argument Parsing ========================
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h|help)
            echo "Usage: $0 <subcommand|--dataset> [options]"
            echo ""
            echo "Subcommands (preset experiment suites):"
            echo "  overview       LongBench overview (budget 1024, 3 models) + RULER overview (4k-64k, 2 models)"
            echo "  budget         LongBench budget sweep (budgets 128-1024, 2 models) + RULER budget (64k, budgets 128-4096, 2 models)"
            echo "  full           overview + budget, all at once"
            echo ""
            echo "Or use --dataset directly:"
            echo "  Required:"
            echo "    --dataset         gsm8k | longbench | longbench-v2 | ruler | all"
            echo "    --model           model name (llama/qwen/glm/deepseek) or HF path"
            echo "    --cache_sizes     space-separated budgets, e.g. \"128 256 512 1024\""
            echo ""
            echo "  Optional (common):"
            echo "    --max_samples     max samples per dataset      (default: 500)"
            echo "    --window_size     window size                  (default: 32)"
            echo "    --device          CUDA device id               (default: 0)"
            echo "    --beta            HeadKV beta                  (default: 1.5)"
            echo "    --temp            HeadKV temp                  (default: 1.0)"
            echo ""
            echo "  Optional (HeadKV advanced):"
            echo "    --method          ReasonKV | AdativeKV        (default: auto by dataset)"
            echo "    --head_choice     reason | random | copy      (default: auto by method)"
            echo "    --kernel_size     kernel size for pooling     (default: 7)"
            echo "    --pooling         pooling type                 (default: maxpool)"
            echo "    --floor           floor value                  (default: 0.2)"
            echo ""
            echo "  Optional (RULER):"
            echo "    --ruler_benchmark       benchmark name        (default: synthetic)"
            echo "    --ruler_seq_lengths     space-separated lengths (default: \"4096\")"
            echo "    --ruler_num_samples     samples per task      (default: 50)"
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
            echo "  $0 --dataset gsm8k --model deepseek --cache_sizes \"360\""
            echo "  $0 --dataset longbench --model qwen --cache_sizes \"256 1024\" --max_samples 500"
            exit 0
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
        --method)
            METHOD="$2"
            shift 2
            ;;
        --head_choice)
            HEAD_CHOICE="$2"
            shift 2
            ;;
        --beta)
            BETA="$2"
            shift 2
            ;;
        --temp)
            TEMP="$2"
            shift 2
            ;;
        --kernel_size)
            KERNEL_SIZE="$2"
            shift 2
            ;;
        --skip)
            SKIP="$2"
            shift 2
            ;;
        --normalize)
            NORMALIZE="$2"
            shift 2
            ;;
        --pooling)
            POOLING="$2"
            shift 2
            ;;
        --floor)
            FLOOR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
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
        --num_shots)
            GSM8K_NUM_SHOTS="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# ======================== Validation ========================
if [[ -n "$SUBCOMMAND" ]]; then
    # Subcommands (overview/budget/full) skip --dataset/--model/--cache_sizes validation
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

if [[ -z "$SUBCOMMAND" ]]; then
    VALID_DATASETS=("gsm8k" "longbench" "longbench-v2" "ruler" "all")
    if [[ ! " ${VALID_DATASETS[*]} " =~ " ${DATASET} " ]]; then
        echo "Error: --dataset must be one of: gsm8k, longbench, longbench-v2, ruler, all"
        exit 1
    fi
fi

# ======================== Unified model name mapping ========================
# Format: key = "dataset:canonical_name" -> value = "benchmark-internal-name"
declare -A MODEL_MAP

# LongBench (LonBench): llama, qwen(普通), glm
MODEL_MAP["longbench:llama"]="meta-llama/Llama-3.1-8B-Instruct"
MODEL_MAP["longbench:qwen"]="Qwen/Qwen2.5-7B-Instruct"
MODEL_MAP["longbench:glm"]="THUDM/glm-4-9b-chat-1m"

# RULER: llama, qwen(1m)  (uses short names from config_models.sh)
MODEL_MAP["ruler:llama"]="llama-3.1-8b"
MODEL_MAP["ruler:qwen"]="qwen-2.5-7b-1m"

# GSM8K: deepseek
MODEL_MAP["gsm8k:deepseek"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# LongBench-v2: qwen(1m)  (uses keys from config/model2path.json)
MODEL_MAP["longbench-v2:qwen"]="Qwen2.5-7B-Instruct-1M"

resolve_model_name() {
    local ds=$1
    local canonical=$2
    local key="${ds}:${canonical}"
    local result="${MODEL_MAP[$key]:-}"

    if [[ -z "$result" ]]; then
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

# ======================== Method selection helpers ========================
# Default methods for each dataset (can be overridden via --method / --head_choice)
get_default_method() {
    local ds=$1
    # For now, use ReasonKV as the primary method
    echo "ReasonKV"
}
get_default_head_choice() {
    local method=$1
    if [[ "$method" == "ReasonKV" ]]; then
        echo "reason"
    elif [[ "$method" == "AdativeKV" ]]; then
        echo "random"
    else
        echo "reason"
    fi
}

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

# ======================== GSM8K ========================
run_gsm8k() {
    local cache_size=$1
    local gsm8k_model=$(resolve_model_name "gsm8k" "$MODEL")
    local local_method="${METHOD:-$(get_default_method gsm8k)}"
    local local_head_choice="${HEAD_CHOICE:-$(get_default_head_choice $local_method)}"
    local model_tag=$(echo "$gsm8k_model" | tr '/' '_')
    local log_dir="${EXPERIMENTS_DIR}/GSM8K/log/${local_method}/${model_tag}/budget${cache_size}"
    local save_dir="${EXPERIMENTS_DIR}/GSM8K/results/${local_method}/${model_tag}/budget${cache_size}"

    echo "============================================"
    echo "[GSM8K] canonical=$MODEL  resolved=$gsm8k_model  cache_size=$cache_size  method=$local_method  head_choice=$local_head_choice"
    echo "============================================"

    mkdir -p "$log_dir" "$save_dir"

    cd "${EXPERIMENTS_DIR}/GSM8K"

    python -u pred_headkv.py \
        --model "$gsm8k_model" \
        --save_dir "$save_dir" \
        --method "$local_method" \
        --head_choice "$local_head_choice" \
        --max_capacity_prompts "$cache_size" \
        --beta "$BETA" \
        --temp "$TEMP" \
        --num_shots "$GSM8K_NUM_SHOTS" \
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

# ======================== LongBench (LonBench) ========================
run_longbench() {
    local cache_size=$1
    local longbench_tasks=${2:-}
    local lb_model=$(resolve_model_name "longbench" "$MODEL")
    local local_method="${METHOD:-$(get_default_method longbench)}"
    local local_head_choice="${HEAD_CHOICE:-$(get_default_head_choice $local_method)}"

    echo "============================================"
    echo "[LongBench] canonical=$MODEL  resolved=$lb_model  cache_size=$cache_size  max_samples=$MAX_SAMPLES  method=$local_method  head_choice=$local_head_choice"
    echo "============================================"

    cd "${EXPERIMENTS_DIR}/LonBench"

    local save_dir="./results/${local_method}/${local_head_choice}/base${cache_size}_beta${BETA}_temp${TEMP}"

    if [ -n "$longbench_tasks" ]; then
        python -u run_longbench.py \
            --model_path "$lb_model" \
            --method "$local_method" \
            --head_choice "$local_head_choice" \
            --max_capacity_prompts "$cache_size" \
            --beta "$BETA" \
            --temp "$TEMP" \
            --save_dir "$save_dir" \
            --use_cache True \
            --attn_implementation flash_attention_2 \
            --max_num_examples "$MAX_SAMPLES" \
            --sample_method topk \
            --tasks "$longbench_tasks"
    else
        python -u run_longbench.py \
            --model_path "$lb_model" \
            --method "$local_method" \
            --head_choice "$local_head_choice" \
            --max_capacity_prompts "$cache_size" \
            --beta "$BETA" \
            --temp "$TEMP" \
            --save_dir "$save_dir" \
            --use_cache True \
            --attn_implementation flash_attention_2 \
            --max_num_examples "$MAX_SAMPLES" \
            --sample_method topk
    fi
}

# ======================== LongBench-v2 ========================
run_longbench_v2() {
    local cache_size=$1
    local lbv2_model=$(resolve_model_name "longbench-v2" "$MODEL")
    local local_method="${METHOD:-$(get_default_method longbench-v2)}"
    local local_head_choice="${HEAD_CHOICE:-$(get_default_head_choice $local_method)}"

    echo "============================================"
    echo "[LongBench-v2] canonical=$MODEL  resolved=$lbv2_model  cache_size=$cache_size  method=$local_method  head_choice=$local_head_choice"
    echo "============================================"

    cd "${EXPERIMENTS_DIR}/LongBench-v2"

    local save_dir="results/${local_method}_${local_head_choice}_base${cache_size}_beta${BETA}_temp${TEMP}"

    python -u pred.py \
        --model "$lbv2_model" \
        --save_dir "$save_dir" \
        --device "$DEVICE" \
        --window_size "$WINDOW_SIZE" \
        --max_capacity_prompts "$cache_size" \
        --head_choice "$local_head_choice" \
        --beta "$BETA" \
        --temp "$TEMP" \
        --kernel_size "$KERNEL_SIZE" \
        --skip "$SKIP" \
        $( [[ "$NORMALIZE" == "true" ]] && echo "--normalize" || echo "" ) \
        --pooling "$POOLING" \
        --floor "$FLOOR" \
        --cot

    # Evaluate
    local model_key=$(python -c "
import json, sys
sys.path.insert(0, '${EXPERIMENTS_DIR}/LongBench-v2')
from pred import _resolve_model_key
print(_resolve_model_key('${lbv2_model}'))
" 2>/dev/null || echo "$lbv2_model")

    local result_file="${save_dir}/${model_key}.jsonl"
    if [[ -f "$result_file" ]]; then
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
    local local_method="${METHOD:-$(get_default_method ruler)}"
    local local_head_choice="${HEAD_CHOICE:-$(get_default_head_choice $local_method)}"

    echo "============================================"
    echo "[RULER] canonical=$MODEL  resolved=$ruler_model  cache_size=$cache_size  method=$local_method  head_choice=$local_head_choice"
    echo "============================================"

    cd "${EXPERIMENTS_DIR}/RULER"

    # Source RULER config scripts
    source "${EXPERIMENTS_DIR}/RULER/config_models.sh"
    source "${EXPERIMENTS_DIR}/RULER/config_tasks.sh"

    # Resolve model short name -> full config
    local model_short="${ruler_model}"
    local model_config=$(MODEL_SELECT "${model_short}" "" "")
    IFS=":" read -r MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE <<< "$model_config"
    if [[ -z "$MODEL_PATH" ]]; then
        echo "[RULER] ERROR: Model '${model_short}' is not supported in config_models.sh"
        return 1
    fi

    # Resolve tasks for the benchmark
    local benchmark="${RULER_BENCHMARK}"
    declare -n TASKS=$benchmark
    if [[ -z "${TASKS:-}" ]]; then
        echo "[RULER] ERROR: Benchmark '${benchmark}' is not supported in config_tasks.sh"
        return 1
    fi

    local max_examples="${MAX_SAMPLES}"
    if [[ "$MAX_SAMPLES" -lt "$RULER_NUM_SAMPLES" ]]; then
        max_examples="$MAX_SAMPLES"
    else
        max_examples="$RULER_NUM_SAMPLES"
    fi

    # Map RULER model_short to actual data directory name
    local _data_model_name
    case "${model_short}" in
        llama3.1-8b) _data_model_name="llama-3.1-8b" ;;
        qwen2.5-7b)  _data_model_name="qwen-2.5-7b-1m" ;;
        *)           _data_model_name="${model_short}" ;;
    esac

    for MAX_SEQ_LENGTH in "${RULER_SEQ_LENGTHS[@]}"; do
        local clw="${cache_size}_${WINDOW_SIZE}"
        local results_dir="./ruler_eval_result/${local_method}/${model_short}/${benchmark}/${MAX_SEQ_LENGTH}"
        local data_dir="${EXPERIMENTS_DIR}/../../../../Sparse-KVCache-Experiment/benchmarks/ruler/benchmark_root/${_data_model_name}/${benchmark}/${MAX_SEQ_LENGTH}/data"
        local pred_dir="${results_dir}/pred/${clw}"
        mkdir -p "$data_dir" "$pred_dir"

        for TASK in "${TASKS[@]}"; do
            # Apply task filter if set (for budget runs)
            if [[ ${#RULER_TASK_FILTER[@]} -gt 0 ]]; then
                local _found=0
                for _ft in "${RULER_TASK_FILTER[@]}"; do
                    if [[ "$TASK" == "$_ft" ]]; then
                        _found=1
                        break
                    fi
                done
                if [[ $_found -eq 0 ]]; then
                    continue
                fi
            fi
            local task_data="${data_dir}/${TASK}/validation.jsonl"
            if [[ ! -f "$task_data" ]]; then
                echo "[RULER] WARNING: task data not found: $task_data, skipping."
                continue
            fi

            echo "[RULER] seq_len=$MAX_SEQ_LENGTH  task=$TASK  method=$local_method  head_choice=$local_head_choice"
            python run_ruler.py \
                --task "$TASK" \
                --task_data "$task_data" \
                --save_dir "$pred_dir" \
                --benchmark "$benchmark" \
                --model_name "$model_short" \
                --model_path "$MODEL_PATH" \
                --method "$local_method" \
                --max_capacity_prompts "$cache_size" \
                --head_choice "$local_head_choice" \
                --beta "$BETA" \
                --temp "$TEMP" \
                --window_size "$WINDOW_SIZE" \
                --seed 42 \
                --max_num_examples "$max_examples"
        done

        # python eval/evaluate.py \
        #     --data_dir "$pred_dir" \
        #     --benchmark "$benchmark"
    done
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

# ======================== Task definitions for informational display ========================

LB_OVERVIEW_TASKS="narrativeqa, qasper, 2wikimqa, musique, gov_report, multi_news, triviaqa, samsum, passage_count, passage_retrieval_en, lcc, repobench-p"
LB_BUDGET_TASKS="narrativeqa, qasper, trec, lcc"
RULER_OVERVIEW_TASKS="niah_single_1, niah_single_2, niah_single_3, niah_multikey_1, niah_multikey_2, niah_multikey_3, niah_multivalue, niah_multiquery, vt, cwe, fwe, qa_1, qa_2"
RULER_BUDGET_TASKS_COMMON="niah_single_3, vt, fwe, qa_1"
RULER_BUDGET_TASKS_QWEN_EXTRA="cwe"
RULER_BUDGET_LONGRANGE_TASKS="niah_single_1, niah_multiquery, vt, fwe"

print_overview_table() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                    Accuracy: Overview                        ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║ LongBench: budget=1024, models=llama/qwen/glm               ║"
    echo "║   Tasks (12): ${LB_OVERVIEW_TASKS}"
    echo "║                                                             ║"
    echo "║ RULER: budget=1024, models=llama/qwen, seq=4k/8k/16k/32k/64k║"
    echo "║   Tasks (13): ${RULER_OVERVIEW_TASKS}"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
}

print_budget_table() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║                           Accuracy: Budget                               ║"
    echo "╠════════════════════════════════════════════════════════════════════════════╣"
    echo "║ LongBench: budgets=128/256/512/1024, models=llama/qwen                   ║"
    echo "║   Tasks (4): ${LB_BUDGET_TASKS}"
    echo "║                                                                            ║"
    echo "║ RULER (64k): budgets=128/384/1024/4096, models=llama/qwen                 ║"
    echo "║   Tasks (4): ${RULER_BUDGET_TASKS_COMMON}"
    echo "║   Qwen extra: + ${RULER_BUDGET_TASKS_QWEN_EXTRA}"
    echo "║                                                                            ║"
    echo "║ RULER (128k): budget=2048, model=qwen only                                ║"
    echo "║   Tasks (4): ${RULER_BUDGET_LONGRANGE_TASKS}"
    echo "║ RULER (192k): budget=4096, model=qwen only                                ║"
    echo "║   Tasks (4): ${RULER_BUDGET_LONGRANGE_TASKS}"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
}

# ======================== Subcommand helpers ========================

run_longbench_overview() {
    # Table already printed by overview_run() if called from subcommand
    local _saved_max_samples="$MAX_SAMPLES"
    # Run LongBench overview: budget=1024, models=llama/qwen/glm
    for _model in "llama" "qwen" "glm"; do
        echo ""
        echo "========== [LongBench Overview] model=$_model  budget=1024 =========="
        MODEL="$_model"
        CACHE_SIZES=(1024)
        run_longbench "1024"
    done
    MAX_SAMPLES="$_saved_max_samples"
}

run_ruler_overview() {
    # No need to re-print table if called together
    local _saved_seq="${RULER_SEQ_LENGTHS[*]}"
    local _saved_max_samples="$MAX_SAMPLES"
    for _model in "llama" "qwen"; do
        RULER_SEQ_LENGTHS=(4096 8192 16384 32768 65536)
        echo ""
        echo "========== [RULER Overview] model=$_model  budget=1024  seqs=${RULER_SEQ_LENGTHS[*]} =========="
        MODEL="$_model"
        CACHE_SIZES=(1024)
        run_ruler "1024"
    done
    RULER_SEQ_LENGTHS=($_saved_seq)
    MAX_SAMPLES="$_saved_max_samples"
}

run_longbench_budget() {
    # Table already printed by budget_run() if called from subcommand
    local _saved_max_samples="$MAX_SAMPLES"
    local _budget_tasks="narrativeqa,qasper,trec,lcc"
    # Run LongBench budget: budgets=128/256/512/1024, models=llama/qwen
    for _cache in 128 256 512 1024; do
        for _model in "llama" "qwen"; do
            echo ""
            echo "========== [LongBench Budget] model=$_model  budget=$_cache =========="
            MODEL="$_model"
            CACHE_SIZES=("$_cache")
            run_longbench "$_cache" "$_budget_tasks"
        done
    done
    MAX_SAMPLES="$_saved_max_samples"
}

run_ruler_budget() {
    # No need to re-print table if called together
    local _saved_seq="${RULER_SEQ_LENGTHS[*]}"
    local _saved_max_samples="$MAX_SAMPLES"

    # ── Part 1: 64k, budgets=128/384/1024/4096, models=llama/qwen ──
    # Common tasks for both models
    RULER_TASK_FILTER=(niah_single_3 vt fwe qa_1)
    RULER_SEQ_LENGTHS=(65536)
    for _cache in 128 384 1024 4096; do
        for _model in "llama" "qwen"; do
            echo ""
            echo "========== [RULER Budget 64k] model=$_model  budget=$_cache =========="
            MODEL="$_model"
            CACHE_SIZES=("$_cache")
            run_ruler "$_cache"
        done
    done

    # ── Part 1b: 64k, qwen extra task (cwe) ──
    RULER_TASK_FILTER=(cwe)
    for _cache in 128 384 1024 4096; do
        echo ""
        echo "========== [RULER Budget 64k qwen+cwe] model=qwen  budget=$_cache =========="
        MODEL="qwen"
        CACHE_SIZES=("$_cache")
        run_ruler "$_cache"
    done

    # ── Part 2: qwen extra long-range (128k@2048, 192k@4096) ──
    RULER_TASK_FILTER=(niah_single_1 niah_multiquery vt fwe)
    for _model in "qwen"; do
        RULER_SEQ_LENGTHS=(131072)
        echo ""
        echo "========== [RULER Budget 128k] model=$_model  budget=2048 =========="
        MODEL="$_model"
        CACHE_SIZES=(2048)
        run_ruler "2048"

        RULER_SEQ_LENGTHS=(196608)
        echo ""
        echo "========== [RULER Budget 192k] model=$_model  budget=4096 =========="
        MODEL="$_model"
        CACHE_SIZES=(4096)
        run_ruler "4096"
    done

    # Reset filter
    RULER_TASK_FILTER=()
    RULER_SEQ_LENGTHS=($_saved_seq)
    MAX_SAMPLES="$_saved_max_samples"
}

overview_run() {
    print_overview_table
    run_longbench_overview
    run_ruler_overview
}

budget_run() {
    print_budget_table
    run_longbench_budget
    run_ruler_budget
}

full_run() {
    print_overview_table
    print_budget_table
    run_longbench_overview
    run_ruler_overview
    run_longbench_budget
    run_ruler_budget
}

# ======================== Main dispatch ========================
if [[ -n "$SUBCOMMAND" ]]; then
    case "$SUBCOMMAND" in
        overview)
            overview_run
            ;;
        budget)
            budget_run
            ;;
        ruler-budget)
            print_budget_table
            run_ruler_budget
            ;;
        full)
            full_run
            ;;
    esac
elif [[ -z "$DATASET" && -z "$MODEL" && ${#CACHE_SIZES[@]} -eq 0 ]]; then
    # No arguments → full auto run
    full_run
elif [[ "$DATASET" == "all" ]]; then
    # --dataset all: requires --model and --cache_sizes; only runs compatible datasets
    echo "============================================================"
    echo "HeadKV Accuracy Runner"
    echo "  Dataset:      $DATASET"
    echo "  Model:        $MODEL"
    echo "  Cache sizes:  ${CACHE_SIZES[*]}"
    echo "  Max samples:  $MAX_SAMPLES"
    echo "  Window size:  $WINDOW_SIZE"
    echo "  Method:       ${METHOD:-auto}"
    echo "  Head choice:  ${HEAD_CHOICE:-auto}"
    echo "  Beta/Temp:    $BETA / $TEMP"
    echo "============================================================"
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
    echo "============================================================"
    echo "HeadKV Accuracy Runner"
    echo "  Dataset:      $DATASET"
    echo "  Model:        $MODEL"
    echo "  Cache sizes:  ${CACHE_SIZES[*]}"
    echo "  Max samples:  $MAX_SAMPLES"
    echo "  Window size:  $WINDOW_SIZE"
    echo "  Method:       ${METHOD:-auto}"
    echo "  Head choice:  ${HEAD_CHOICE:-auto}"
    echo "  Beta/Temp:    $BETA / $TEMP"
    echo "============================================================"
    run_dataset "$DATASET"
fi

echo ""
echo "All done."