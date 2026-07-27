#!/bin/bash
set -euo pipefail
set -x

usage() {
    cat <<'USAGE'
Usage:
  ./run_model.sh [--model MODEL] [--compressor COMPRESSOR] [--budget BUDGET] [--device DEVICE] [--exp-name NAME]

Environment variables with the same uppercase names are also supported, for example:
  MODEL=qwen-2.5-7b COMPRESSOR=no_drop_lb BUDGET=512 ./run_model.sh

Common compressors:
  no_drop_lb
  no_drop_lb_topp
  no_drop_lb_32
  no_drop_lb_topp32
  pq_search
  original

Extra args after -- are passed through to vq_pred_budget.py.
USAGE
}

MODEL="${MODEL:-qwen-2.5-7b}"
COMPRESSOR="${COMPRESSOR:-no_drop_lb}"
BUDGET="${BUDGET:-1024}"
DEVICE="${DEVICE:-0}"
EXP_NAME="${EXP_NAME:-pro6000}"
SEED="${SEED:-4321}"
COMPRESS="${COMPRESS:-0.1}"
CORE_OFFSET="${CORE_OFFSET:-0}"
TOPK="${TOPK:-0.5}"
RECENT_RATIO="${RECENT_RATIO:-0.5}"
SINK_SIZE="${SINK_SIZE:-0}"
SUBVEC="${SUBVEC:-2}"
SUBBITS="${SUBBITS:-6}"
TOPR="${TOPR:-1}"
METRIC="${METRIC:-euc}"
GQA="${GQA:-True}"
MEAN_V_TRICK="${MEAN_V_TRICK:-False}"
MAX_ITER="${MAX_ITER:-0}"
MAX_CPU_IN_USE="${MAX_CPU_IN_USE:-24}"
DROP="${DROP:-0}"
RECENT_SIZE="${RECENT_SIZE:-0}"
PRESERVE_LAYER="${PRESERVE_LAYER:-0}"
THRESHOLD="${THRESHOLD:-100000}"
SCORE_FUNC="${SCORE_FUNC:-sum}"
KEYFORMER_MODE="${KEYFORMER_MODE:-0}"
FIXTHRESHOLD="${FIXTHRESHOLD:-0.9}"
PP_SIZE="${PP_SIZE:-1}"
PRED_SCRIPT="${PRED_SCRIPT:-vq_pred_budget.py}"
FIXBUDGET_FLAG="${FIXBUDGET_FLAG:---fixbudget}"

PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --compressor)
            COMPRESSOR="$2"
            shift 2
            ;;
        --budget)
            BUDGET="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --exp-name)
            EXP_NAME="$2"
            shift 2
            ;;
        --script)
            PRED_SCRIPT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            PASSTHROUGH_ARGS+=("$@")
            break
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

TOPP_SAVE_TOPK_VALUE="${TOPP_SAVE_TOPK:-}"
TOPK_SAVE_TOPP_VALUE="${TOPK_SAVE_TOPP:-}"
# if [[ "${COMPRESSOR}" == "no_drop_lb_topp" || "${COMPRESSOR}" == "no_drop_lb_topp32" || "${COMPRESSOR}" == "topp" ]]; then
#     TOPP_SAVE_TOPK_VALUE="${TOPP_SAVE_TOPK_VALUE:-${MODEL}_${COMPRESSOR}_threshold_${FIXTHRESHOLD}_budget_${BUDGET}}"
# fi

export CORE_OFFSET
export MAX_CPU_IN_USE
export RANDOM_SEED="${SEED}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export CUDA_VISIBLE_DEVICES="${DEVICE}"
export TOKENIZERS_PARALLELISM=false
export SUBVEC
export SUBBITS
export METRIC
if [[ -n "${TOPP_SAVE_TOPK_VALUE}" ]]; then
    export TOPP_SAVE_TOPK="${TOPP_SAVE_TOPK_VALUE}"
else
    unset TOPP_SAVE_TOPK
fi
if [[ -n "${TOPK_SAVE_TOPP_VALUE}" ]]; then
    export TOPK_SAVE_TOPP="${TOPK_SAVE_TOPP_VALUE}"
else
    unset TOPK_SAVE_TOPP
fi

python "${PRED_SCRIPT}" \
    --model "${MODEL}" \
    --compress_ratio "${COMPRESS}" \
    ${FIXBUDGET_FLAG} \
    --budget "${BUDGET}" \
    --fixthreshold "${FIXTHRESHOLD}" \
    --important_ratio "${TOPK}" \
    --recent_ratio "${RECENT_RATIO}" \
    --recent_size "${RECENT_SIZE}" \
    --drop_ratio "${DROP}" \
    --enable_vq_cache \
    --fp16 \
    --pp-size "${PP_SIZE}" \
    --sink-size "${SINK_SIZE}" \
    --exp_name "${EXP_NAME}" \
    --score_func "${SCORE_FUNC}" \
    --preserve_layer "${PRESERVE_LAYER}" \
    --keyformer_mode "${KEYFORMER_MODE}" \
    --compressor "${COMPRESSOR}" \
    --threshold "${THRESHOLD}" \
    --n_subvec_per_head "${SUBVEC}" \
    --n_subbits "${SUBBITS}" \
    --topr "${TOPR}" \
    --gqa "${GQA}" \
    --sparq_mean_v_trick "${MEAN_V_TRICK}" \
    --max_iter "${MAX_ITER}" \
    "${PASSTHROUGH_ARGS[@]}"
