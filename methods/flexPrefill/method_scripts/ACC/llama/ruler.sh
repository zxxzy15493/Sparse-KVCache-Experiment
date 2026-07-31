SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
METHODS_ROOT=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)
FLEX_ROOT="$METHODS_ROOT/flexPrefill"
cd "$FLEX_ROOT"
export PYTHONPATH="$FLEX_ROOT:${PYTHONPATH:-}"

# py       run-debug ruler-debug 
#
python "$FLEX_ROOT/experiments/benchmark/ruler_single.py" \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --chat \
    --task ruler \
    --attention flex_prefill \
    --cfg block_size=128,flex_prefill_min_budget=512,flex_prefill_gamma=0.9,flex_prefill_tau=0.1 \
    --tag flex_prefill

#attentionflex_prefill
