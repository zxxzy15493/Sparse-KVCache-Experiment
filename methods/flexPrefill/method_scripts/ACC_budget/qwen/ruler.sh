SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
METHODS_ROOT=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)
FLEX_ROOT="$METHODS_ROOT/flexPrefill"
cd "$FLEX_ROOT"
export PYTHONPATH="$FLEX_ROOT:${PYTHONPATH:-}"


for p in 0.8 0.85 0.95 0.9;do
    python "$FLEX_ROOT/experiments/benchmark/budget_ruler_single.py" \
        --model Qwen/Qwen2.5-7B-Instruct-1M \
        --chat \
        --task ruler \
        --attention flex_prefill \
        --tag flex_prefill \
        --p ${p}
done
