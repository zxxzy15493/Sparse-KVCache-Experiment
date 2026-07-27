cd "$(dirname "$0")"
REPO_ROOT=$(cd ../../../.. && pwd)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
conda activate kvcache
# Input 4k, output 512, 4 warmup + 1 measure
python run_timing_test.py \
    --model llama3.1-8b-128k \
    --input_max_token 4096 \
    --warmup 9 \
    --measure 1

# Input 64k, decode timing uses fixed 32-step total
python run_timing_test.py \
    --model llama3.1-8b-128k \
    --input_max_token 65536 \
    --warmup 4 \
    --measure 1
