#!/bin/bash
# HeadKV timing test: ReasonKV and AdativeKV

cd "$(dirname "$0")"


python run_timing_test.py \
    --model llama3.1-8b-128k \
    --method ReasonKV \
    --input_max_token 4096 \
    --max_capacity_prompts 1024

echo ""
echo "===== Done 4k ReasonKV ====="
echo ""


python run_timing_test.py \
    --model llama3.1-8b-128k \
    --method ReasonKV \
    --input_max_token 65536 \
    --max_capacity_prompts 1024

echo ""
echo "===== Done 64k ReasonKV ====="
echo ""

