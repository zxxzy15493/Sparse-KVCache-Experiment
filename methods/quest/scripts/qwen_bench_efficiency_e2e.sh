cd "$(dirname -- "${BASH_SOURCE[0]}")"
BUDGET_POOL=('512' '1024' '2048' '4096' '102400') # 102400 is full cache version
CONTEXT_POOL=('8192' '16384' '32768')

for budget in "${BUDGET_POOL[@]}"
do
    for context in "${CONTEXT_POOL[@]}"
    do
        python3 bench_textgen.py --model_name Qwen --model_type qwen2 --dtype bfloat16 --seqlen $context --max_new_tokens 32 --token_budget $budget --iteration 4 \
        --model_path Qwen/Qwen2.5-7B-Instruct-1M \
        --dataset_path ../../../benchmarks/myinput.txt \
        --bench_type LATENCY
    done
done
