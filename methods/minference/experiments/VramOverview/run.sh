
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash efficencyOverview.sh llama-3.1-8b 2 4096 2
bash efficencyOverview.sh llama-3.1-8b 2 8192 2
bash efficencyOverview.sh llama-3.1-8b 2 16384 2
bash efficencyOverview.sh llama-3.1-8b 2 32768 2
bash efficencyOverview.sh llama-3.1-8b 2 65536 2
bash efficencyOverview.sh llama-3.1-8b 2 131072 2
bash efficencyOverview.sh llama-3.1-8b 2 262144 2


bash efficencyOverview.sh qwen-2.5-7b 2 4096 2
bash efficencyOverview.sh qwen-2.5-7b 2 8192 2
bash efficencyOverview.sh qwen-2.5-7b 2 16384 2
bash efficencyOverview.sh qwen-2.5-7b 2 32768 2
bash efficencyOverview.sh qwen-2.5-7b 2 65536 2
bash efficencyOverview.sh qwen-2.5-7b 2 131072 2
bash efficencyOverview.sh qwen-2.5-7b 2 262144 2


