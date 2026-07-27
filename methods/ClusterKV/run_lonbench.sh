pushd accuracy/LongBench

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python mypred.py --model qwen2.5-7b-chat-32k  --cluster  > ../../log/qwen_cluster_0509_1.log 2>&1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python mypred.py --model llama3.1-8b-chat-32k  --cluster  > ../../log/llama_cluster_0509_1.log 2>&1
popd