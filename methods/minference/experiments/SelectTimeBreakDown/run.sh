# ，GPU
export CUDA_VISIBLE_DEVICES=1

# 
# bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 4096 512 1
# bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 8192 512 1
# bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 16384 512 1
# bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 32768 512 1
# bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 65536 32 1
# bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 81920 32 1 # 80k

# bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 98304 32 1 # 96k

# bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 114688 32 1 # 112k
# bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 131072 512 1
# bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 196608 32 1
# bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 262144 32 1


bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 4096  32
bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 8192  32
bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 16384  32
bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 32768  32
bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 65536  32
bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 131072 32
bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 196608 32
bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 262144 32



