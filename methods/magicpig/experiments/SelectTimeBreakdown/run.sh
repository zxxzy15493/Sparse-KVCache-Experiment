# timeBreakdown
# llama
# SISO(4k, 32) budget 128, 256, 512, 1024
# 128 - 48 = 80 1.953%
# bash timeBreak.sh llama3.1-8b-Instruct 9 110 4096 32 128
# 256 - 48 = 208 5.078%
# bash timeBreak.sh llama3.1-8b-Instruct 8 85 4096 32 256
# # # 512 - 48 = 464 11.328%
# bash timeBreak.sh llama3.1-8b-Instruct 8 135 4096 32 512
# # # 1024 - 48 = 976 23.828%
# bash timeBreak.sh llama3.1-8b-Instruct 7 120 4096 32 1024

# # SISO(64k, 32) budget 128, 384, 1024, 4096, 16k
# # 128 - 48 = 80 0.122%
# bash timeBreak.sh llama3.1-8b-Instruct 11 63 65536 32 128
# # # 384 - 48 = 336 0.5126%
# bash timeBreak.sh llama3.1-8b-Instruct 10 73 65536 32 384
# # 1024 - 48 = 976 1.489%
bash timeBreak.sh llama3.1-8b-Instruct 10 143 65536 32 1024
# # 4096 - 48 = 4048 6.17%  
# bash timeBreak.sh llama3.1-8b-Instruct 8 85 65536 32 4096
# # 16384 - 48 = 16336 24.9%
# bash timeBreak.sh llama3.1-8b-Instruct 7 120 65536 32 16384

# # timeBreakdown
# # qwen
# # SISO(4k, 32) budget 128, 256, 512, 1024
# # 128 - 48 = 80 1.953%
# bash timeBreak.sh qwen2.5-7b-1m 9 112 4096 32 128
# # 256 - 48 = 208 5.078%
# bash timeBreak.sh qwen2.5-7b-1m 8 88 4096 32 256
# # 512 - 48 = 464 11.328%
# bash timeBreak.sh qwen2.5-7b-1m 8 135 4096 32 512
# # 1024 - 48 = 976 23.828%
# bash timeBreak.sh qwen2.5-7b-1m 7 120 4096 32 1024

# # SISO(64k, 32) budget 128, 384, 1024, 4096, 16k
# # 128 - 48 = 80 0.122%
# bash timeBreak.sh qwen2.5-7b-1m 11 63 65536 32 128
# # # 384 - 48 = 336 0.5126%
# bash timeBreak.sh qwen2.5-7b-1m 10 73 65536 32 384
# # 1024 - 48 = 976 1.489%
# bash timeBreak.sh qwen2.5-7b-1m 10 143 65536 32 1024
# # 4096 - 48 = 4048 6.17%
# bash timeBreak.sh qwen2.5-7b-1m 8 88 65536 32 4096
# # 16384 - 48 = 16336 24.9%
# bash timeBreak.sh qwen2.5-7b-1m 7 120 65536 32 16384
