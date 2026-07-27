# # EfficencyOverview
# bash efficencyOverview.sh llama3.1-8b-Instruct 7 120 4096 32 1024
# bash efficencyOverview.sh llama3.1-8b-Instruct 8 135 8192 32 1024
# bash efficencyOverview.sh llama3.1-8b-Instruct 8 85 16384 32 1024
# bash efficencyOverview.sh llama3.1-8b-Instruct 9 110 32768 32 1024
# bash efficencyOverview.sh llama3.1-8b-Instruct 10 125 65536 32 1024 
# bash efficencyOverview.sh llama3.1-8b-Instruct 10 90 131072 32 1024


# bash efficencyOverview.sh qwen2.5-7b-1m 7 120 4096 32 1024
# bash efficencyOverview.sh qwen2.5-7b-1m 8 135 8192 32 1024
# bash efficencyOverview.sh qwen2.5-7b-1m 8 88 16384 32 1024
# bash efficencyOverview.sh qwen2.5-7b-1m 9 110 32768 32 1024
# bash efficencyOverview.sh qwen2.5-7b-1m 10 143 65536 32 1024
# bash efficencyOverview.sh qwen2.5-7b-1m 10 95 131072 32 1024



# # EfficencyBudget
# # llama
# # SISO(4k, 32) budget 128, 256, 512, 1024
# # 128 - 48 = 80 1.953%
# bash efficencyBudget.sh llama3.1-8b-Instruct 9 110 4096 32 128
# # 256 - 48 = 208 5.078%
# bash efficencyBudget.sh llama3.1-8b-Instruct 8 85 4096 32 256
# # # 512 - 48 = 464 11.328%
# bash efficencyBudget.sh llama3.1-8b-Instruct 8 135 4096 32 512
# # # 1024 - 48 = 976 23.828%
# bash efficencyBudget.sh llama3.1-8b-Instruct 7 120 4096 32 1024

# # # SISO(64k, 32) budget 128, 384, 1024, 4096, 16k
# # # 128 - 48 = 80 0.122%
# bash efficencyBudget.sh llama3.1-8b-Instruct 11 65 65536 32 128
# # # # 384 - 48 = 336 0.5126%
# bash efficencyBudget.sh llama3.1-8b-Instruct 10 73 65536 32 384
# # 1024 - 48 = 976 1.489%
# bash efficencyBudget.sh llama3.1-8b-Instruct 10 125 65536 32 1024
# # 4096 - 48 = 4048 6.17%  
# bash efficencyBudget.sh llama3.1-8b-Instruct 8 85 65536 32 4096
# # 16384 - 48 = 16336 24.9%
# bash efficencyBudget.sh llama3.1-8b-Instruct 7 110 65536 32 16384



# # EfficencyBudget
# # qwen
# # SISO(4k, 32) budget 128, 256, 512, 1024
# # 128 - 48 = 80 1.953%
# bash efficencyBudget.sh qwen2.5-7b-1m 9 112 4096 32 128
# # 256 - 48 = 208 5.078%
# bash efficencyBudget.sh qwen2.5-7b-1m 8 88 4096 32 256
# # 512 - 48 = 464 11.328%
# bash efficencyBudget.sh qwen2.5-7b-1m 8 135 4096 32 512
# # 1024 - 48 = 976 23.828%
# bash efficencyBudget.sh qwen2.5-7b-1m 7 120 4096 32 1024

# # SISO(64k, 32) budget 128, 384, 1024, 4096, 16k
# # 128 - 48 = 80 0.122%
# bash efficencyBudget.sh qwen2.5-7b-1m 11 70 65536 32 128
# # 384 - 48 = 336 0.5126%
# bash efficencyBudget.sh qwen2.5-7b-1m 10 75 65536 32 384
# # 1024 - 48 = 976 1.489%
# bash efficencyBudget.sh qwen2.5-7b-1m 10 143 65536 32 1024
# # 4096 - 48 = 4048 6.17%
# bash efficencyBudget.sh qwen2.5-7b-1m 8 88 65536 32 4096
# # 16384 - 48 = 16336 24.9%
# bash efficencyBudget.sh qwen2.5-7b-1m 7 115 65536 32 16384


