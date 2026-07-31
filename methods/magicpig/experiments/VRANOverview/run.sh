# VRAN Overview


# {
#     VRAMOverview:
#         InputLength: 1024
#             ExceptBudget: 64; 64 - 8 = 56; 56/1024=5.468%
#                 K_L: 9 150
#             ExceptBudget: 512; 512 - 8 = 504; 504/1024=49.2187%
#                 K_L: 7 200
#         InputLength: 65536
#             ExceptBudget: 512; 512 - 10 = 502; 502/65536=0.76599%
#                 K_L: 10 90
#             ExceptBudget: 8192; 8192 - 10 = 8182; 8182/65536=12.48%
#                 K_L: 8 135
# }

# bash VRAMOverview.sh llama-3.1-8b-Instruct 9 150 1024 2 64
# bash VRAMOverview.sh llama-3.1-8b-Instruct 7 200 1024 2 512
# bash VRAMOverview.sh llama-3.1-8b-Instruct 9 150 1024 4096 64
# bash VRAMOverview.sh llama-3.1-8b-Instruct 7 200 1024 4096 512
# bash VRAMOverview.sh llama-3.1-8b-Instruct 10 90 65536 2 512
# bash VRAMOverview.sh llama-3.1-8b-Instruct 8 135 65536 2 8192

bash VRAMOverview.sh qwen-2.5-7b-1m 9 150 1024 2 64
bash VRAMOverview.sh qwen-2.5-7b-1m 7 200 1024 2 512
bash VRAMOverview.sh qwen-2.5-7b-1m 9 150 1024 4096 64
bash VRAMOverview.sh qwen-2.5-7b-1m 7 200 1024 4096 512
bash VRAMOverview.sh qwen-2.5-7b-1m 10 90 65536 2 512    
bash VRAMOverview.sh qwen-2.5-7b-1m 8 135 65536 2 8192



