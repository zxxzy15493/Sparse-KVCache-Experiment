# VRAN Overview


# {
#     RECALLOverview:
#         LongBench:
#             narrativeqa: 
#                 mean:
#                     29958 
#                 llama3.1-8b-Instruct:
#                     80:
#                         K_L: 11 90
#                     208:
#                         K_L: 10 80
#                     464:
#                         K_L: 10 120
#                     976:
#                         K_L: 9 110
#             qasper: 
#                 mean
#                     5221.4
#                 llama3.1-8b-Instruct:
#                     80:
#                         K_L: 9 75
#                     208:
#                         K_L: 9 125
#                     464:
#                         K_L: 8 110
#                     976:
#                         K_L: 9 80
#         RULER:
#             niah_single_3: 65536
#                 llama3.1-8b-Instruct:
#                     80:
#                         K_L: 1_3
#                     336:
#                         K_L: 1_3
#                     976:
#                         K_L: 1_3
#                     4048:
#                         K_L: 1_3
#             vt: 65536
#                 llama3.1-8b-Instruct:
#                     80:
#                         K_L: 1_3
#                     336:
#                         K_L: 1_3
#                     976:
#                         K_L: 1_3
#                     4048:
#                         K_L: 1_3
#             fwe: 65536
#                 llama3.1-8b-Instruct:
#                     80:
#                         K_L: 1_3
#                     336:
#                         K_L: 1_3
#                     976:
#                         K_L: 1_3
#                     4048:
#                         K_L: 1_3  
# }
echo "Usage: $0 <model_name> $1 <K> $2 <L> $3 <except_budget> $4 <Benchmark> $5 <task>"
######## Llama3.1-8b-Instruct ########
## LongBench
# # 80 / 24878 = 0.0032
# bash RECALLOverview.sh llama3.1-8b-Instruct 11 100 80 LongBench narrativeqa
# # 208 / 24878 = 0.0084
# bash RECALLOverview.sh llama3.1-8b-Instruct 10 90 208 LongBench narrativeqa
# # 464 / 24878 = 0.0186
# bash RECALLOverview.sh llama3.1-8b-Instruct 9 80 464 LongBench narrativeqa
# # 976 / 24878 = 0.0392
# bash RECALLOverview.sh llama3.1-8b-Instruct 9 120 976 LongBench narrativeqa

# # # 80 / 4577 = 0.0175
# bash RECALLOverview.sh llama3.1-8b-Instruct 9 80 80 LongBench qasper
# # 208 / 4577 = 0.0455
# bash RECALLOverview.sh llama3.1-8b-Instruct 9 125 208 LongBench qasper
# # 464 / 4577 = 0.1013
# bash RECALLOverview.sh llama3.1-8b-Instruct 8 120 464 LongBench qasper
# # 976 / 4577 = 0.2129
# bash RECALLOverview.sh llama3.1-8b-Instruct 7 100 976 LongBench qasper

# # ############### qwen2.5-7b ###############
# # ## LongBench
# # 80 / 24808 = 0.0032
# bash RECALLOverview.sh qwen2.5-7b 11 100 80 LongBench narrativeqa
# # 208 / 24808 = 0.0084
# bash RECALLOverview.sh qwen2.5-7b 10 90 208 LongBench narrativeqa
# # 464 / 24808 = 0.0187
# bash RECALLOverview.sh qwen2.5-7b 9 80 464 LongBench narrativeqa
# # 976 / 24808 = 0.0393
# bash RECALLOverview.sh qwen2.5-7b 9 120 976 LongBench narrativeqa

# # 80 / 4680 = 0.0171
# bash RECALLOverview.sh qwen2.5-7b 9 80 80 LongBench qasper
# # 208 / 4680 = 0.0444
# bash RECALLOverview.sh qwen2.5-7b 9 125 208 LongBench qasper
# # 464 / 4680 = 0.0991
# bash RECALLOverview.sh qwen2.5-7b 8 120 464 LongBench qasper
# # 976 / 4680 = 0.2085 
# bash RECALLOverview.sh qwen2.5-7b 7 100 976 LongBench qasper





######## Llama3.1-8b-Instruct ########

## RULER
# bash RECALLOverview.sh llama3.1-8b-Instruct 11 65 80 synthetic niah_single_3 ##
# bash RECALLOverview.sh llama3.1-8b-Instruct 10 75 336 synthetic niah_single_3 ##
# bash RECALLOverview.sh llama3.1-8b-Instruct 10 130 976 synthetic niah_single_3 ##
# bash RECALLOverview.sh llama3.1-8b-Instruct 8 85 4048 synthetic niah_single_3 ##

# bash RECALLOverview.sh llama3.1-8b-Instruct 11 60 80 synthetic vt ##
# bash RECALLOverview.sh llama3.1-8b-Instruct 10 70 336 synthetic vt ##
# bash RECALLOverview.sh llama3.1-8b-Instruct 10 120 976 synthetic vt ##
# bash RECALLOverview.sh llama3.1-8b-Instruct 9 150 4048 synthetic vt ##

# bash RECALLOverview.sh llama3.1-8b-Instruct 11 60 80 synthetic fwe ##
# bash RECALLOverview.sh llama3.1-8b-Instruct 10 70 336 synthetic fwe ##k
# bash RECALLOverview.sh llama3.1-8b-Instruct 10 125 976 synthetic fwe ##
# bash RECALLOverview.sh llama3.1-8b-Instruct 8 85 4048 synthetic fwe ##




# ############### qwen2.5-7b-Instruct-1M ###############
# ## RULER
# bash RECALLOverview.sh qwen2.5-7b-Instruct-1M 11 70 80 synthetic niah_single_3 ##
# bash RECALLOverview.sh qwen2.5-7b-Instruct-1M 10 80 336 synthetic niah_single_3 ## 
# bash RECALLOverview.sh qwen2.5-7b-Instruct-1M 10 140 976 synthetic niah_single_3 ##
# bash RECALLOverview.sh qwen2.5-7b-Instruct-1M 8 88 4048 synthetic niah_single_3 ##

# bash RECALLOverview.sh qwen2.5-7b-Instruct-1M 11 70 80 synthetic vt ##
# bash RECALLOverview.sh qwen2.5-7b-Instruct-1M 10 80 336 synthetic vt ###
bash RECALLOverview.sh qwen2.5-7b-Instruct-1M 10 140 976 synthetic vt ##
# bash RECALLOverview.sh qwen2.5-7b-Instruct-1M 8 88 4048 synthetic vt ##

# bash RECALLOverview.sh qwen2.5-7b-Instruct-1M 11 70 80 synthetic fwe ##
# bash RECALLOverview.sh qwen2.5-7b-Instruct-1M 10 80 336 synthetic fwe ##
bash RECALLOverview.sh qwen2.5-7b-Instruct-1M 10 140 976 synthetic fwe ##
# bash RECALLOverview.sh qwen2.5-7b-Instruct-1M 8 88 4048 synthetic fwe ##







