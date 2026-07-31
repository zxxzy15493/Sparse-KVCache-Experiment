# VRAN Overview


echo "Usage: $0 <model_name> $1 <K> $2 <L> $3 <except_budget> $4 <Benchmark> $5 <task>"
######## llama-3.1-8b ########
## LongBench
# # 80 / 24878 = 0.0032
bash RECALLOverview.sh llama-3.1-8b 11 100 80 LongBench narrativeqa
# # 208 / 24878 = 0.0084
# bash RECALLOverview.sh llama-3.1-8b 10 90 208 LongBench narrativeqa
# # 464 / 24878 = 0.0186
# bash RECALLOverview.sh llama-3.1-8b 9 80 464 LongBench narrativeqa
# # 976 / 24878 = 0.0392
# bash RECALLOverview.sh llama-3.1-8b 9 120 976 LongBench narrativeqa

# # # 80 / 4577 = 0.0175
# bash RECALLOverview.sh llama-3.1-8b 9 80 80 LongBench qasper
# # 208 / 4577 = 0.0455
# bash RECALLOverview.sh llama-3.1-8b 9 125 208 LongBench qasper
# # 464 / 4577 = 0.1013
# bash RECALLOverview.sh llama-3.1-8b 8 120 464 LongBench qasper
# # 976 / 4577 = 0.2129
# bash RECALLOverview.sh llama-3.1-8b 7 100 976 LongBench qasper

# # ############### qwen-2.5-7b ###############
# # ## LongBench
# # 80 / 24808 = 0.0032
# bash RECALLOverview.sh qwen-2.5-7b 11 100 80 LongBench narrativeqa
# # 208 / 24808 = 0.0084
# bash RECALLOverview.sh qwen-2.5-7b 10 90 208 LongBench narrativeqa
# # 464 / 24808 = 0.0187
# bash RECALLOverview.sh qwen-2.5-7b 9 80 464 LongBench narrativeqa
# # 976 / 24808 = 0.0393
# bash RECALLOverview.sh qwen-2.5-7b 9 120 976 LongBench narrativeqa

# # 80 / 4680 = 0.0171
# bash RECALLOverview.sh qwen-2.5-7b 9 80 80 LongBench qasper
# # 208 / 4680 = 0.0444
# bash RECALLOverview.sh qwen-2.5-7b 9 125 208 LongBench qasper
# # 464 / 4680 = 0.0991
# bash RECALLOverview.sh qwen-2.5-7b 8 120 464 LongBench qasper
# # 976 / 4680 = 0.2085 
# bash RECALLOverview.sh qwen-2.5-7b 7 100 976 LongBench qasper





######## llama-3.1-8b ########

## RULER
# bash RECALLOverview.sh llama-3.1-8b 11 65 80 synthetic niah_single_3 ##
# bash RECALLOverview.sh llama-3.1-8b 10 75 336 synthetic niah_single_3 ##
# bash RECALLOverview.sh llama-3.1-8b 10 130 976 synthetic niah_single_3 ##
# bash RECALLOverview.sh llama-3.1-8b 8 85 4048 synthetic niah_single_3 ##

# bash RECALLOverview.sh llama-3.1-8b 11 60 80 synthetic vt ##
# bash RECALLOverview.sh llama-3.1-8b 10 70 336 synthetic vt ##
# bash RECALLOverview.sh llama-3.1-8b 10 120 976 synthetic vt ##
# bash RECALLOverview.sh llama-3.1-8b 9 150 4048 synthetic vt ##

# bash RECALLOverview.sh llama-3.1-8b 11 60 80 synthetic fwe ##
# bash RECALLOverview.sh llama-3.1-8b 10 70 336 synthetic fwe ##k
# bash RECALLOverview.sh llama-3.1-8b 10 125 976 synthetic fwe ##
# bash RECALLOverview.sh llama-3.1-8b 8 85 4048 synthetic fwe ##




# ############### qwen-2.5-7b-Instruct-1M ###############
# ## RULER
# bash RECALLOverview.sh qwen-2.5-7b-1m 11 70 80 synthetic niah_single_3 ##
# bash RECALLOverview.sh qwen-2.5-7b-1m 10 80 336 synthetic niah_single_3 ## 
# bash RECALLOverview.sh qwen-2.5-7b-1m 10 140 976 synthetic niah_single_3 ##
# bash RECALLOverview.sh qwen-2.5-7b-1m 8 88 4048 synthetic niah_single_3 ##

# bash RECALLOverview.sh qwen-2.5-7b-1m 11 70 80 synthetic vt ##
# bash RECALLOverview.sh qwen-2.5-7b-1m 10 80 336 synthetic vt ###
# bash RECALLOverview.sh qwen-2.5-7b-1m 10 140 976 synthetic vt ##
# bash RECALLOverview.sh qwen-2.5-7b-1m 8 88 4048 synthetic vt ##

# bash RECALLOverview.sh qwen-2.5-7b-1m 11 70 80 synthetic fwe ##
# bash RECALLOverview.sh qwen-2.5-7b-1m 10 80 336 synthetic fwe ##
# bash RECALLOverview.sh qwen-2.5-7b-1m 10 140 976 synthetic fwe ##
# bash RECALLOverview.sh qwen-2.5-7b-1m 8 88 4048 synthetic fwe ##







