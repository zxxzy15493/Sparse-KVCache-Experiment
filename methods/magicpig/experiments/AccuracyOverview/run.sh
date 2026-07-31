#!/bin/bash

# =============================================================================
# Accuracy Experiment Commands
# Generated from method_configs/magicpig.yaml
# =============================================================================
#
# Usage: bash Accuracy.sh <model> <benchmark> <task> <num_examples> <max_len> <K> <L>
#
# Models:  llama-3.1-8b, qwen-2.5-7b, glm-4-9b-chat-1m
# Lengths: 4096, 8192, 16384, 32768, 65536
#
# Synthetic tasks (13 total):
#   niah_single_1, niah_single_2, niah_single_3,
#   niah_multikey_1, niah_multikey_2, niah_multikey_3,
#   niah_multiquery, niah_multivalue,
#   vt, cwe, fwe, qa_1, qa_2
# =============================================================================

# =============================================================================
# budget-1024 (main config)
# =============================================================================

# -------------------------------------------------------------------
# budget-1024 | llama-3.1-8b | LongBench
# -------------------------------------------------------------------
# bash Accuracy.sh llama-3.1-8b LongBench narrativeqa          -1 65536  9 120
# bash Accuracy.sh llama-3.1-8b LongBench qasper               -1 65536  7 100
# bash Accuracy.sh llama-3.1-8b LongBench 2wikimqa             -1 65536  7  75
# bash Accuracy.sh llama-3.1-8b LongBench musique              -1 65536  9 150
# bash Accuracy.sh llama-3.1-8b LongBench gov_report           -1 65536  8 120
# bash Accuracy.sh llama-3.1-8b LongBench multi_news           -1 65536  7 150
# bash Accuracy.sh llama-3.1-8b LongBench triviaqa             -1 65536  8 100
# bash Accuracy.sh llama-3.1-8b LongBench samsum               -1 65536  8 120
# bash Accuracy.sh llama-3.1-8b LongBench passage_count        -1 65536  8 100
# bash Accuracy.sh llama-3.1-8b LongBench passage_retrieval_en -1 65536  8 100
# bash Accuracy.sh llama-3.1-8b LongBench lcc                  -1 65536  7 150
# bash Accuracy.sh llama-3.1-8b LongBench repobench-p          -1 65536  8 100


# -------------------------------------------------------------------
# budget-1024 | qwen-2.5-7b | LongBench
# -------------------------------------------------------------------
# bash Accuracy.sh qwen-2.5-7b LongBench narrativeqa          -1 65536  9 120
# bash Accuracy.sh qwen-2.5-7b LongBench qasper               -1 65536  7 100
# bash Accuracy.sh qwen-2.5-7b LongBench 2wikimqa             -1 65536  7  75
# bash Accuracy.sh qwen-2.5-7b LongBench musique              -1 65536  9 150
# bash Accuracy.sh qwen-2.5-7b LongBench gov_report           -1 65536  8 120
# bash Accuracy.sh qwen-2.5-7b LongBench multi_news           -1 65536  7 150
# bash Accuracy.sh qwen-2.5-7b LongBench triviaqa             -1 65536  8 100
# bash Accuracy.sh qwen-2.5-7b LongBench samsum               -1 65536  8 120
# bash Accuracy.sh qwen-2.5-7b LongBench passage_count        -1 65536  8 100
# bash Accuracy.sh qwen-2.5-7b LongBench passage_retrieval_en -1 65536  8 100
# bash Accuracy.sh qwen-2.5-7b LongBench lcc                  -1 65536  7 150
# bash Accuracy.sh qwen-2.5-7b LongBench repobench-p          -1 65536  8 100 


# -------------------------------------------------------------------
# budget-1024 | glm-4-9b-chat-1m | LongBench
# -------------------------------------------------------------------
# bash Accuracy.sh glm-4-9b-chat-1m LongBench narrativeqa          -1 65536  9 120
# bash Accuracy.sh glm-4-9b-chat-1m LongBench qasper               -1 65536  7 100
# bash Accuracy.sh glm-4-9b-chat-1m LongBench 2wikimqa             -1 65536  7  75
# bash Accuracy.sh glm-4-9b-chat-1m LongBench musique              -1 65536  9 150
# bash Accuracy.sh glm-4-9b-chat-1m LongBench gov_report           -1 65536  8 120
# bash Accuracy.sh glm-4-9b-chat-1m LongBench multi_news           -1 65536  7 150
# bash Accuracy.sh glm-4-9b-chat-1m LongBench triviaqa             -1 65536  8 100
# bash Accuracy.sh glm-4-9b-chat-1m LongBench samsum               -1 65536  8 120
# bash Accuracy.sh glm-4-9b-chat-1m LongBench passage_count        -1 65536  8 100
# bash Accuracy.sh glm-4-9b-chat-1m LongBench passage_retrieval_en -1 65536  8 100
# bash Accuracy.sh glm-4-9b-chat-1m LongBench lcc                  -1 65536  7 150
# bash Accuracy.sh glm-4-9b-chat-1m LongBench repobench-p          -1 65536  8 100

# -------------------------------------------------------------------
# budget-1024 | llama-3.1-8b | Synthetic (ruler-4096)  K=7  L=120
# -------------------------------------------------------------------
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_1    -1 4096 7 120
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_2    -1 4096 7 120
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_3    -1 4096 7 120
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_1  -1 4096 7 120
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_2  -1 4096 7 120
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_3  -1 4096 7 120
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multiquery  -1 4096 7 120
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multivalue  -1 4096 7 120
# bash Accuracy.sh llama-3.1-8b Synthetic vt               -1 4096 7 120
# bash Accuracy.sh llama-3.1-8b Synthetic cwe              -1 4096 7 120
# bash Accuracy.sh llama-3.1-8b Synthetic fwe              -1 4096 7 120
# bash Accuracy.sh llama-3.1-8b Synthetic qa_1             -1 4096 7 120
# bash Accuracy.sh llama-3.1-8b Synthetic qa_2             -1 4096 7 120

# # -------------------------------------------------------------------
# # budget-1024 | llama-3.1-8b | Synthetic (ruler-8192)  K=8  L=135
# # -------------------------------------------------------------------
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_1    -1 8192 8 135
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_2    -1 8192 8 135
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_3    -1 8192 8 135
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_1  -1 8192 8 135
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_2  -1 8192 8 135
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_3  -1 8192 8 135
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multiquery  -1 8192 8 135
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multivalue  -1 8192 8 135
# bash Accuracy.sh llama-3.1-8b Synthetic vt               -1 8192 8 135
# bash Accuracy.sh llama-3.1-8b Synthetic cwe              -1 8192 8 135
# bash Accuracy.sh llama-3.1-8b Synthetic fwe              -1 8192 8 135
# bash Accuracy.sh llama-3.1-8b Synthetic qa_1             -1 8192 8 135
# bash Accuracy.sh llama-3.1-8b Synthetic qa_2             -1 8192 8 135

# # -------------------------------------------------------------------
# # budget-1024 | llama-3.1-8b | Synthetic (ruler-16384)  K=8  L=85
# # -------------------------------------------------------------------
bash Accuracy.sh llama-3.1-8b Synthetic niah_single_1    -1 16384 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_2    -1 16384 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_3    -1 16384 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_1  -1 16384 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_2  -1 16384 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_3  -1 16384 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multiquery  -1 16384 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multivalue  -1 16384 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic vt               -1 16384 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic cwe              -1 16384 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic fwe              -1 16384 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic qa_1             -1 16384 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic qa_2             -1 16384 8 85

# # -------------------------------------------------------------------
# # budget-1024 | llama-3.1-8b | Synthetic (ruler-32768)  K=7  L=45
# # -------------------------------------------------------------------
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_1    -1 32768 7 45
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_2    -1 32768 7 45
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_3    -1 32768 7 45
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_1  -1 32768 7 45
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_2  -1 32768 7 45
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_3  -1 32768 7 45
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multiquery  -1 32768 7 45
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multivalue  -1 32768 7 45
# bash Accuracy.sh llama-3.1-8b Synthetic vt               -1 32768 7 45
# bash Accuracy.sh llama-3.1-8b Synthetic cwe              -1 32768 7 45
# bash Accuracy.sh llama-3.1-8b Synthetic fwe              -1 32768 7 45
# bash Accuracy.sh llama-3.1-8b Synthetic qa_1             -1 32768 7 45
# bash Accuracy.sh llama-3.1-8b Synthetic qa_2             -1 32768 7 45

# # -------------------------------------------------------------------
# # budget-1024 | llama-3.1-8b | Synthetic (ruler-65536)  K=10  L=125
# # -------------------------------------------------------------------
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_1    -1 65536 10 125
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_2    -1 65536 10 125
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_3    -1 65536 10 125
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_1  -1 65536 10 125
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_2  -1 65536 10 125
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_3  -1 65536 10 125
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multiquery  -1 65536 10 125
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multivalue  -1 65536 10 125
# bash Accuracy.sh llama-3.1-8b Synthetic vt               -1 65536 10 125
# bash Accuracy.sh llama-3.1-8b Synthetic cwe              -1 65536 10 125
# bash Accuracy.sh llama-3.1-8b Synthetic fwe              -1 65536 10 125
# bash Accuracy.sh llama-3.1-8b Synthetic qa_1             -1 65536 10 125
# bash Accuracy.sh llama-3.1-8b Synthetic qa_2             -1 65536 10 125




# -------------------------------------------------------------------
# budget-1024 | qwen-2.5-7b | Synthetic (ruler-4096)  K=7  L=120
# -------------------------------------------------------------------
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_1    -1 4096 7 120
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_2    -1 4096 7 120
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_3    -1 4096 7 120
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_1  -1 4096 7 120
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_2  -1 4096 7 120
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_3  -1 4096 7 120
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multiquery  -1 4096 7 120
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multivalue  -1 4096 7 120
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic vt               -1 4096 7 120
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic cwe              -1 4096 7 120
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic fwe              -1 4096 7 120
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic qa_1             -1 4096 7 120
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic qa_2             -1 4096 7 120

# -------------------------------------------------------------------
# budget-1024 | qwen-2.5-7b | Synthetic (ruler-8192)  K=8  L=135
# -------------------------------------------------------------------
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_1    -1 8192 8 135
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_2    -1 8192 8 135
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_3    -1 8192 8 135
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_1  -1 8192 8 135
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_2  -1 8192 8 135
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_3  -1 8192 8 135
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multiquery  -1 8192 8 135
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multivalue  -1 8192 8 135
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic vt               -1 8192 8 135
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic cwe              -1 8192 8 135
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic fwe              -1 8192 8 135
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic qa_1             -1 8192 8 135
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic qa_2             -1 8192 8 135

# -------------------------------------------------------------------
# budget-1024 | qwen-2.5-7b | Synthetic (ruler-16384)  K=8  L=88
# -------------------------------------------------------------------
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_1    -1 16384 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_2    -1 16384 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_3    -1 16384 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_1  -1 16384 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_2  -1 16384 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_3  -1 16384 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multiquery  -1 16384 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multivalue  -1 16384 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic vt               -1 16384 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic cwe              -1 16384 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic fwe              -1 16384 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic qa_1             -1 16384 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic qa_2             -1 16384 8 88

# -------------------------------------------------------------------
# budget-1024 | qwen-2.5-7b | Synthetic (ruler-32768)  K=9  L=112
# -------------------------------------------------------------------
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_1    -1 32768 9 112
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_2    -1 32768 9 112
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_3    -1 32768 9 112
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_1  -1 32768 9 112
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_2  -1 32768 9 112
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_3  -1 32768 9 112
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multiquery  -1 32768 9 112
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multivalue  -1 32768 9 112
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic vt               -1 32768 9 112
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic cwe              -1 32768 9 112
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic fwe              -1 32768 9 112
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic qa_1             -1 32768 9 112
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic qa_2             -1 32768 9 112

# -------------------------------------------------------------------
# budget-1024 | qwen-2.5-7b | Synthetic (ruler-65536)  K=10  L=143
# -------------------------------------------------------------------
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_1    -1 65536 10 143
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_2    -1 65536 10 143
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_3    -1 65536 10 143
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_1  -1 65536 10 143
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_2  -1 65536 10 143
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_3  -1 65536 10 143
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multiquery  -1 65536 10 143
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multivalue  -1 65536 10 143
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic vt               -1 65536 10 143
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic cwe              -1 65536 10 143
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic fwe              -1 65536 10 143
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic qa_1             -1 65536 10 143
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic qa_2             -1 65536 10 143




# =============================================================================
# budget-128 (ruler-65536 only)
# =============================================================================

# -------------------------------------------------------------------
# budget-128 | llama-3.1-8b | Synthetic (ruler-65536)  K=11  L=65
# -------------------------------------------------------------------
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_1    -1 65536 11 65
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_2    -1 65536 11 65
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_3    -1 65536 11 65
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_1  -1 65536 11 65
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_2  -1 65536 11 65
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_3  -1 65536 11 65
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multiquery  -1 65536 11 65
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multivalue  -1 65536 11 65
# bash Accuracy.sh llama-3.1-8b Synthetic vt               -1 65536 11 65
# bash Accuracy.sh llama-3.1-8b Synthetic cwe              -1 65536 11 65
# bash Accuracy.sh llama-3.1-8b Synthetic fwe              -1 65536 11 65
# bash Accuracy.sh llama-3.1-8b Synthetic qa_1             -1 65536 11 65
# bash Accuracy.sh llama-3.1-8b Synthetic qa_2             -1 65536 11 65

# -------------------------------------------------------------------
# budget-128 | qwen-2.5-7b | Synthetic (ruler-65536)  K=11  L=70
# -------------------------------------------------------------------
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_1    -1 65536 11 70
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_2    -1 65536 11 70
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_3    -1 65536 11 70
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_1  -1 65536 11 70
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_2  -1 65536 11 70
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_3  -1 65536 11 70
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multiquery  -1 65536 11 70
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multivalue  -1 65536 11 70
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic vt               -1 65536 11 70
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic cwe              -1 65536 11 70
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic fwe              -1 65536 11 70
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic qa_1             -1 65536 11 70
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic qa_2             -1 65536 11 70


# =============================================================================
# budget-384 (ruler-65536 only)
# =============================================================================

# -------------------------------------------------------------------
# budget-384 | llama-3.1-8b | Synthetic (ruler-65536)  K=10  L=75
# -------------------------------------------------------------------
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_1    -1 65536 10 75
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_2    -1 65536 10 75
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_3    -1 65536 10 75
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_1  -1 65536 10 75
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_2  -1 65536 10 75
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_3  -1 65536 10 75
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multiquery  -1 65536 10 75
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multivalue  -1 65536 10 75
# bash Accuracy.sh llama-3.1-8b Synthetic vt               -1 65536 10 75
# bash Accuracy.sh llama-3.1-8b Synthetic cwe              -1 65536 10 75
# bash Accuracy.sh llama-3.1-8b Synthetic fwe              -1 65536 10 75
# bash Accuracy.sh llama-3.1-8b Synthetic qa_1             -1 65536 10 75
# bash Accuracy.sh llama-3.1-8b Synthetic qa_2             -1 65536 10 75

# -------------------------------------------------------------------
# budget-384 | qwen-2.5-7b | Synthetic (ruler-65536)  K=10  L=80
# -------------------------------------------------------------------
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_1    -1 65536 10 80
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_2    -1 65536 10 80
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_3    -1 65536 10 80
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_1  -1 65536 10 80
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_2  -1 65536 10 80
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_3  -1 65536 10 80
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multiquery  -1 65536 10 80
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multivalue  -1 65536 10 80
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic vt               -1 65536 10 80
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic cwe              -1 65536 10 80
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic fwe              -1 65536 10 80
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic qa_1             -1 65536 10 80
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic qa_2             -1 65536 10 80


# =============================================================================
# budget-4096 (ruler-65536 only)
# =============================================================================

# -------------------------------------------------------------------
# budget-4096 | llama-3.1-8b | Synthetic (ruler-65536)  K=8  L=85
# -------------------------------------------------------------------
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_1    -1 65536 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_2    -1 65536 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic niah_single_3    -1 65536 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_1  -1 65536 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_2  -1 65536 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multikey_3  -1 65536 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multiquery  -1 65536 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic niah_multivalue  -1 65536 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic vt               -1 65536 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic cwe              -1 65536 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic fwe              -1 65536 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic qa_1             -1 65536 8 85
# bash Accuracy.sh llama-3.1-8b Synthetic qa_2             -1 65536 8 85

# -------------------------------------------------------------------
# budget-4096 | qwen-2.5-7b | Synthetic (ruler-65536)  K=8  L=88
# -------------------------------------------------------------------
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_1    -1 65536 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_2    -1 65536 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_3    -1 65536 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_1  -1 65536 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_2  -1 65536 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multikey_3  -1 65536 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multiquery  -1 65536 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_multivalue  -1 65536 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic vt               -1 65536 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic cwe              -1 65536 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic fwe              -1 65536 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic qa_1             -1 65536 8 88
# bash Accuracy.sh qwen-2.5-7b-1m Synthetic qa_2             -1 65536 8 88
