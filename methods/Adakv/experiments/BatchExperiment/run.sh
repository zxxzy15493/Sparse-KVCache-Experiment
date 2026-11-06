#!/usr/bin/env bash

# bash BatchExperiment.sh MODEL BENCHMARK TASK BATCH_SIZE [CONTEXT_LENGTH] [NUM_EXAMPLES]
# batch_size: 1 2 4 8
# RULER context_length: 4096 8192 16384 32768 65536
# LongBench does not use context_length; use 0 placeholder when specifying NUM_EXAMPLES.
# HeadKV params via env vars: METHOD / HEAD_CHOICE / MAX_CAPACITY_PROMPTS / BETA / TEMP / WINDOW_SIZE

# qwen3-32b RULER, different batch_size (4K) -- use existing data folder under benchmark_root
bash BatchExperiment.sh qwen3-32b RULER niah_single_1 1 4096
bash BatchExperiment.sh qwen3-32b RULER niah_single_1 2 4096
bash BatchExperiment.sh qwen3-32b RULER niah_single_1 4 4096
# bash BatchExperiment.sh qwen3-32b RULER niah_single_1 8 4096

# ---- backup: llama/qwen RULER (need to populate data folders under benchmark_root first) ----
# bash BatchExperiment.sh qwen RULER niah_single_1 1 4096
# bash BatchExperiment.sh qwen RULER niah_single_1 2 4096
# bash BatchExperiment.sh qwen RULER niah_single_1 4 4096
# bash BatchExperiment.sh llama RULER niah_single_1 1 4096
# bash BatchExperiment.sh llama RULER niah_single_1 2 4096
# bash BatchExperiment.sh llama RULER niah_single_1 4 4096

# qwen LongBench, different batch_size
bash BatchExperiment.sh qwen LongBench narrativeqa 1
bash BatchExperiment.sh qwen LongBench narrativeqa 2
bash BatchExperiment.sh qwen LongBench narrativeqa 4
# bash BatchExperiment.sh qwen LongBench narrativeqa 8

# llama LongBench, different batch_size
bash BatchExperiment.sh llama LongBench narrativeqa 1
bash BatchExperiment.sh llama LongBench narrativeqa 2
bash BatchExperiment.sh llama LongBench narrativeqa 4
# bash BatchExperiment.sh llama LongBench narrativeqa 8

# ---- backup: other context_length (batch_size=4) ----
# bash BatchExperiment.sh qwen RULER niah_single_1 4 8192
# bash BatchExperiment.sh qwen RULER niah_single_1 4 16384
# bash BatchExperiment.sh qwen RULER niah_single_1 4 32768
# bash BatchExperiment.sh qwen RULER niah_single_1 4 65536

# ---- backup: other LongBench tasks (batch_size=4) ----
# bash BatchExperiment.sh qwen LongBench qasper 4
# bash BatchExperiment.sh qwen LongBench 2wikimqa 4
# bash BatchExperiment.sh qwen LongBench musique 4
# bash BatchExperiment.sh qwen LongBench gov_report 4
# bash BatchExperiment.sh qwen LongBench multi_news 4
# bash BatchExperiment.sh qwen LongBench triviaqa 4
# bash BatchExperiment.sh qwen LongBench samsum 4
# bash BatchExperiment.sh qwen LongBench passage_count 4
# bash BatchExperiment.sh qwen LongBench passage_retrieval_en 4
# bash BatchExperiment.sh qwen LongBench lcc 4
# bash BatchExperiment.sh qwen LongBench repobench-p 4

# ---- backup: other RULER tasks (batch_size=4, 32K) ----
# bash BatchExperiment.sh qwen RULER niah_single_2 4 32768
# bash BatchExperiment.sh qwen RULER niah_single_3 4 32768
# bash BatchExperiment.sh qwen RULER niah_multikey_1 4 32768
# bash BatchExperiment.sh qwen RULER niah_multikey_2 4 32768
# bash BatchExperiment.sh qwen RULER niah_multikey_3 4 32768
# bash BatchExperiment.sh qwen RULER niah_multiquery 4 32768
# bash BatchExperiment.sh qwen RULER niah_multivalue 4 32768
# bash BatchExperiment.sh qwen RULER vt 4 32768
# bash BatchExperiment.sh qwen RULER cwe 4 32768
# bash BatchExperiment.sh qwen RULER fwe 4 32768
# bash BatchExperiment.sh qwen RULER qa_1 4 32768
# bash BatchExperiment.sh qwen RULER qa_2 4 32768

# ---- backup: glm (LongBench / RULER only to LongBench combo) ----
# bash BatchExperiment.sh glm LongBench narrativeqa 4
# bash BatchExperiment.sh glm-4-9b-chat-1m RULER niah_single_1 4 32768