#!/usr/bin/env bash

# AccuracyOverview quick-entry script.  It takes no arguments: add or remove
# direct Accuracy.sh calls below to choose the model/task to run.

# ============================================================
# LongBench -- llama-3.1-8b
# ============================================================
# bash Accuracy.sh llama-3.1-8b RetroInfer LongBench narrativeqa 130000 -1 1024
# bash Accuracy.sh llama-3.1-8b RetroInfer LongBench qasper 130000 -1 1024
# bash Accuracy.sh llama-3.1-8b RetroInfer LongBench 2wikimqa 130000 -1 1024
# bash Accuracy.sh llama-3.1-8b RetroInfer LongBench musique 130000 -1 1024

# ============================================================
# LongBench -- Qwen2.5
# ============================================================
# bash Accuracy.sh qwen-2.5-7b RetroInfer LongBench narrativeqa 130000 -1 1024
# bash Accuracy.sh qwen-2.5-7b RetroInfer LongBench qasper 130000 -1 1024

# ============================================================
# LongBench -- Qwen3-4B
# Qwen3 uses 40,960 native positions; the adapter applies YaRN beyond that.
# ============================================================
# bash Accuracy.sh qwen3-4b RetroInfer LongBench narrativeqa 40960 -1 1024
# bash Accuracy.sh qwen3-4b RetroInfer LongBench qasper 40960 -1 1024

# ============================================================
# Synthetic (RULER) -- Qwen2.5
# ============================================================
# bash Accuracy.sh qwen-2.5-7b-1m RetroInfer Synthetic niah_single_1 4096 -1 1024

# ============================================================
# Synthetic (RULER) -- Qwen3-4B
# This is the default no-argument run.  The installed Qwen3 RULER prompts are
# shared across dense Qwen3 sizes via the qwen3-32b data directory.
# ============================================================
# bash Accuracy.sh qwen3-4b RetroInfer Synthetic niah_single_1 4096 -1 1024

# CUDA_VISIBLE_DEVICES=0 bash Accuracy.sh qwen3-32b RetroInfer LongBench narrativeqa 66536 -1 1024
# CUDA_VISIBLE_DEVICES=0 bash Accuracy.sh qwen3-32b RetroInfer LongBench qasper 66536 -1 1024
# CUDA_VISIBLE_DEVICES=0 bash Accuracy.sh qwen3-32b RetroInfer LongBench 2wikimqa 66536 -1 1024
# CUDA_VISIBLE_DEVICES=0 bash Accuracy.sh qwen3-32b RetroInfer LongBench musique 66536 -1 1024

CUDA_VISIBLE_DEVICES=0 bash Accuracy.sh qwen3-32b RetroInfer LongBench gov_report 66536 -1 1024
CUDA_VISIBLE_DEVICES=0 bash Accuracy.sh qwen3-32b RetroInfer LongBench multi_news 66536 -1 1024
CUDA_VISIBLE_DEVICES=0 bash Accuracy.sh qwen3-32b RetroInfer LongBench triviaqa 66536 -1 1024
CUDA_VISIBLE_DEVICES=0 bash Accuracy.sh qwen3-32b RetroInfer LongBench samsum 66536 -1 1024
CUDA_VISIBLE_DEVICES=0 bash Accuracy.sh qwen3-32b RetroInfer LongBench passage_count 66536 -1 1024
CUDA_VISIBLE_DEVICES=0 bash Accuracy.sh qwen3-32b RetroInfer LongBench passage_retrieval_en 66536 -1 1024
CUDA_VISIBLE_DEVICES=0 bash Accuracy.sh qwen3-32b RetroInfer LongBench lcc 66536 -1 1024
CUDA_VISIBLE_DEVICES=0 bash Accuracy.sh qwen3-32b RetroInfer LongBench repobench-p 66536 -1 1024