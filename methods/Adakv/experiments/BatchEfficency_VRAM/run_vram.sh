#!/usr/bin/env bash

# Same as AccuracyOverview/run.sh: add or uncomment commands below.
# Parameters (counted after script name):
#  1 MODEL: model alias (config/model2path.json), HF model ID, or local path.
#  2 BENCHMARK: LongBench or RULER (also accepts ruler/Synthetic, case-insensitive).
#  3 TASK: dataset task name; RULER e.g. niah_single_1, LongBench e.g. narrativeqa.
#  4 BATCH_SIZE: samples per batch, default 1; partial final batch also runs.
#  5 CONTEXT_LENGTH: RULER context length in tokens, default 65536; use 0 placeholder for LongBench.
#  6 NUM_EXAMPLES: max first N samples from dataset, default -1 means all (sample count, not batch count).
#  7 BUDGET: HeadKV cache budget (max_capacity_prompts), default 1024.
#  8 METHOD: HeadKV method (AdativeKV/ReasonKV/SnapKV/PyramidKV/fullkv), default AdativeKV.
#  9 HEAD_CHOICE: random/copy/reason, default random.
# 10 BETA: HeadKV beta, default 1.5.
# 11 TEMP: HeadKV temp, default 1.0.
# 12 MAX_LEN: max prompt length; 0 = model default (LongBench) or = context_length (RULER).
# 13 FIXED_OUTPUT_LENGTH: fixed output tokens per sample; VRAM entry default 2;
#    use 0 for default max output length, allowing early EOS per sample.
# First 3 params required, rest optional; later params require preserving earlier positions.
# VRAM entry defaults to 2 warmup runs (WARMUP overridable), warmups excluded from results.
# Example: llama-3.1-8b RULER niah_single_1, 4K data, batch_size=4, 2 output tokens each.
# bash VRAM.sh llama-3.1-8b RULER niah_single_1 4 4096 -1 1024 AdativeKV random 1.5 1.0 0 2

# ============================================================
# RULER -- qwen3-32b, different batch_size (4K) -- use existing data folder directly
# ============================================================
bash VRAM.sh qwen3-32b RULER niah_single_1 1 4096 -1 1024 AdativeKV random 1.5 1.0 0 2
bash VRAM.sh qwen3-32b RULER niah_single_1 2 4096 -1 1024 AdativeKV random 1.5 1.0 0 2
bash VRAM.sh qwen3-32b RULER niah_single_1 4 4096 -1 1024 AdativeKV random 1.5 1.0 0 2

# ---- backup: llama/qwen RULER (need to populate data folders under benchmark_root first) ----
# bash VRAM.sh llama-3.1-8b RULER niah_single_1 2 4096 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh llama-3.1-8b RULER niah_single_1 4 4096 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh qwen-2.5-7b-1m RULER niah_single_1 2 4096 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh qwen-2.5-7b-1m RULER niah_single_1 4 4096 -1 1024 AdativeKV random 1.5 1.0 0 2

# ============================================================
# RULER -- different context_length (batch_size=4)
# ============================================================
# bash VRAM.sh llama-3.1-8b RULER niah_single_1 4 8192 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh llama-3.1-8b RULER niah_single_1 4 16384 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh llama-3.1-8b RULER niah_single_1 4 32768 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh llama-3.1-8b RULER niah_single_1 4 65536 -1 1024 AdativeKV random 1.5 1.0 0 2

# ============================================================
# LongBench -- narrativeqa, different batch_size (CONTEXT_LENGTH use 0 placeholder)
# ============================================================
# bash VRAM.sh llama-3.1-8b LongBench narrativeqa 1 0 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh llama-3.1-8b LongBench narrativeqa 2 0 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh llama-3.1-8b LongBench narrativeqa 4 0 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh qwen-2.5-7b LongBench narrativeqa 2 0 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh glm-4-9b-chat-1m LongBench narrativeqa 2 0 -1 1024 AdativeKV random 1.5 1.0 0 2