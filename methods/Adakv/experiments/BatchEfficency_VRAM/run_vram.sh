#!/usr/bin/env bash

# 与 AccuracyOverview/run.sh 一样，直接添加或取消注释下面的命令。
# 参数（从脚本名后开始计数）：
#  1 MODEL：模型别名（config/model2path.json）、Hugging Face 模型 ID 或本地路径。
#  2 BENCHMARK：LongBench 或 RULER（也接受 ruler/Synthetic，不区分大小写）。
#  3 TASK：数据集任务名；RULER 例如 niah_single_1，LongBench 例如 narrativeqa。
#  4 BATCH_SIZE：每批样本数，默认 1；最后不足一批也会运行。
#  5 CONTEXT_LENGTH：RULER 上下文长度（token），默认 65536；LongBench 用 0 占位。
#  6 NUM_EXAMPLES：最多使用数据集前 N 条样本，默认 -1 表示全部（这是样本数，不是批次数）。
#  7 BUDGET：HeadKV 缓存预算（max_capacity_prompts），默认 1024。
#  8 METHOD：HeadKV 方法（AdativeKV/ReasonKV/SnapKV/PyramidKV/fullkv），默认 AdativeKV。
#  9 HEAD_CHOICE：random/copy/reason，默认 random。
# 10 BETA：HeadKV beta，默认 1.5。
# 11 TEMP：HeadKV temp，默认 1.0。
# 12 MAX_LEN：prompt 长度上限，0 表示按模型默认（LongBench）或等于 context_length（RULER）。
# 13 FIXED_OUTPUT_LENGTH：每条样本固定生成的 token 数，显存入口默认 2；
#    填 0 时使用任务默认最大输出长度，并允许每条样本遇到 EOS 提前结束。
# 前 3 个参数必填，其余可省略；指定后面的参数时需保留前面的位置。
# 显存入口默认预热 2 次（WARMUP 可覆盖），预热不计入结果。
# 示例：llama-3.1-8b RULER niah_single_1，4K 数据，每批 4 条，每条输出 2 token。
# bash VRAM.sh llama-3.1-8b RULER niah_single_1 4 4096 -1 1024 AdativeKV random 1.5 1.0 0 2

# ============================================================
# RULER -- qwen3-32b，不同 batch_size（4K）—— 直接用现成数据文件夹
# ============================================================
bash VRAM.sh qwen3-32b RULER niah_single_1 1 4096 -1 1024 AdativeKV random 1.5 1.0 0 2
bash VRAM.sh qwen3-32b RULER niah_single_1 2 4096 -1 1024 AdativeKV random 1.5 1.0 0 2
bash VRAM.sh qwen3-32b RULER niah_single_1 4 4096 -1 1024 AdativeKV random 1.5 1.0 0 2

# ---- 备用：llama/qwen RULER（需先补齐 benchmark_root 下对应数据文件夹） ----
# bash VRAM.sh llama-3.1-8b RULER niah_single_1 2 4096 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh llama-3.1-8b RULER niah_single_1 4 4096 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh qwen-2.5-7b-1m RULER niah_single_1 2 4096 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh qwen-2.5-7b-1m RULER niah_single_1 4 4096 -1 1024 AdativeKV random 1.5 1.0 0 2

# ============================================================
# RULER -- 不同 context_length（batch_size=4）
# ============================================================
# bash VRAM.sh llama-3.1-8b RULER niah_single_1 4 8192 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh llama-3.1-8b RULER niah_single_1 4 16384 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh llama-3.1-8b RULER niah_single_1 4 32768 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh llama-3.1-8b RULER niah_single_1 4 65536 -1 1024 AdativeKV random 1.5 1.0 0 2

# ============================================================
# LongBench -- narrativeqa，不同 batch_size（CONTEXT_LENGTH 用 0 占位）
# ============================================================
# bash VRAM.sh llama-3.1-8b LongBench narrativeqa 1 0 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh llama-3.1-8b LongBench narrativeqa 2 0 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh llama-3.1-8b LongBench narrativeqa 4 0 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh qwen-2.5-7b LongBench narrativeqa 2 0 -1 1024 AdativeKV random 1.5 1.0 0 2
# bash VRAM.sh glm-4-9b-chat-1m LongBench narrativeqa 2 0 -1 1024 AdativeKV random 1.5 1.0 0 2
