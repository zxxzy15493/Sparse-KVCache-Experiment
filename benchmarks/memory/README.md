# 推理 CPU 内存基准

在仓库根目录运行。`--models`、`--methods`、`--batch-sizes` 和 `--context-lens` 支持传入多个值，脚本逐个执行这些组合。每个组合使用独立的 Python 进程加载模型并完成预热和正式推理。

```bash
conda activate /home/kv/miniconda3/envs/kv
CUDA_VISIBLE_DEVICES=1 python benchmarks/memory/run_memory.py \
  --models llama-3.1-8b \
  --methods full_attention snapkv \
  --batch-sizes 1 2 \
  --context-lens 4096 8192 \
  --data-file 'benchmarks/ruler/benchmark_root/{model}/synthetic/4096/data/niah_single_1/validation.jsonl' \
  --budget 1024 --output-length 32 --warmup 1 --measured-rounds 3
```

默认可见的物理 GPU 为 1，进程内使用 `cuda:0`。通过外部 `CUDA_VISIBLE_DEVICES` 选择设备。`run_memory.py` 构造每个组合的加载参数，`inference.py` 和 `serial_prefill.py` 负责输入处理和定长生成。模型加载使用仓库公共 `loaders` 和方法配置。`--efficiency-backend` 为 Quest 和 ClusterKV 选择原生 KV cache 后端。

`--context-lens` 是每条输入的目标 token 数，过长的样本从中间截断，过短的样本通过重复原有 token 补足。`--data-file` 接受 RULER JSONL，至少需要 `最大 batch size × 正式轮数` 条样本。路径中的 `{model}` 会替换为模型别名。

`--serial-prefill auto` 对支持的多样本 batch 逐条执行 prefill 后合并 KV cache，再按 batch decode。Adakv 和 HeadKV 在 Llama、Qwen 路径上保留批量 forward，并逐样本压缩 KV。`on` 允许其他方法尝试逐条 prefill，`off` 使用批量 prefill。实际执行的模式见阶段明细。

## 结果

结果直接写入 `benchmarks/memory/output/`，也可以用 `--output-root` 指定根目录。根目录的汇总表记录最近一次运行，重复测量同一组合时，阶段明细更新为最近一次成功测量的结果。终端显示各组合的推理峰值、最大增量、进程峰值和汇总路径。用等宽字体打开 `.txt` 可查看列对齐的表格，CSV 可用于分析。内存统一用 MiB 表示，保留两位小数。失败的组合在汇总表中标为 `failed`，其余组合继续执行，整次运行最终返回非零退出码。

组合目录使用模型别名，例如 `output/qwen3-0.6b/full_attention/b2_c4096/`。

| 文件 | 内容 |
| --- | --- |
| `summary.txt` / `summary.csv` | 每个组合的状态、模型加载峰值、正式推理峰值、正式推理最大增量和进程峰值 |
| `<模型>/<方法>/b<batch>_c<context>/phases.txt` / `phases.csv` | 运行库初始化、数据加载、模型加载、每轮预热及每轮正式推理的起始 RSS、结束 RSS、采样峰值和峰值增量，同时记录正式轮次的实际 prefill 模式 |

`cpu_memory.py` 每隔 0.10 秒及每个阶段的起止点，从 Linux `/proc/self/status` 读取推理进程的 `VmRSS`。正式推理覆盖输入准备、生成及批次清理。每轮峰值增量为该轮采样峰值减去该轮的起始 RSS，汇总表取正式轮次的最大值。

进程峰值取 `VmHWM`，覆盖整个推理进程，并可捕捉采样间隔内出现的峰值。RSS 统计范围是单个推理进程的 CPU 常驻内存，包括驻留的模型数据、CPU KV cache、pinned memory 和运行库开销。子进程内存和 GPU 显存各自独立于这一统计口径。
