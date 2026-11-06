# 效率与显存测试

复用 LongBench 的 `benchmarks.common`、`method_configs/*.yaml` 和
`loaders.load_model_and_tokenizer(args)`。各部分职责如下：

| 文件 | 职责 |
| --- | --- |
| `pred.py` | 参数转换、RULER 数据准备、模型加载、定长预测、方法缓存重置、结果保存 |
| `tools/efficiency.py` | 独立计时器 `EfficiencyMeter`，统计生成总耗时和逐 token 耗时 |
| `tools/memory.py` | 独立显存监测器 `MemoryMonitor`，统计参与 GPU 的 allocator baseline 和 peak |
| `tools/run_grid.py` | 批量安排模型、方法、batch size 和输入长度组合，支持续跑、超时与失败剪枝 |
| `tools/run_method_batch.py` | 每个模型、方法和 batch 的 worker；复用 `pred.py` 的预测和测量逻辑 |
| `tools/consolidate_results.py` | 将旧版逐组合结果合并到统一的 `pred.json` |
| `run_efficiency_vram.sh` | 当前单卡矩阵实验的启动脚本 |

`tools/reporting.py` 汇总批次结果，`tools/smoke_all_methods.py` 在独立进程中运行各方法，
`tools/test_benchmark.py` 提供回归测试。
计时器和显存监测器不依赖预测入口、模型接口或方法实现，可以分别包住任意调用。
`pred.generate()` 默认直接返回新生成的 token；需要逐步计时时，传入 `timer.step`。
批量 runner 也调用 `pred.py` 的 `predict_batch()`，所以单次测试与矩阵测试共享相同的计时和显存测量实现。
`efficiency.py` 和 `memory.py` 分别是两种测量器，不是矩阵运行器的替代入口。

```python
with MemoryMonitor(devices) as memory, EfficiencyMeter(devices, batch_size, output_length) as timer:
    tokens = generate(model, input_ids, attention_mask, output_length, step_context=timer.step)
```

入口只做参数翻译，不维护方法能力表、重复校验、接口探测兜底或异常包装。
输入格式、参数范围和方法支持情况遵循数据、loader 与模型本身的接口。
清理资源和恢复临时状态仍使用 `finally`。

## 运行

```bash
conda activate kvcache
CUDA_VISIBLE_DEVICES=0 python benchmarks/efficency_vram/pred.py \
  --method full_attention --model qwen3-32b \
  --batch_size 1 --input_length 4096 --fixed_output_length 128 \
  --task niah_single_1 --num_samples 8 --warmup 2 \
  --measurement both --save-predictions --experiment full_4k
```

`--measurement` 支持 `efficiency`、`vram`、`both` 和 `none`。
`none` 只运行预测并保存文本和 token IDs；其他模式添加 `--save-predictions` 保存预测。
`--dry-run` 只读取数据、显示批次选择和结果路径。参数支持连字符和下划线写法。

运行当前配置的长上下文矩阵：

```bash
bash benchmarks/efficency_vram/run_efficiency_vram.sh
```

矩阵 runner 会把状态表、汇总 CSV、报告、日志和预测结果都保存在 `output/` 下本次实验的目录。
该目录由 `.gitignore` 排除。启动脚本中的 Conda 环境、GPU、模型和输入长度列表可按机器调整。

数据默认位于：

```text
benchmarks/ruler/benchmark_root/<model alias>/synthetic/<context_length>/data/<task>/validation.jsonl
```

`context_length` 默认等于 `input_length`，`--data-root` 可替换 `benchmark_root`。
`--data-dir` 可指向 JSONL 文件、包含 `validation.jsonl` 的任务目录，或包含任务目录的 `data` 目录。
JSONL 每行使用 RULER 的 `index` 和 `input` 字段。例如复用 Qwen3-32B 的数据：

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/efficency_vram/pred.py \
  --method streaming --model qwen3-4b \
  --data-dir benchmarks/ruler/benchmark_root/qwen3-32b/synthetic/4096/data \
  --batch_size 2 --input_length 4096 --fixed_output_length 128 \
  --sink_token 16 --local_window_token 1008 --measurement both
```

## 长度与批次参数（重点）

**`--input-length` 指定每条样本最终送入模型的 token 数，会主动调整输入内容。**
它与源数据目录的长度、方法自身的容量限制分别控制不同环节。所有长度均以 token 为单位，
不以字符数为单位。参数既支持 `--input-length`，也支持 `--input_length` 写法。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--input-length` | `4096` | 最终输入长度。分词后过长则从中间截断，过短则按 `--short-input` 处理 |
| `--short-input` | `repeat` | 只处理短输入：`repeat` 重复原有 tokens 补足；`error` 直接报错 |
| `--context-length` | 等于 `input_length` | 未指定 `--data-dir` 时，选择 RULER 的源数据长度目录；不保证分词后的实际长度 |
| `--fixed-output-length` | `128` | 固定生成的新 token 数，不包含输入；即使遇到 EOS 也继续生成 |
| `--batch-size` | `1` | 每批样本数。输入张量为 `[actual_batch_size, input_length]` |
| `--num-samples` / `--limit` | `8` | 选择多少条不同样本参与测量；`-1` 选择全部 |
| `--num-batches` | 不限制 | 进一步限制测量批次数；与样本数限制同时生效 |
| `--warmup` | `2` | 重复第一批的预热次数，不额外消耗样本，不计入测量结果；`0` 表示不预热 |
| `--allow-partial-batch` | 关闭 | 开启后测量不足一整批的尾部样本，否则丢弃；尾批按实际 batch size 独立汇总 |
| `--budget` | 方法 YAML 中的值，缺省为 `1024` | 方法使用的缓存预算，含义由方法决定；不改变输入或输出长度 |

### 什么时候截断或补足

每个 batch（包括 warmup）都会先对 RULER 的 `input` 直接分词，不添加 chat template
或特殊 token，再把实际长度与 `--input-length` 比较。这一步对所有方法使用相同规则，
不由某个方法的 `max_length` 触发。RULER 目录名称是源数据的目标长度，
更换 tokenizer 等情况都可能使实际 token 数与目录长度不同。

以 `--input-length 4096` 为例：

| 分词后的原始长度 | `--short-input` | 实际处理 |
| --- | --- | --- |
| `5000` | `repeat` 或 `error` | 保留前 2048 和后 2048 个 tokens，删除中间部分 |
| `4096` | `repeat` 或 `error` | 原样使用 |
| `3000` | `repeat` | 在原序列中间插入 1096 个从原有 tokens 取出的重复 tokens，保留末尾问题 |
| `3000` | `error` | 报错，不进行该批生成 |

**`--short-input error` 只禁止短输入的重复补足，长输入仍会截断。**
`repeat` 补入的内容会参与模型计算，不是被 attention mask 屏蔽的 padding。
调整后每条输入都恰好为 `input_length`，attention mask 全为 1。
指标中的 `original_input_tokens` 保存原始长度，`input_tokens` 保存调整后长度，
`input_adjustments` 记录 `unchanged`、`middle_truncated` 或 `repeated`。
这种长度调整用于效率对照实验，输出不作为 RULER 准确率结果。

### 源数据长度与测试长度可以不同

例如从 8192 长度的 RULER 目录读取数据，将每条输入统一为 4096 tokens，再生成 128 tokens：

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/efficency_vram/pred.py \
  --method full_attention --model qwen3-32b \
  --context-length 8192 --input-length 4096 \
  --short-input repeat --fixed-output-length 128
```

指定 `--data-dir` 后，源文件由该路径决定，`--context-length` 不再用于寻找数据目录；
`--input-length` 仍控制每条样本最终送入模型的长度。
输出 N 个 tokens 时，HF 路径执行一次 prefill 和 N−1 次 cached decode；
原生引擎使用各自的定长生成参数。

### 与方法内部 max_length 的区别

方法内部的 `max_length`、`max_seq_length` 用于上下文或缓存容量，入口没有统一的
`--max-length` 参数，也不会根据这些容量自动触发输入截断。
入口会从输入长度、源数据长度和输出长度推导 loader 所需的容量提示：
PQCache 的 loader 默认额外预留 1024 tokens，更长输出会相应扩大提示；
MagicPig 的 RULER loader 额外预留 4096 tokens。RetroInfer 则按模型配置设置 `max_length`，
并在原生生成中检查“输入长度 + 输出长度”是否超限。
输入调整与这些容量检查属于不同环节，方法自身的限制见 [ISSUES.md](tools/ISSUES.md)。

## 测量口径

| 字段 | 含义 |
| --- | --- |
| `ttft_s` | prefill forward + 首 token argmax 的秒数 |
| `tpot_ms` | 后续 N−1 次 forward + argmax 的平均毫秒数 |
| `decode_latency_s` | 后续步骤耗时之和 |
| `step_latencies_s` | 第一个值为 prefill，后续值为 decode |
| `generation_latency_s` | 整个生成调用的墙钟秒数，包括 cache/mask 更新和逐步同步开销 |
| `output_throughput_tokens_per_s` | batch_size × 输出长度 / 生成总耗时 |
| `decode_throughput_tokens_per_s` | batch_size × (输出长度−1) / decode 耗时 |
| `peak_allocated_mib` / `peak_reserved_mib` | 整次生成中参与 GPU 的最大单卡 allocator 峰值，包含 prefill 和 decode |
| `decode_peak_allocated_mib` / `decode_peak_reserved_mib` | 首 token/prefill 结束后重置峰值计数，仅统计 decode 阶段；绝对值仍包含 decode 开始时常驻的权重和 KV cache |
| `sum_device_peak_*` | 各 GPU 峰值之和；各卡峰值不一定同时发生 |
| `memory_by_device` | 每卡 baseline、peak 和 allocated 增量 |

`output_throughput_tokens_per_s` 衡量整段生成速度，包含 prefill 和全部输出 token；
`decode_throughput_tokens_per_s` 只衡量首 token 之后的 decode 阶段，不包含 TTFT。

加载、tokenize、输入搬运、方法准备、清理和输出解码均在测量范围外。
整次显存峰值在输出移到 CPU 前读取，包含常驻权重、输入、KV、logits 和 workspace；
增量不是纯 KV-cache 大小。MiB = 2**20 字节，统计不包含 CPU/offload 内存，
也不包含绕过 PyTorch allocator 的扩展分配。

计时和显存统计分别同步模型实际参与的 CUDA 设备。`both` 带逐步同步开销，
`vram` 只在生成前后同步。N=1 时 TPOT 为空。
RetroInfer/MagicPig 使用原生 `generate`，只有总耗时和显存，统一 TTFT/TPOT 为空。

## 方法参数与结果

沿用 `--budget`、方法 YAML 和 `--set KEY=VALUE`，另外提供以下便捷映射：

| 方法 | `--sink_token` | `--local_window_token` |
| --- | --- | --- |
| streaming | `start_size` | `recent_size` |
| duo-attention | `sink_size` | `recent_size` |
| pqcache/topk/topk32 | `sink_size` | `recent_size` |
| clusterkv | `sink` | — |
| h2o/keyformer | — | `recent_size` |

便捷参数覆盖同名 `--set`。Streaming 未显式传入 budget 时，用指定的 sink + recent
推导 budget；显式 budget 按 loader 的规则决定最终 recent_size。
其他方法参数直接通过 `--set` 传给现有 loader，支持范围见 [ISSUES.md](tools/ISSUES.md)。

所有运行结果追加到输出根目录唯一的 `pred.json`：

```text
output/pred.json
```

每条记录包含模型、方法、batch size、输入/输出长度、逐轮指标、汇总、预测和运行配置。
多次运行会继续追加记录；`result_id` 用于区分同一组合的不同尝试。
文件名不再按组合拆分。

长上下文矩阵会自动导出 CSV，和该模型/方法/任务的逐组合结果放在同一目录：
`output/<model alias>/<experiment>/<method>/<task>/history.csv` 保存每轮测量和失败状态，
`average.csv` 按 batch size 和输入长度汇总测量均值。CSV 不重复存模型名；浮点指标显示两位小数，
batch size 和 token 长度仍显示为整数。平均表保留整次生成与 decode 阶段的峰值均值。
旧结果没有阶段峰值记录，decode 峰值列需要重新运行后才会有值。
只从已有状态重新生成这些 CSV 时，运行矩阵脚本并传入 `--export-csv-only`；此选项不会加载模型或启动 GPU 测试。

## 验证与全方法冒烟

```bash
python -m pytest -q benchmarks/efficency_vram/tools/test_benchmark.py
```

测试使用随机初始化的微型 Qwen3，无需下载模型或 GPU，覆盖 HF 生成一致性、
定长/EOS、cached decode、计时单位、多卡显存统计、批次和结果输出。

全方法冒烟在各方法独立进程中运行，并保存命令、日志和 JSON/CSV 报告：

```bash
CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/kvcache/huggingface HF_HUB_OFFLINE=1 \
  MAX_CPU_IN_USE=16 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python benchmarks/efficency_vram/tools/smoke_all_methods.py \
  --output-root benchmarks/efficency_vram/output/qwen3_0.6b_smoke_new
```

输出目录须为新目录。每个方法默认超时 300 秒，超时后终止进程组。
已有实际模型验证记录及方法限制见 [ISSUES.md](tools/ISSUES.md)。
