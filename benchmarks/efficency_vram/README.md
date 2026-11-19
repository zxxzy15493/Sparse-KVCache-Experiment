# 效率与显存测试

正常入口只有 `pred.py`。从 `run_matrix()` 看矩阵循环，从 `run_test()`、`run_cell()` 看实际测试。

```text
pred.py：读取 history → 遍历模型/方法/batch/容量、阈值或稀疏率 → 启动本文件的子进程
  run_test：加载一次模型 → 依次测试多个输入长度
    run_cell：准备输入 → warmup → 测量各轮 → 保存预测和统计
```

| 文件 | 职责 |
| --- | --- |
| `pred.py` | 命令行、矩阵循环、模型加载和核心测试过程 |
| `generation.py` | 保留原输入归一化、方法初始化、生成、串行 prefill 和 batch decode |
| `metrics.py` | 原 CUDA 同步、时间及显存测量 |
| `report.py` | history、average、预测文本和简单日志输出 |
| `runtime.py` | 子进程隔离、硬超时、本次运行的超时支配集合 |
| `adaptive.py` | 原 Adakv/HeadKV 适配，不在本次重构中改变其执行方式 |

## 运行

运行前在外部激活环境，脚本使用当前 PATH 的 Python：

```bash
conda activate /home/kv/miniconda3/envs/kv
CUDA_VISIBLE_DEVICES=0 python benchmarks/efficency_vram/pred.py \
  --models llama-3.1-8b \
  --methods full_attention snapkv \
  --batch-sizes 1 4 \
  --input-lengths 4096 65536 \
  --budgets 128 1024 \
  --data-file 'benchmarks/ruler/benchmark_root/{model}/synthetic/4096/data/niah_single_1/validation.jsonl' \
  --output-root benchmarks/efficency_vram/output/example \
  --warmup 1 --measured-rounds 3 --output-length 32 \
  --timeout-seconds 1800 --efficiency-backend
```

启动脚本统一放在 `scripts/`，共享路径、默认模型和运行环境位于 `scripts/common.sh`；每个入口显式列出自己的 METHODS。
原有 7 个 `.sh` 原样移入 `tmp/` 留作归档，不修改内容，不作为当前入口。

```bash
bash benchmarks/efficency_vram/scripts/batch_overview.sh
# 只核对配置，不加载模型：
bash benchmarks/efficency_vram/scripts/trade_off_b1.sh --dry-run
```

| 新入口 | 模型 | 方法数 | batch | 输入长度 / K | warmup / measured |
| --- | --- | ---: | --- | --- | --- |
| batch_overview.sh | Llama-3.1-8B、Qwen2.5-7B-1M | 19 | 1、4 | 4、8、16、32、48、64、80、96、112、128 | 1 / 3 |
| batch_overview_b16.sh | 同上 | 19 | 16 | 4、8、16、32、48、64 | 0 / 1 |
| trade_off_b1.sh | 同上 | 18，去除 minference | 1 | 64 | 1 / 1 |
| trade_off_b4.sh | 同上 | 16，去除 minference、xattention、flexprefill | 4 | 64 | 1 / 1 |
| mul_batch.sh | 同上 | 19 | 1 至 16，每个整数 | 64 | 1 / 1 |
| qwen3.sh | Qwen3-32B | 19 | 1 | 4、8、16、32、40、48、64、80、96、112、128 | 1 / 3 |

19 个方法包含 Full Attention；沿用原 batch_overview 的方法集合，不包括 TopK/TopP 基线。
所有入口固定输出 32 token，超时 1800 秒。Overview、mul_batch、qwen3 使用 budget=1024，
阈值方法默认 0.90，DuoAttention 默认 sparsity=0.50。H2O/KeyFormer 与其他方法使用相同长度列表。
各入口使用独立的 `<入口名>_efficiency_vram` 输出目录，位于本目录的 output/ 下；可通过 OUTPUT_ROOT 环境变量覆盖。

`--budget` 仍可指定单个预算；不能与 `--budgets` 同时使用。
`--dry-run` 只展示参数；`--export-csv-only` 从 history 更新 average，不加载模型。
`--state-dir` 仅决定 log 保存位置，默认等于 output-root，不再写状态文件。

## budget、阈值与稀疏率

整数容量与阈值分别配置，阈值型方法不会重复扫描整数 budget：

```bash
BUDGETS=(128 256 512 1024 2048 3072 4096 5120 6144 7168 8192)
THRESHOLDS=(0.80 0.825 0.85 0.875 0.90 0.91 0.92 0.93 0.94 0.95 0.96)
SPARSITIES=(0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40)
# pred.py 的对应参数：
# --budgets "${BUDGETS[@]}" --thresholds "${THRESHOLDS[@]}"
# --sparsities "${SPARSITIES[@]}"
```

| 方法 | 扫描列表 | 传给现有 loader 的参数 |
| --- | --- | --- |
| topp、topp32 | THRESHOLDS | fixthreshold |
| xattention | THRESHOLDS | fixthreshold → p |
| flexprefill | THRESHOLDS | fixthreshold → gamma；tau 保留方法默认值 0.10 |
| duo-attention | SPARSITIES | sparsity |
| 其余方法 | BUDGETS | budget |

两个新 trade-off 入口使用上面的完整参数列表。直接调用 pred.py 时，阈值默认仅测 0.90，DuoAttention 稀疏率默认仅测 0.50。
阈值位于 (0, 1]；0.825、0.875 等参数在 CSV、pred.json、日志和 history 判断中保留原精度，测量指标仍保留两位小数。
XAttention 当前只有 0.80、0.85、0.90、0.95 的标定表；按本次配置要求保留全部 11 档，
缺表档位由 loader 报错并记录失败，随后继续其他组合，不插值或替换标定表。
每个阈值启动独立进程，在该进程内加载一次模型并测试多个长度。
loader 内部仍使用 YAML 的默认 budget 满足原接口，结果表的 budget 留空，threshold 记录实际阈值。
DuoAttention 同样不扫描整数 budget，结果记录在独立的 sparsity 列。
sparsity 表示 streaming head 的目标比例，取值 [0, 1]；值越大，保留完整 KV 的 head 越少。
例如 sparsity=0.80 表示目标约 80% 的 head 只保留 sink/recent KV，其余使用完整 KV；
实际比例受 head 数量、分数并列及原实现的分位数二值化影响，loader 日志打印 true sparsity。
TopP/TopP32 和稀疏 prefill 方法仍保留其原来的完整 KV 存储方式；阈值不等价于保留的 KV token 数。

## history 与本次超时规则

每次准备测试组合，直接读取对应方法的 `history.csv`，跳过实际执行过的组合。
没有独立续跑文件，也不会从历史失败构建新的剪枝规则。

- 真实执行：`attempted=true`；包括成功、OOM、普通失败及实际超时。
- 本次被超时支配、没有执行：`status=timeout, attempted=false, phase=not_started`。
- 相同模型、相同方法中，budget、batch、输入长度都大于等于本次超时点时，未测组合记为推导超时。
- 支配集合只在当前父进程内存中保存。再次启动时集合为空，旧推导超时不会阻止执行。
- 阈值型方法只在同一阈值下按 batch、输入长度传播超时，不跨阈值推断。
- DuoAttention 同样只在同一 sparsity 下传播超时，不跨 sparsity 推断。
- 真实已执行记录不覆盖。需要重测这些组合时，使用新的 output-root，或自行清理对应历史记录。
- 普通失败/OOM/不支持只记录当前实际尝试的结果，完整 traceback 保存在 log 中，不向其他组合传播。

子进程每个长度开始时发送一条 stdout 事件，父进程在内存中计时。
达到超时上限后终止整个子进程；仍有不被支配的输入长度时，会重新加载模型继续。
模型加载也有相同的超时上限。不会创建 current_cell.txt、runner.pid 或 overview_report.md。
旧输出目录内已存在的这些文件不读取、不修改。

## 输出

- `<output-root>/<model>/<method>/history.csv`：每轮原始测量与状态。
- 同目录 `average.csv`：当前组合完成后更新，保留原均值计算与字段口径。
- `<output-root>/<model>/<experiment>/<method>/<task>/pred.json`：生成文本及样本元信息。
- `<state-dir>/logs/<model>/<method>/k<budget>_b<batch>_worker.log`：该组的完整输出及 traceback。
- 阈值型方法的 log 使用 `t<threshold>_b<batch>_worker.log`，CSV 和 pred.json 都记录 threshold。
- DuoAttention 的 log 使用 `s<sparsity>_b<batch>_worker.log`，CSV 和 pred.json 都记录 sparsity。

新 history 增加 attempted 列；已有 history 保持原表头写入，使用 phase/status 区分未执行记录。
写入阈值或稀疏率测试时，旧 history 若缺少 threshold/sparsity 列，会保留原记录并补上对应列。
旧记录未标明阈值，不会作为某个阈值已测过的依据；固定 budget 的历史跳过逻辑保持原样。

## 保持的测试口径

RULER 分词不添加特殊 token，按原方式截断或重复补足到指定输入长度。
每个长度固定种子 42，用第一批样本 warmup，再测量指定数量的完整 batch。
固定输出 token 数、greedy decode、方法 backend 和现有 prefill 策略保持不变。
模型加载、分词、输入搬运、解码文本不计入生成计时。
TTFT、TPOT、吞吐量及 prefill/decode CUDA allocator 峰值计算沿用原实现；测量指标输出两位小数，实验参数保留原精度。

Adakv/HeadKV 在 Llama、Qwen2/Qwen3 上，batch>1 且 `--serial-prefill auto/on` 时使用完整请求串行：
每次只 forward 一条 prompt，收集每层 packed KV 和 head_lens，按 `[sample, head]` 顺序合并，
重建长度及 varlen 偏移后使用原 batch decode。该路径不会再退回 `serial_kv_compression`。
逐条 prefill 同样调用方法自身的 prepare_inputs_for_generation，保持位置参数和初始化流程一致。
合并失败直接记录失败，不悄悄替换成批量 prefill；无需额外跑两条样本做合并探测。
`--serial-prefill off` 保留原批量路径；batch=1 使用原单条路径，ChatGLM 尚不启用该适配。
批量模式整批执行原生 Linear、attention 和 KV 压缩，不安装逐样本 Linear 包装。
串行验证也直接对照原生批量结果；数值差异如实记录，不通过改动批量计算口径消除差异。
完整请求串行改变了 BF16 forward 的 batch 形状及自适应选择，不能保证与批量路径逐 token 相同。
KV 合并本身按原 dtype 复制，不重新计算或重新压缩，逐层释放源 cache。
同时修正了原 Qwen2/Qwen3 batch decode 的 batch/head 混排：varlen 输出按 `[B, H, Q, D]`
还原为 `[B, Q, H, D]`，不再交换 batch 与 head；klen_sum 按全部 `B*H` 个 head 累加。
该修正适用于串行和批量 prefill 后的 batch decode，不改变 KV 压缩算法或 varlen kernel。
动态 cache 的无界容量标志 -1 不再当作实际容量计算，避免空 cache 的历史长度为负、
导致未显式提供 position_ids 时 RoPE 从负位置开始。
重测旧组合时使用新的 OUTPUT_ROOT，避免已有 history 将旧策略下已完成的组合跳过。
Quest/ClusterKV 的 `--efficiency-backend`、RetroInfer/MagicPig 原生生成路径均保留。

## 计时边界

每个 step 在前后同步模型使用的 CUDA 设备，计入前向和 `argmax`。
Quest、ClusterKV 以及 MagicPig 的 CPU/GPU backend 使用同一套逐步计时。
MagicPig 的所有串行请求或 prefill chunk 合计为首步，首个 token 产生后切换 decode 显存监控。
MagicPig 的计时观察器只在本次调用期间安装，结束或失败后恢复。
benchmark 统计来自统一的 `EfficiencyMeter`，不读取 MagicPig 自带的 `TPOT` 字段。

输出 32 个 token 对应首步加 31 个 decode step；TPOT 等于后 31 步总耗时除以 31。
逐步计时不包含步外的 mask/序列扩展；串行 cache 合并也在首步之外。
生成总时间在整体前后同步，包含 cache 合并、mask 更新和原生方法内部清理。
因此总时间通常大于 TTFT 加 decode 时间；TTFT 不代表包含全部调度开销的请求延迟。

## 验证代码归档

tests 和 tools 已从正常 benchmark 目录移除。仍被运行流程使用的 batch 错误识别并入 runtime.py；
适配逻辑已在 adaptive.py，历史检查脚本无需合入生成主流程。
原回归测试、检查脚本和本次修改前代码备份在 `/tmp/efficency_vram-threshold-cleanup-ARLi8g/`。
