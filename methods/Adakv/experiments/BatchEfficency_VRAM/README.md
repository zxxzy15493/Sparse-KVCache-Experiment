# AdaKV (HeadKV) 批处理效率与显存实验

基于 AdaKV 现有的批量输入逻辑（左 padding、按 padded 宽度切片）与 HeadKV 计时/显存口径，
分别测量：

- `run_efficency.sh` → `Efficency.sh` → `pred_efficency.py`：TTFT、TPOT 和 decode 吞吐量。
- `run_vram.sh` → `VRAM.sh` → `pred_vram.py`：GPU 峰值 allocated / reserved 显存，以及 decode 阶段显存采样。

两条入口各自启动 Python 进程，保存独立的测量文件；没有合并测量的入口。
两个 `run_*.sh` 都直接列出命令，按需取消注释即可。

```text
BatchEfficency_VRAM/
├── config/                # dataset2prompt / dataset2maxlen / model2path / model2maxlen
├── run_efficency.sh        # 效率实验命令列表
├── run_vram.sh             # 显存实验命令列表
├── Efficency.sh / VRAM.sh  # 单任务 shell 入口
├── pred_efficency.py       # 效率专用 Python 入口
├── pred_vram.py            # 显存专用 Python 入口
├── pred.py                # 公共批量推理与结果保存逻辑
├── decode_memory.py       # decode 阶段显存采样与峰值统计
└── utils.py
```

## 运行

激活安装了 AdaKV / HeadKV 依赖的 Python 环境（例如 `/home/kv/miniconda3/envs/kv/bin/python`，
可通过 `PYTHON` 环境变量指定）：

```bash
cd methods/Adakv/experiments/BatchEfficency_VRAM

# 只测效率
bash run_efficency.sh

# 只测显存；作为另一组独立实验运行
bash run_vram.sh
```

也可以直接指定单个任务，位置参数顺序为：

```text
MODEL BENCHMARK TASK BATCH_SIZE CONTEXT_LENGTH NUM_EXAMPLES BUDGET METHOD
HEAD_CHOICE BETA TEMP MAX_LEN FIXED_OUTPUT_LENGTH
```

```bash
# batch=4、RULER 4K、全部样本；固定生成 32 token，测 TTFT/TPOT
bash Efficency.sh llama-3.1-8b RULER niah_single_1 4 4096 -1 1024 AdativeKV random 1.5 1.0 0 32

# 相同批量与输入参数；固定生成 2 token，单独测显存
bash VRAM.sh llama-3.1-8b RULER niah_single_1 4 4096 -1 1024 AdativeKV random 1.5 1.0 0 2

# LongBench 的 context_length 使用占位值 0
bash Efficency.sh llama-3.1-8b LongBench narrativeqa 2 0 -1 1024 AdativeKV random 1.5 1.0 0 32
```

前 3 个参数必填，其余默认值依次为
`1 0 -1 1024 AdativeKV random 1.5 1.0 0 32`（显存入口末尾默认 2）。
RULER 的 `CONTEXT_LENGTH=0` 默认使用 65536；`FIXED_OUTPUT_LENGTH=0` 使用任务默认长度并逐样本检查 EOS。
`run_efficency.sh` 的示例固定生成 32 token，`run_vram.sh` 的示例固定生成 2 token。
保留 `PYTHON`、`DEVICE`、`DTYPE`、`DATA_DIR` 环境变量；`WARMUP` 可指定预热次数，
也可用 `WINDOW_SIZE/KERNEL_SIZE/POOLING/FLOOR/SKIP/NORMALIZE` 覆盖 HeadKV 运行参数。
效率默认预热 7 次，显存默认预热 2 次。预热使用第一批样本，结果不会写入正式测量文件。

Python 参数增加可选 `--warmup`，每个入口固定自己的测量类型：

```bash
python pred_efficency.py --model llama-3.1-8b --benchmark ruler \
  --task niah_single_1 --batch_size 2 --context_length 4096 \
  --num_samples 10 --max_capacity_prompts 1024 --method AdativeKV \
  --fixed_output_length 32 --warmup 7 --save_dir ./results/efficency-demo

python pred_vram.py --model llama-3.1-8b --benchmark ruler \
  --task niah_single_1 --batch_size 2 --context_length 4096 \
  --num_samples 10 --max_capacity_prompts 1024 --method AdativeKV \
  --fixed_output_length 2 --warmup 2 --save_dir ./results/vram-demo
```

## 测量口径

效率沿用 AdaKV 模型内部的计时字段 `model.model._timing`（由
`headkv/adaptive_llama_hijack.py` 的 `adaptive_LlamaModel_forward` 写入）：

- `ttft`：`_timing["ttft"]`，单位秒，为一次请求从 prefill 开始到首 token 的耗时。
- `TPOT`：`_timing["tpot"]` **乘以 1000**，单位毫秒。hijack 内部 `tpot` 是
  `decode_time / decode_steps`（秒/步），这里统一换算成毫秒，与参考实现的毫秒口径一致。
- `decode_latency`：`_timing["decode_time"]`，单位秒；`latency`：`_timing["latency"]`，单位秒。
- `throughput`：`实际 batch_size × decode_steps / decode_latency`，单位 tokens/s。

指标不包含模型加载、cache 初始化、tokenize、文件写入等模型计时区间外的工作。
TTFT 是整个 batch 完成 prefill 的耗时，TPOT 是整个 batch 的一次 decode 步骤耗时，两者都不除以 batch 大小。
只有首 token、没有 decode 步骤时，`TPOT` 保存为 `null`，汇总时排除。

注意：HeadKV 的计时只在 LLaMA 家族（`adaptive_LlamaModel_forward`）里写入。
Qwen2 / ChatGLM 的 hijack 不写 `_timing`，此时 `pred.py` 回退到内部的 `EfficiencyTimer`：
包装语言模型模块的 `forward`（LLaMA/Qwen 为 `model.model`，ChatGLM 为 `model.transformer.encoder`），
把第一次 forward 记为 prefill（TTFT），后续 forward 累加为 decode 步，
再按与 `_timing` 相同的公式产出 `ttft/tpot/decode_steps/decode_latency/latency` 字段，
保证 llama/qwen/glm 三类模型口径一致。

显存沿用记忆实验的口径：模型加载和预热完成后，每个正式 batch 开始前，对所有可见 GPU
执行 `torch.cuda.reset_peak_memory_stats`；batch 生成完成、cache 释放之前，读取
`torch.cuda.max_memory_allocated` 和 `torch.cuda.max_memory_reserved`。
统计包含常驻模型权重、该批次 KV cache 和推理临时 tensor，不包含模型加载期间的瞬时峰值。

- `peak_alloc_by_device`、`peak_reserved_by_device`：每张可见 GPU 的峰值，单位 MiB。
- `peak_alloc`、`peak_reserved`：各 GPU 峰值中的最大值；不是多卡峰值之和。

VRAM 结果另含 decode 阶段指标（均为 allocated 显存、单位 MiB）：

- `decode_avg_alloc`：每次实际 decode forward 完成后，同步所有可见 GPU，采样
  `torch.cuda.memory_allocated`，对步骤采样值取算术平均。
- `decode_peak_alloc`：第一次 decode 开始前重置峰值计数器，到生成结束时读取 decode 阶段的 allocated 峰值。

`decode_avg_alloc_by_device`、`decode_peak_alloc_by_device` 保存各 GPU 的对应值；
标量值取各 GPU 平均值或峰值中的最大值。`decode_num_steps` 是实际执行的批量 decode 步数，不乘以 batch 大小。
若仅生成 prefill 的首 token、没有 decode，则两个指标均为 `null`。
prefill 峰值在 decode 开始前保存，之后与 decode 峰值合并，因此原有全程 `peak_alloc` / `peak_reserved` 指标仍然保留。
监测只在正式 VRAM 测量中启用；预热和效率入口不安装采样钩子。

显存为当前进程的 PyTorch CUDA allocator 统计，不是整张卡的 `nvidia-smi` 占用。

## 结果

shell 入口分别保存至 `results/efficency/` 和 `results/vram/`；日志分别位于 `log/efficency/` 和 `log/vram/`。
目录包含模型、数据集、上下文长度、batch 大小、预算、方法与固定输出长度。

- `<task>.efficency.jsonl` / `<task>.vram.jsonl`：每个正式 batch 一条测量记录。
- `<task>.efficency.pred.jsonl` / `<task>.vram.pred.jsonl`：每条样本的预测和原始答案字段。
- `<task>.efficency.summary.json` / `<task>.vram.summary.json`：效率均值或显存峰值，按实际 batch 大小分组汇总。
- `<task>.efficency.run.json` / `<task>.vram.run.json`：`run.json` 配置与本批输入的指纹（见下）。
- `<task>.efficency.summary.csv` / `<task>.vram.summary.csv`、`<task>.efficency.summary.txt` / `<task>.vram.summary.txt`：
  与 summary.json 同名的表格汇总，csv 列名带单位，多卡显存字段展开为 `<device>_<field>`；txt 为可直接阅读的文本表格。

## 测量严谨性改进

### run.json 清单与一致性校验

每个 save_dir 对应一个任务、测量类型组合的 `run.json`（`SCHEMA_VERSION=1`），记录
`measurement/model/benchmark/task/batch_size/context_length/max_len/num_samples/dtype/device/`
`fixed_output_length/warmup/method/head_choice/max_capacity_prompts/beta/temp/window_size/kernel_size/`
`pooling/floor/data_dir/data_file(绝对路径)/max_new_tokens/input_sha256/loader_fingerprint/seed=42`。

- `loader_fingerprint`：`load_model` 源码的 sha256，模型加载入口改变即视为不一致。
- `input_sha256`：对规划中的**全部样本**（而非仅剩余样本）按 `{"index","input"}` 逐条计算的 sha256，
  续跑时保持可复现。
- 一致性校验在 `load_dataset` 中进行：`run.json` 已存在但任一配置键不同 → 报错要求更换 save_dir；
  测量文件已存在而 `run.json` 缺失 → 报错（不会自动补写覆盖）。已保存的批次若
  schema 版本缺失/不符、indices 不在规划批次内、`len(indices)!=batch_size`、或与已完成集合有交集 → 视为无效报错。
- 效率与显存各有独立 `run.json`（文件名含 measurement），互不冲突。

### 三段吞吐与加权 TPOT（仅效率）

在 `run_batch` 给效率记录新增派生字段（不新计时）：

- `prefill_tokens_per_s = sum(input_tokens) / ttft`
- `decode_tokens_per_s = sum(c-1 for c in output_tokens) / decode_latency`（decode_steps=0 时为 null）
- `output_tokens_per_s = sum(output_tokens) / latency`
- 保留 `throughput` 不变。

`write_summary` 效率分支新增 `weighted_tpot_ms = sum(decode_latency)*1000 / sum(decode_steps)`
（decode_steps 总和为 0 时为 null），并新增 `prefill_tokens_per_s`、`decode_tokens_per_s` 组间汇总。
旧效率记录通过 `load_efficiency_records` 回填这些派生字段，避免新旧记录口径不一致。

### summary.csv / summary.txt

`write_summary` 在写入 summary.json 之后追加同名 `.summary.csv`（csv.DictWriter，列名含单位，
多卡字段展开为 `<device>_field`）与 `.summary.txt`。效率表列：
`Batch/Batches/Samples/TTFT(s)/TPOT(ms)/weightedTPOT(ms)/throughput(token/s)/prefill(token/s)/decode(token/s)`；
显存表列：`Batch/Batches/Samples/peak_alloc(mib)/peak_reserved(mib)` + 逐卡显存。无可用值时显示 `N/A`。
summary.json 键保持不变（向后兼容）。

### 多 GPU 设备发现与同步

新增 `model_cuda_devices(model)`（从 `model.device`、`hf_device_map`、`parameters()`、`buffers()` 枚举，
过滤 CUDA 并按 index 排序）与 `synchronize_devices(devices)`。`reset_peak_memory_stats` /
`collect_peak_memory_mib` 接受可选 `devices` 参数（缺省仍遍历所有可见 GPU）；
`run_batch` 一次计算 `devices=model_cuda_devices(model)` 并传给显存侧；效率侧在读取 `_timing`
前后用 `synchronize_devices(devices)` 同步。返回字段名（`peak_alloc_by_device` 等）与数值含义不变。

### CUDA 冒烟与 CPU 单测

`tests/cuda_smoke.py` 用随机两层 Qwen3（hidden 256、layers 2、heads 4、kv_heads 2、head_dim 64、vocab 100）
配合 `unittest.mock` 替换 `AutoModelForCausalLM.from_pretrained` / `AutoTokenizer.from_pretrained`，
对真实 Adakv/HeadKV patch 跑 B=1/2/4、尾批、左 padding 不等长、两个入口；结果写 `results/cuda-smoke/`。
无 CUDA 时明确打印 SKIP；若 HeadKV 在随机小模型上确实无法驱动（如 kernel 不适用），脚本打印 SKIP 原因而不必然失败。
`tests/test_batch_measurements.py` 为不依赖 CUDA 的纯 CPU 单测，覆盖 `iter_batches`、续跑过滤、
按 padded 宽度切片、run.json 一致性校验（改配置键→报错）等。

效率汇总包含 `mean_ttft_s`、`mean_tpot_ms`、
`throughput_tokens_per_s`（组内总 decode token 槽位数 / 总有效 decode 时间），
以及 `mean_throughput_tokens_per_s`（各批次吞吐量的算术平均），均为 tokens/s。

VRAM 汇总包含 `peak_alloc_mib`、`peak_reserved_mib`、
`decode_avg_alloc_mib` 和 `decode_peak_alloc_mib`，并保留逐 GPU 数据。
平均值按各批次实际 decode 步数加权汇总，峰值取组内最大值；没有 decode 的批次不参与这两个指标的汇总。

尾批照常测量并独立分组，避免混入完整 batch 的均值。
续跑按测量记录中的样本索引跳过已完成批次；相同 `save_dir` 中的两类实验文件名独立，互不跳过。
全部测量完成时，重复运行会跳过模型加载和预热。
已有效率结果保留 decode 步数和延迟时，按原命令重新执行会直接计算并补写吞吐量及汇总，无需重新推理。
已有 VRAM 结果缺少 decode 指标时，按原命令重新执行会自动补测，并按样本索引替换旧测量和预测记录。
这里保存性能测量及预测，不自动运行准确率评测。

## 验证

```bash
# 语法检查（不执行 GPU 推理）
python -m py_compile pred.py pred_efficency.py pred_vram.py decode_memory.py utils.py
bash -n Efficency.sh VRAM.sh run_efficency.sh run_vram.sh

# 纯 CPU 单测（不依赖 CUDA 与权重）
python -m unittest discover -s tests -v

# 真实 HeadKV patch + 随机小模型的 CUDA 冒烟（需 GPU；无 CUDA 时打印 SKIP）
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 python tests/cuda_smoke.py
```

冒烟覆盖 B=1/2/4、不同输入长度（左 padding 不等长）、尾批与两个测量入口；结果在 `results/cuda-smoke/`。
本仓库不自动运行 GPU 冒烟，由使用者按需执行。
