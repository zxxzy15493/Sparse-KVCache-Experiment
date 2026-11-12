# 长上下文效率与显存测试

## 执行流程

运行 `run_efficiency_vram_3base_18methods.sh` 后，调用关系如下：

```text
run_efficiency_vram_3base_18methods.sh
  -> tools/run_grid.py
    -> tools/run_method_batch.py
      -> pred.py 中的预测函数和 loaders
```

`pred.py` 是 worker 使用的函数模块，不提供独立命令行入口。

| 文件 | 用途 |
| --- | --- |
| `tools/run_grid.py` | 遍历模型、方法、budget、batch size 和输入长度，启动 worker |
| `tools/run_method_batch.py` | 对一个模型、方法、batch 和 budget 加载一次模型，逐个测试输入长度 |
| `pred.py` | 准备 RULER 输入、执行定长生成、测量效率与显存 |
| `tools/efficiency.py` | 统计 TTFT、TPOT 和生成耗时 |
| `tools/memory.py` | 统计 CUDA allocator 显存峰值 |
| `tools/reporting.py` | 保存逐轮记录并生成平均结果 |
| `tools/result_store.py` | 保存生成文本 |
| `tools/runtime.py` | 识别 OOM、超时并剪枝后续组合 |
| `tools/unsupported_batch.py` | 识别方法明确报告的 batch 限制 |

## 运行

```bash
conda activate /home/kv/miniconda3/envs/kv
bash benchmarks/efficency_vram/run_efficiency_vram_3base_18methods.sh
```

脚本通过 `CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"` 选择物理 GPU；进程内设备为 `cuda:0`。
模型、方法、budget、batch size、输入长度和数据文件都在启动脚本中设置。`tradeoff.sh` 使用 `{model}` 模板，按模型 alias 从 `benchmark_root` 读取对应的 RULER 数据。
矩阵按模型、方法、batch size 和 budget 分组；每个 budget 启动一个 worker，模型加载一次后逐个测试待测输入长度。切换 budget 时重新加载模型。

启动前确认 `DATA_FILE` 指向有效的 RULER `validation.jsonl`。每行需要包含 `index` 和 `input` 字段。
每组测试使用 `batch_size × measured_rounds` 条样本；runner 会在开始前检查数据条数。

`INPUT_LENGTHS` 中的数值是每条样本最终输入模型的 token 数。分词不添加特殊 token；长输入从中间截断，短输入通过重复原有 token 补足。`OUTPUT_LENGTH` 是固定生成 token 数。

## 结果

结果保存在启动脚本指定的 `output/` 目录下。CSV 按模型和方法保存，实验名已包含在 output 根目录中：

```text
output/<model alias>/<method>/
```

- `history.csv` 保存每轮测量和运行状态。
- `average.csv` 按 budget、batch size 和输入长度汇总测量均值。
- `pred.json` 保存生成文本，仍位于 `output/<model alias>/<experiment>/<method>/<task>/`，供人工核对。
- `logs/` 保存 worker 日志。

CSV 浮点值保留两位小数。

## 测量口径

| 指标 | 含义 |
| --- | --- |
| `ttft_ms` | Prefill 和首 token 选择的耗时 |
| `tpot_ms` | 首 token 后每个 decode step 的平均耗时 |
| `generation_latency_s` | 整个定长生成调用的耗时 |
| `decode_throughput_tokens_per_s` | 首 token 后生成阶段的吞吐量 |
| `peak_allocated_mib` / `peak_reserved_mib` | 生成期间 CUDA allocator 的峰值 |
| `decode_peak_allocated_mib` / `decode_peak_reserved_mib` | Prefill 后 decode 阶段的峰值 |

模型加载、分词、输入搬运、方法准备和输出解码不计入生成耗时。显存峰值包含常驻权重、输入、KV cache、logits 和 workspace；它不是纯 KV cache 大小，也不统计 CPU/offload 或绕过 PyTorch allocator 的扩展分配。

方法自身的限制和已验证情况见 [ISSUES.md](tools/ISSUES.md)。
