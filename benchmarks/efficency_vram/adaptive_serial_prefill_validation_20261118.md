# Adakv / HeadKV 串行适配验证（2026-11-18）

2026-11-19 更新：默认策略已改为完整请求串行 prefill，再合并 packed KV、执行 batch decode。
下面的 `serial_kv_compression` 结果是历史方案；当前策略及输出差异说明见 README。

本文保留当时的验证过程。2026-11-19 清理目录后，文末 tests/tools 命令不再作为当前入口；
原验证代码归档于 `/tmp/efficency_vram-threshold-cleanup-ARLi8g/`，正常测试统一使用 `pred.py`。

## 最终采用的方式

本次只在 `benchmarks/efficency_vram` 内适配，没有修改 Adakv、HeadKV 的核心代码，
也没有处理 CakeKV。继续使用原压缩算法、packed KV、cache update 和 varlen batch decode。

完整请求逐条 prefill 的最简方案已经实现并测试，但没有通过输出一致性检查：
Llama、batch=4、输入 4096、budget=1024、输出 32 时，Adakv 有 34 个 token 不一致，
HeadKV 有 54 个 token 不一致。检查发现第一层 KV 可以完全相同，但后续层的自适应选择发生变化；
完整请求从 batch 改为单条 forward 后，BF16 数值路径发生变化，缓存正确拼接并不足以保证输出不变。
实验路径保留在 `AdaptivePrefillAdapter` 和 `generate_serial_prefill()` 中，默认不启用。

最终默认采用 `prefill_mode=serial_kv_compression`：

1. 保留原 batch 的 projection、attention、FFN 和首 token 计算。
2. 每层调用原 `kv_cluster.update_kv()` 时，按样本逐条执行压缩。
3. 按 `[sample, head]` 顺序拼接 packed K/V，重建完整的长度及偏移元数据。
4. 使用原 batch decode，生成结束或异常时恢复运行时包装。

`--serial-prefill auto/on` 为这两个方法选择此模式；`off` 保留原批量 prefill；
batch=1 和 ChatGLM 保持批量路径。它只减少压缩阶段的临时显存，不能等同于整条请求串行 prefill。

## 验证口径

- 环境：PyTorch 2.8.0+cu128、Transformers 4.57.1、FlashAttention 2.7.3。
- GPU 0 与其他进程共用；只评价本进程 CUDA allocator 的 `peak_allocated_mib`，不评价效率。
- 每组使用 4 条不同的 RULER `niah_single_1` 样本；通过原 runner 归一化输入长度；固定生成 32 个 token。
- 比较 `off` 和 `auto` 的所有输出 token IDs，不使用数值容差。
- 逐层计算 prefill K/V 的 SHA-256 字节摘要，并比较全部 `head_lens`，避免只检查答案恰好相同。
- 核查脚本会把摘要计算的 CPU 时间计入生成耗时，因此其中的时间指标不能用于效率比较。
- 表中的节省量由原始数值相减后取两位小数，可能与已四舍五入的两列直接相减相差 0.01 MiB。

## Llama-3.1-8B，batch=4

以下 12 组的 prefill K/V、head 长度和输出 token 全部一致，共比较 1536 个输出 token。

| 方法 | budget | 输入长度 | 原批量峰值 MiB | 压缩串行峰值 MiB | 节省 MiB | 降幅 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adakv | 128 | 4096 | 16573.03 | 16541.04 | 31.99 | 0.19% |
| adakv | 128 | 65536 | 31581.14 | 31069.14 | 512.00 | 1.62% |
| adakv | 1024 | 4096 | 18588.17 | 18303.30 | 284.88 | 1.53% |
| adakv | 1024 | 65536 | 33316.59 | 32804.59 | 512.00 | 1.54% |
| adakv | 8192 | 4096 | 24357.63 | 24357.39 | 0.25 | 0.00% |
| adakv | 8192 | 65536 | 48568.98 | 46693.37 | 1875.60 | 3.86% |
| headkv | 128 | 4096 | 16576.60 | 16544.61 | 31.99 | 0.19% |
| headkv | 128 | 65536 | 31584.58 | 31072.58 | 512.00 | 1.62% |
| headkv | 1024 | 4096 | 18411.27 | 18171.07 | 240.20 | 1.30% |
| headkv | 1024 | 65536 | 33350.01 | 32838.01 | 512.00 | 1.54% |
| headkv | 8192 | 4096 | 24357.76 | 24357.39 | 0.37 | 0.00% |
| headkv | 8192 | 65536 | 49838.57 | 46865.37 | 2973.20 | 5.97% |

64K 各配置的 decode 峰值完全相同，说明节省来自 prefill 临时张量，而非减少最终 KV 预算。
4K、budget=8192 不需要压缩，显存基本不变；小于 1 MiB 的差异不视为有效收益。

## Qwen2.5-7B-Instruct-1M 接口检查

补测 batch=4、4096、budget=1024，并首先执行 `auto`，覆盖 Qwen 的冷启动初始化和方法恢复。
两个方法的 KV 字节摘要、head 长度及 256 个输出 token 全部一致。与 Llama 合计 14 组、1792 个输出 token。

| 方法 | 原批量峰值 MiB | 压缩串行峰值 MiB | 节省 MiB |
| --- | ---: | ---: | ---: |
| adakv | 17024.86 | 16945.83 | 79.03 |
| headkv | 17009.77 | 16840.20 | 169.57 |

Qwen 的更长输入及其他预算未在本轮覆盖。

## 正式入口和单元测试

两个方法分别通过了 `run_method_batch.py` 的冷启动检查：batch=4、4096、budget=1024、
warmup=0，首次直接执行默认 `auto`，再通过 `--serial-prefill-verify` 对照原批量路径。
均为 `status=passed`、`prefill_mode=serial_kv_compression`、`serial_prefill_verify=match`。

6 个单元测试覆盖不同样本长度、packed 顺序、全部偏移重建、非法元数据拒绝、
请求状态重置，以及正常/异常退出后的方法恢复。

## 复现

在外部激活运行环境，按需设置 `CUDA_VISIBLE_DEVICES` 和 Hugging Face 缓存环境变量：

```bash
python benchmarks/efficency_vram/tools/check_adaptive_serial_prefill.py \
  --model llama-3.1-8b \
  --method adakv \
  --input-lengths 4096 65536 \
  --batch-sizes 4 \
  --budgets 128 1024 8192 \
  --output-length 32 \
  --output-dir benchmarks/efficency_vram/output/adaptive_serial_check

python -m unittest discover -s benchmarks/efficency_vram/tests -v
```

更换 `--method headkv` 可检查另一个方法。原始结果位于
`output/adaptive_serial_validation_20261118/`：根目录保留完整请求串行的失败结果，
`serial_compression/` 保存通过的 token IDs、字节摘要和显存数据，`worker/` 保存正式入口检查记录。

未覆盖完整 tradeoff 预算矩阵、batch=16 或全部模型；不能把当前测试范围扩展为任意输入的逐 token 保证。
