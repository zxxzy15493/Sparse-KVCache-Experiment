# 64k KV Cache Budget Tradeoff

在 RULER 合成任务上比较模型、方法与 KV cache budget 的精度表现。推理、数据准备和评测直接复用 [`benchmarks/ruler/run_experiment.sh`](../ruler/run_experiment.sh)。

- 模型：`llama-3.1-8b`、`qwen-2.5-7b-1m`
- 长度：`65536`
- 任务：`niah_single_3`、`vt`、`fwe`
- 样本数：每个任务 50 条
- 预算：`128 256 512 1024 2048 3072 4096 5120 6144 7168 8192`

运行前在外部终端激活环境：

```bash
conda activate /home/kv/miniconda3/envs/kv
```

分别在对应 GPU 上运行：

```bash
bash benchmarks/tradeoff/run_card0.sh
bash benchmarks/tradeoff/run_card1.sh
```

两个入口各自定义本卡的模型、方法、任务、长度和预算，并直接调用 RULER 的 `run_experiment.sh`。可以分别编辑两个脚本来控制每张卡的运行内容。数据写入本目录的 `data/`，预测和评测结果写入 `pred/`。

卡 0 上，Llama 方法列表以 `h2o` 结束，Qwen 方法列表以 `keyformer` 结束；卡 1 上，Llama 方法列表以 `keyformer` 结束，Qwen 方法列表以 `h2o` 结束。所有方法都使用上面同一组预算。

每个方法和预算都按 RULER runner 的流程执行数据准备、推理和评测；`summary.csv` 保存在对应的 `pred/` 子目录中。
