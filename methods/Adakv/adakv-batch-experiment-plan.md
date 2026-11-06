# Adakv 多批次实验适配计划（BatchExperiment + BatchEfficency_VRAM）

## Context / 背景

Adakv（HeadKV/AdaKV）现有实验脚本基本只跑 `batch_size=1`：

- 准确率：`experiments/LonBench/run_longbench.py` 虽有 `--eval_batch_size` 但未在 `run_accuracy.sh` 暴露，切片/续跑/截断逻辑不完整；`experiments/RULER/run_ruler.py` 是逐条 `model.generate(...)[0]` 处理，无批量。
- 效率/显存：`experiments/efficiency/efficiency.py` 与 `experiments/memory/memory.py` 都是**单条 prompt** 的手工 prefill+decode 循环（`time.perf_counter` 计时 / `torch.cuda.max_memory_allocated` 取峰值）。

minference 与 retroinfer 已分别通过 `experiments/BatchExperiment`（批量准确率）与 `experiments/BatchEfficency_VRAM`（批量 TTFT/TPOT/吞吐量 + 显存峰值）完成适配。本次为 Adakv 补齐同等能力，沿用其参考实现的目录与脚本约定。

用户已确认：放在 `methods/Adakv/experiments/` 下新建 `BatchExperiment/` 与 `BatchEfficency_VRAM/`；**两者都做**；覆盖 **LongBench + RULER，模型 llama/qwen/glm**。

## 关键前提（已验证）

- HeadKV 注意力本身支持 batch：`headkv/snapkv_utils.py` 中 `update_kv` 按 `flat_heads = bsz * num_heads` 构造，`cu_qlen/cu_klen` 为 `bsz*num_heads+1` 长度，decode 走 varlen/SDPA 展平路径。
- Adakv 已在 `adaptive_LlamaModel_forward` 里把计时写进 `model.model._timing`（`prefill_time / decode_time / decode_steps / ttft / tpot / latency`），效率实验可直接读取，无需改模型。
- RULER 数据：`benchmarks/ruler/benchmark_root/<data_model>/synthetic/<seq_len>/data/<task>/validation.jsonl`；模型目录映射 llama→`llama-3.1-8b`，qwen→`qwen-2.5-7b-1m`。
- RULER 评分依赖 `experiments/RULER/eval/synthetic/constants.py`（`metric_fn`）+ `benchmarks/ruler/synthetic.yaml` 的任务配置；`experiments/RULER/eval/evaluate.py` 有 `cakekv.experiments...` 硬编码 import，**不要复用**，改为自包含。

## 复用的现有代码

| 用途 | 来源 |
|---|---|
| 模型加载 + HeadKV patch（`replace_llama/replace_qwen2/replace_chatglm`、glm 走 `get_class_from_dynamic_module`） | `experiments/LonBench/run_longbench.py`（`__main__` 加载段） |
| 运行时 config（`window_size/base_capacity/head_choice/beta/temp/kernel_size/skip/normalize/pooling/floor`） | `run_longbench.py` / `efficiency.py` 的 `apply_headkv_runtime_config` |
| LongBench prompt/maxlen | `run_longbench.py` 的 `model2prompt` / `dataset2maxlen` |
| RULER 任务模板与 tokens_to_generate | `experiments/RULER/run_ruler.py` 的 `TASKS` |
| LongBench 评分 | `experiments/LonBench/metrics.py` |
| RULER 评分 | `experiments/RULER/eval/synthetic/constants.py` |
| 效率口径（TTFT/TPOT/latency） | `experiments/efficiency/efficiency.py` |
| 显存口径（peak alloc/reserved） | `experiments/memory/memory.py` |
| 批量/续跑/输出结构、`decode_memory.py` | `methods/minference/experiments/BatchExperiment/pred.py`、`methods/retroinfer/benchmark/BatchEfficency_VRAM/{pred.py,decode_memory.py}` |

## A. `methods/Adakv/experiments/BatchExperiment/`（批量准确率）

新增文件：

- `pred.py` — 批量预测入口，支持 LongBench 与 RULER。
  - 参数：`--model`(别名或 HF 路径)、`--benchmark longbench|ruler|synthetic`、`--task`、`--batch_size`、`--context_length`(RULER)、`--num_samples`、`--method`、`--head_choice`、`--max_capacity_prompts`、`--beta/--temp/--window_size/--kernel_size/--pooling/--floor`、`--dtype`、`--device`、`--save_dir`。
  - 左 padding；`tokenizer(padding="longest", truncation=True, max_length=input_budget)`；`model.generate(max_new_tokens, do_sample=False, num_beams=1, eos/pad_token_id)`；按 **padded 宽度**切片（参考 minference，不能用各自未 pad 长度）。
  - 每批同时写 `<task>.jsonl`（逐样本 pred/answers/index）与 `<task>.batch_stats.jsonl`（indices、实际/配置 batch、padding 宽度、input/output tokens、latency、tokens/s）。
  - 续跑：按 `index`/`_id` 跳过已完成，重复 ID 报错。
- `eval.py` — LongBench 评分（`metrics.py`）→ 累写到 `result.json`。
- `evaluate.py` — RULER 评分：自包含加载 `synthetic.yaml` + `synthetic/constants.py`，输出 `summary.csv`（不复用旧 `eval/evaluate.py`）。
- `config/` — `dataset2prompt.json`、`dataset2maxlen.json`、`model2path.json`、`model2maxlen.json`（从 Adakv 代码内字典 + `run_accuracy.sh` 的 `MODEL_MAP` 提取）。
- `utils.py` — `load_data/iter_jsonl`。
- `BatchExperiment.sh` — 单任务入口：`MODEL BENCHMARK TASK [BATCH_SIZE=1] [CONTEXT_LENGTH] [NUM_EXAMPLES=-1]`；HeadKV 参数用环境变量（`METHOD/HEAD_CHOICE/MAX_CAPACITY_PROMPTS/BETA/TEMP/WINDOW_SIZE`）；跑 `pred.py` → `eval.py`/`evaluate.py`，日志写 `log/pred/<model>/<benchmark>...`。
- `run.sh` — 命令清单，默认取消注释：qwen/llama 的 `RULER niah_single_1 4096` batch=1/2/4，及 `LongBench narrativeqa` batch=1/2/4；其余任务注释备用。
- `README.md` — 用法、数据、输出、续跑、验证说明（对齐 minference `BatchExperiment/README.md`）。
- `tests/` — 最小单元测试：批次切分、续跑过滤、padded 宽度切片。

## B. `methods/Adakv/experiments/BatchEfficency_VRAM/`（批量效率 + 显存）

- `pred.py` — 复用 A 的批量推理，增加 `--measurement efficency|vram`、`--fixed_output_length`、`--warmup`：
  - efficency：读 `model.model._timing`，`ttft=ttft`、`TPOT=tpot`(ms)、`decode_latency/decode_steps`、`latency`、`throughput = batch_size * decode_steps / decode_latency`。
  - vram：每正式 batch 前对可见 GPU `reset_peak_memory_stats`，批后读 `max_memory_allocated/max_memory_reserved`（MiB，逐卡+最大值），并叠加 decode 阶段采样。
- `pred_efficency.py` / `pred_vram.py` — 两个入口（固定各自 measurement）。
- `decode_memory.py` — 移植 retroinfer 版；Adakv 无 `decode_forward`，改为包装 `model.model.forward`，在 `_timing["decode_steps"]` 递增时（decode 步）同步并采样 `torch.cuda.memory_allocated`。
- `Efficency.sh` / `VRAM.sh`（单任务，位置参数与 `BatchExperiment.sh` 一致 + `BUDGET/BETA/TEMP/FIXED_OUTPUT_LENGTH`）与 `run_efficency.sh` / `run_vram.sh`（命令清单，效率默认固定 32 token、显存固定 2 token，对齐参考）。
- `config/`、`utils.py`、`README.md`（测量口径、结果文件、续跑说明）。
- 结果：`<task>.efficency.jsonl`+`.pred.jsonl`+`.summary.json`；`<task>.vram.jsonl`+`.pred.jsonl`+`.summary.json`，按实际 batch 分组汇总。

## 需要注意 / 风险

- **混合长度批量的压缩选择**：`snapkv_utils` 的注意力打分未显式排除左 padding，可能影响选 KV。计划用 `batch_size=1` 与 `batch_size=2`（相同样本）对比输出做冒烟验证；若不一致，在 `update_kv` 前对 pad 位置做最小 mask。
- decode 的 varlen/SDPA 展平路径需在 `bsz>1` 下确认 `cu_klen` 正确（当前已按 `flat_heads=bsz*num_heads` 构造，需实测）。
- glm 走 `get_class_from_dynamic_module` + `replace_chatglm`，与 llama/qwen 加载分支不同，pred.py 需保留该分支。

## 验证方式

1. 语法：`bash -n` 所有 `.sh`；`python -m py_compile` 所有新增 `.py`。
2. 冒烟（单卡可用时）：
   - `cd methods/Adakv/experiments/BatchExperiment && bash run.sh`（或直接 `BatchExperiment.sh qwen RULER niah_single_1 2 4096 5`）。
   - 对比 `batch_size=1` 与 `2` 在相同样本上的 `pred` 一致性；检查 `.batch_stats.jsonl` 与波形。
   - LongBench：`BatchExperiment.sh qwen LongBench narrativeqa 2 0 5`，`eval.py` 出 `result.json`。
3. 效率/显存：`bash BatchEfficency_VRAM/run_efficency.sh` 与 `run_vram.sh`，确认 `.efficency.jsonl` 含 `ttft/TPOT/throughput`、`.vram.jsonl` 含 `peak_alloc/peak_reserved`，summary 按 batch 分组。
