# Adakv BatchExperiment

沿用 Adakv（HeadKV/AdaKV）现有多批次准确率逻辑：模型加载、HeadKV monkey patch 与运行时 config 与
`../LonBench/run_longbench.py` 一致，LongBench prompt/生成长度取自其 `model2prompt`/`dataset2maxlen`，
RULER 任务模板与生成长度取自 `../RULER/run_ruler.py` 的 `TASKS` 与 `benchmarks/ruler/synthetic.yaml`。
每次 `model.generate` 输入至多 `batch_size` 个**不同**样本，最后不足一批的样本也会执行。模型只加载一次，
按数据顺序分批，使用左侧 padding（`padding_side="left"`，`pad_token=eos`）与 attention mask。

生成结果按 **padding 后的公共宽度**切片（`out[:, padded_length:]`），而不是各样本未 padding 的实际长度；
每个输出先截断到首个 EOS。`eval.py` 负责 LongBench 逐样本评分，`evaluate.py` 负责 RULER 评分，
批量推理由 `pred.py` 执行。

## 运行

`BatchExperiment.sh` 可从任意工作目录调用；`run.sh` 在本实验目录下运行。使用已安装 PyTorch、Transformers、
FlashAttention 与 HeadKV 的环境。`pred.py` 会自动把 `methods/Adakv`（含 `headkv` 包）加入 `sys.path`。

```bash
cd methods/Adakv/experiments/BatchExperiment

# MODEL BENCHMARK TASK [BATCH_SIZE=1] [CONTEXT_LENGTH] [NUM_EXAMPLES=-1]
# LongBench 不使用 context_length；指定样本数量时用 0 占位。
bash BatchExperiment.sh qwen LongBench narrativeqa 4 0 10

# RULER，batch_size=4，context_length=4096，前 10 个样本。
bash BatchExperiment.sh qwen RULER niah_single_1 4 4096 10

# Synthetic/ruler 与 RULER 等价；不传 context_length 时默认为 65536。
bash BatchExperiment.sh llama Synthetic niah_single_1 2 8192

# run.sh 直接列出命令，按需取消注释后运行。默认运行 qwen/llama 的
# RULER niah_single_1（4K）与 LongBench narrativeqa，batch_size=1/2/4。
bash run.sh
```

`PYTHON`、`DEVICE`、`DTYPE`、`CUDA_VISIBLE_DEVICES` 可通过环境变量设置；默认值为 `python`、`auto`、`bf16`。
HeadKV 参数用环境变量传递：`METHOD`（默认 `AdativeKV`）、`HEAD_CHOICE`（默认 `random`）、
`MAX_CAPACITY_PROMPTS`（默认 `512`）、`BETA`（默认 `1.5`）、`TEMP`（默认 `1.0`）、`WINDOW_SIZE`（默认 `32`）。
`RESULTS_ROOT` 和 `LOG_ROOT` 可覆盖输出根目录。

也可以直接调用 Python：

```bash
python pred.py --model qwen --benchmark ruler --task niah_single_1 \
  --batch_size 4 --context_length 4096 --num_samples 10 \
  --save_dir results/pred/qwen/RULER/4096/batch-4 --device cuda:0
```

`--model` 支持 `config/model2path.json` 中的简称或完整模型 ID/本地路径。RULER 数据目录由模型名推断
（实现见 `pred.py:ruler_model_short`）。

## 数据

- LongBench 默认从 `THUDM/LongBench` 的 test split 加载，使用与 `run_longbench.py` 相同的 task prompt 与
  生成长度；非生成类任务（非 `trec/triviaqa/samsum/lsht/lcc/repobench-p`）套用 chat template。
  输入按 `config/model2maxlen.json` 推导的 `input_budget = model_max_len - max_new_tokens` 截断。
- RULER 默认读取 `benchmarks/ruler/benchmark_root/<data_model>/synthetic/<context_length>/data/<task>/validation.jsonl`。
  模型目录映射：Llama → `llama-3.1-8b`，Qwen → `qwen-2.5-7b-1m`。`validation.jsonl` 的 `input` 已带模板，直接使用。
- RULER 生成长度（`tokens_to_generate`）：`niah` 128、`variable_tracking`/`vt` 30、`cwe` 120、`fwe` 50、`qa` 32；
  任务名到基础模板的映射取自 `benchmarks/ruler/synthetic.yaml`。
- `context_length` 用于选择已生成的 RULER 数据，不会生成数据或截断 prompt。
- Python 参数 `--data_dir` 或脚本环境变量 `DATA_DIR` 可指定本地 JSONL 文件或任务目录。任务目录下
  LongBench 文件名为 `<task>.jsonl`，RULER 为 `validation.jsonl`。

## 输出与续跑

默认结果目录：

```text
results/pred/<model>/LongBench/batch-<batch_size>/
results/pred/<model>/RULER/<context_length>/batch-<batch_size>/
```

日志在对应的 `log/pred/` 目录。完整模型 ID 的 `/` 在目录名中替换为 `--`。

- `<task>.jsonl`：逐样本预测与参考答案，附实际批大小、输入和输出 token 数（RULER 另含 `input/outputs/others/truncation/length`）。
- `<task>.batch_stats.jsonl`：每批样本 ID、实际/配置批大小、padding 后输入宽度、生成耗时和有效输出 tokens/s。
  耗时统计 `generate` 调用并同步参与模型计算的 GPU，包含 prefill 和 decode；输出 token 数不含 EOS 及其后的 padding。
- `result.json`：LongBench 准确率，逐任务累加更新。
- `summary.csv`：RULER 各任务准确率与空预测数量。

相同输出目录再次运行时，会按样本 ID（LongBench 用 `_id`，RULER 用 `index`）跳过已有预测，再对剩余样本分批。
`--num_samples` 先限制原始数据，再执行续跑过滤。缺少 ID 时使用数据行序号；重复 ID 会报错。
直接运行 Python 时，请为不同批大小、上下文长度或数据源分别指定 `--save_dir`。

批次统计包含首次调用，未单独做 warmup。比较吞吐量时应使用相同任务、样本数量和生成参数，并按需要排除首次调用。

## 验证

```bash
python -m unittest discover -s tests -v
bash -n BatchExperiment.sh run.sh
python -m py_compile pred.py eval.py evaluate.py utils.py
```

测试覆盖批次切分（`iter_batches`）、续跑过滤（含 `--num_samples` 与重复 ID）与按 padded 宽度切片。
