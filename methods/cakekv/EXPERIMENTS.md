# CakeKV Experiment Guide

CakeKV's **accuracy evaluation** is integrated into the project's unified evaluation framework via `run_accuracy.sh`, which provides preset subcommands (`overview`, `budget`, `full`) to run complete experiment suites with a single command.

---

## 1. Accuracy Evaluation

The entry point for all accuracy benchmarks is:

```
bash run_accuracy.sh <subcommand>
```

Available subcommands:

| Subcommand | Description |
|------------|-------------|
| `overview` | LongBench overview (budget=1024, 3 models) + RULER overview (4k-64k, 2 models) |
| `budget`   | LongBench budget sweep (budgets 128/256/512/1024, 2 models) + RULER budget (64k, 128k, 192k) |
| `full`     | overview + budget combined |

### 1.1 LongBench

**Datasets** (overview — 12 tasks): narrativeqa, qasper, 2wikimqa, musique, gov_report, multi_news, triviaqa, samsum, passage_count, passage_retrieval_en, lcc, repobench-p
**Datasets** (budget — 4 tasks): narrativeqa, qasper, trec, lcc
**Models**: llama-3.1-8b, qwen-2.5-7b, glm-4-9b-1m

```bash
cd methods/cakekv/experiments/scripts

# LongBench overview: budget=1024, all 12 tasks, 3 models
bash run_accuracy.sh overview

# LongBench budget sweep: budgets=128/256/512/1024, 4 tasks, 2 models (llama/qwen)
bash run_accuracy.sh budget
```

**Experimental parameters — LongBench:**
- `--max_capacity_prompts` (budget): Controls how many KV cache entries are kept. Default overview=1024, budget sweep=128/256/512/1024.
- `--max_samples`: Max samples per task (default: 500, use 1 for quick smoke test)
- CakeKV-specific: `--gamma`, `--tau1`, `--tau2`, `--compress`, `--cascading`

For **individual test** (single model, single budget, single task):

```bash
cd methods/cakekv/experiments/scripts
# e.g. LongBench: model=qwen, budget=1024, 1 sample
bash run_accuracy.sh --dataset longbench --model qwen --cache_sizes "1024" --max_samples 1
# --model: llama, qwen, glm  |  --cache_sizes: budget value  |  --max_samples: 1 for quick test
```

### 1.2 RULER

**Tasks** (overview): niah_single_1, niah_single_2, niah_single_3, niah_multikey_1, niah_multikey_2, niah_multikey_3, niah_multivalue, niah_multiquery, vt, cwe, fwe, qa_1, qa_2
**Tasks** (budget 64k): niah_single_1, niah_single_2, niah_single_3, vt, fwe, qa_1, qa_2 (+ cwe for qwen)
**Tasks** (budget 128k/192k): niah_single_1, niah_single_2, niah_single_3, vt
**Models**: llama-3.1-8b, qwen-2.5-7b-1m
**Sequence lengths**: 4096, 8192, 16384, 32768, 65536 (overview); 65536 (budget 64k); 131072 (budget 128k); 196608 (budget 192k)

```bash
cd methods/cakekv/experiments/scripts

# RULER overview: budget=1024, 4k-64k, all tasks
bash run_accuracy.sh overview

# RULER budget sweep (64k/128k/192k): budgets=128/384/1024/4096
bash run_accuracy.sh budget
```

**Experimental parameters — RULER:**
- Same as LongBench above (`--max_capacity_prompts`, `--gamma`, `--tau1`, `--tau2`, `--compress`, `--cascading`)
- `--ruler_seq_lengths`: Sequence lengths to evaluate, e.g. "4096 8192 16384 32768 65536"
- `--ruler_num_samples`: Samples per task (default: 50)
- Data directory: `../../Sparse-KVCache-Experiment/benchmarks/ruler/benchmark_root/`

For **individual test** (single model, single budget, single seq length):

```bash
cd methods/cakekv/experiments/scripts
# e.g. RULER: model=qwen, budget=128, seq_length=4096, 1 sample
bash run_accuracy.sh --dataset ruler --model qwen --cache_sizes "128" --ruler_seq_lengths "4096" --ruler_num_samples 1
# --model: llama, qwen  |  --cache_sizes: budget value  |  --ruler_seq_lengths: sequence length
```

### 1.3 LongBench-v2

**Models**: llama-3.1-8b, qwen-2.5-7b-1m

```bash
bash methods/cakekv/experiments/LongBench-v2/run.sh
```

**Experimental parameters — LongBench-v2:**
- `--max_capacity_prompts`: Default 4096
- `--window_size`: 32, `--kernel_size`: 7, `--pooling`: maxpool, `--floor`: 0.2

### 1.4 GSM8K

**Model**: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

```bash
bash methods/cakekv/experiments/GSM8K/run.sh
```

**Experimental parameters — GSM8K:**
- `--max_capacity_prompts`: 360 (fixed budget)
- `--num_shots`: 8 (few-shot examples)
- `--max_new_tokens`: 10000

---

## 2. Efficiency (Latency and GPU Memory)

Located at `methods/cakekv/experiments/efficiency/`; not included in `run_accuracy.sh`.

### 2.1 Efficiency Overview

```bash
cd methods/cakekv/experiments/efficiency
bash efficiency_overview.sh
```

### 2.2 Efficiency Budget

```bash
cd methods/cakekv/experiments/efficiency
bash efficiency_budget.sh
```

**Experimental parameters — Efficiency:**
- `--model`: Model name (e.g. llama-3.1-8b, qwen-2.5-7b)
- `--models`: Batch mode — space-separated list of models
- `--input_lengths`: Input token lengths

---

## 3. Memory (GPU Memory Measurement)

Located at `methods/cakekv/experiments/efficiency/`. Adapted from the efficiency code — **measures GPU memory only**.

```bash
cd methods/cakekv/experiments/efficiency
bash memory.sh
```

---

## 4. Recall

Located at `methods/cakekv/experiments/recall/`; not included in `run_accuracy.sh`.

```bash
cd methods/cakekv/experiments/recall
bash run_recall.sh
```

---

## 5. Time (Latency)

Located at `methods/cakekv/experiments/time/`; measures **sub-component latencies** via CUDA Events.

### Prefill Phase Components:
prefill, pattern, attn, write_cache, others

### Decode Phase Components:
decode, attn, write_cache, others

### Overhead Components:
load, unload, retrieve

```bash
cd methods/cakekv/experiments/time
bash run.sh
```

**Experimental parameters — Time:**
- `--model`: Model name (e.g. llama3.1-8b-128k)
- `--input_max_token`: Input sequence length (e.g. 4096, 65536)
- `--max_capacity_prompts`: KV cache budget (e.g. 1024)