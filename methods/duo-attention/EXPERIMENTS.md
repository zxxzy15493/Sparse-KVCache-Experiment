# DuoAttention Experiment Guide

DuoAttention's **accuracy evaluation** is integrated into the project's unified evaluation framework via `run_accuracy.sh`, which provides preset subcommands (`overview`, `budget`, `full`) to run complete experiment suites with a single command.

---

## 1. Accuracy Evaluation

The entry point for all accuracy benchmarks is:

```
bash run_accuracy.sh <subcommand>
```

Available subcommands:

| Subcommand | Description |
|------------|-------------|
| `overview` | LongBench overview (sparsity=0.5, 3 models) + RULER overview (4k-64k, 2 models) |
| `budget`   | LongBench budget sweep (sparsity 0.6/0.7/0.8/0.9, 2 models) + RULER budget (64k, 128k, 192k) |
| `full`     | overview + budget combined |

### 1.1 LongBench

**Datasets** (overview — 12 tasks): narrativeqa, qasper, 2wikimqa, musique, gov_report, multi_news, triviaqa, samsum, passage_count, passage_retrieval_en, lcc, repobench-p
**Datasets** (budget — 4 tasks): narrativeqa, qasper, trec, lcc
**Models**: llama-3.1-8b, qwen-2.5-7b, glm-4-9b-1m

```bash
cd methods/duo-attention/eval/scripts

# LongBench overview: sparsity=0.5, all 12 tasks, 3 models
bash run_accuracy.sh overview

# LongBench budget sweep: sparsity=0.6/0.7/0.8/0.9, 4 tasks, 2 models (llama/qwen)
bash run_accuracy.sh budget
```

**Experimental parameters — LongBench:**
- `--sparsity`: Ratio of attention heads to prune. Lower = more heads retained. Default overview=0.5, budget sweep=0.6/0.7/0.8/0.9.
- `--max_samples`: Max samples per task (default: 500, use 1 for quick smoke test)
- Attention pattern directory: `../../attn_patterns/`

For **individual test** (single model, single sparsity, single task):

```bash
cd methods/duo-attention/eval/scripts
# e.g. LongBench: model=qwen, sparsity=0.5, 1 sample
bash run_accuracy.sh --dataset longbench --model qwen --sparsities "0.5" --max_samples 1
# --model: llama, qwen, glm  |  --sparsities: sparsity ratio  |  --max_samples: 1 for quick test
# Lower sparsity = more heads retained
```

### 1.2 RULER

**Tasks** (overview): niah_single_1, niah_single_2, niah_single_3, niah_multikey_1, niah_multikey_2, niah_multikey_3, niah_multivalue, niah_multiquery, vt, cwe, fwe, qa_1, qa_2
**Tasks** (budget 64k): niah_single_1, niah_single_2, niah_single_3, vt, fwe, qa_1, qa_2 (+ cwe for qwen)
**Tasks** (budget 128k/192k): niah_single_1, niah_single_2, niah_single_3, vt
**Models**: llama-3.1-8b, qwen-2.5-7b-1m
**Sequence lengths**: 4096, 8192, 16384, 32768, 65536 (overview); 65536 (budget 64k); 131072 (budget 128k); 196608 (budget 192k)

```bash
cd methods/duo-attention/eval/scripts

# RULER overview: sparsity=0.5, 4k-64k, all tasks
bash run_accuracy.sh overview

# RULER budget sweep (64k/128k/192k): sparsity=0.6/0.7/0.8/0.9 (64k), 0.5 (128k/192k)
bash run_accuracy.sh budget
```

**Experimental parameters — RULER:**
- Same as LongBench above (`--sparsity`)
- `--ruler_seq_lengths`: Sequence lengths to evaluate, e.g. "4096 8192 16384 32768 65536"
- `--ruler_num_samples`: Samples per task (default: 50)
- Data directory: `../../Sparse-KVCache-Experiment/benchmarks/ruler/benchmark_root/`
- Sparsity passed via `--fixthreshold` (RULER framework naming), mapped to `--sparsity` internally

For **individual test** (single model, single sparsity, single seq length):

```bash
cd methods/duo-attention/eval/scripts
# e.g. RULER: model=qwen, sparsity=0.5, seq_length=4096, 1 sample
bash run_accuracy.sh --dataset ruler --model qwen --sparsities "0.5" --ruler_seq_lengths "4096" --ruler_num_samples 1
# --model: llama, qwen  |  --sparsities: sparsity ratio  |  --ruler_seq_lengths: sequence length
# Lower sparsity = more heads retained
```

### 1.3 LongBench-v2

**Models**: llama-3.1-8b, qwen-2.5-7b-1m

```bash
bash methods/duo-attention/eval/LongBench-v2/run.sh
```

**Experimental parameters — LongBench-v2:**
- `--sparsity`: Default 0.5
- `--window_size`: 32, `--kernel_size`: 7, `--pooling`: maxpool, `--floor`: 0.2

### 1.4 GSM8K

**Model**: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

```bash
bash methods/duo-attention/eval/GSM8K/run_duo.sh
```

**Experimental parameters — GSM8K:**
- `--sparsity`: 0.5
- `--num_shots`: 8 (few-shot examples)
- `--max_new_tokens`: 10000

---

## 2. Efficiency (Latency and GPU Memory)

Located at `methods/duo-attention/eval/efficiency/`; not included in `run_accuracy.sh`.

### 2.1 Efficiency Overview

```bash
cd methods/duo-attention/eval/efficiency
bash efficiency_overview.sh
```

### 2.2 Efficiency Budget

```bash
cd methods/duo-attention/eval/efficiency
bash efficiency_budget.sh
```

**Experimental parameters — Efficiency:**
- `--model`: Model name (e.g. llama-3.1-8b, qwen-2.5-7b)
- `--models`: Batch mode — space-separated list of models
- `--input_lengths`: Input token lengths

---

## 3. Memory (GPU Memory Measurement)

Located at `methods/duo-attention/eval/efficiency/`. Adapted from the efficiency code — **measures GPU memory only**.

```bash
cd methods/duo-attention/eval/efficiency
bash memory.sh
```

---

## 4. Recall

Located at `methods/duo-attention/eval/recall/`; not included in `run_accuracy.sh`.

```bash
cd methods/duo-attention/eval/recall
bash run_recall.sh
```

---

## 5. Time (Latency)

Located at `methods/duo-attention/eval/time/`; measures **sub-component latencies** via CUDA Events.

### Prefill Phase Components:
prefill, pattern, attn, write_cache, others

### Decode Phase Components:
decode, attn, write_cache, others

### Overhead Components:
load, unload, retrieve

```bash
cd methods/duo-attention/eval/time
bash run.sh
```

**Experimental parameters — Time:**
- `--model`: Model name (e.g. llama3.1-8b-128k)
- `--input_max_token`: Input sequence length (e.g. 4096, 65536)
- `--sparsity`: Sparsity level (e.g. 0.5)