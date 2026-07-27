# Adakv Experiment Guide

Adakv's **accuracy evaluation** is integrated into the project's unified evaluation framework (`benchmarks/`). It shares the same entry points and output format as all other methods, enabling directly comparable results.

---

## 1. Accuracy Evaluation (Unified Framework)

Scripts for the following four benchmarks are located at `benchmarks/<dataset>/method_script/adakv-*.sh`.

### 1.1 LongBench

**Datasets**: narrativeqa, qasper, 2wikimqa, musique, gov_report, multi_news, triviaqa, samsum, passage_count, passage_retrieval_en, lcc, repobench-p, trec
**Models**: llama-3.1-8b, qwen-2.5-7b, glm-4-9b-1m

```bash


# Overview: all datasets, budget=1024
bash benchmarks/longbench/method_script/adakv-overview.sh

# Budget: narrativeqa, qasper, trec, lcc, budget=128/256/384/512/1024/4096
bash benchmarks/longbench/method_script/adakv-budget.sh
```

### 1.2 RULER

**Tasks**: niah_single_1/2/3, niah_multikey_1/2/3, niah_multivalue, niah_multiquery, vt, cwe, fwe, qa_1, qa_2
**Models**: llama-3.1-8b, qwen-2.5-7b-1m

```bash


# Overview: 4k-64k, budget=1024
bash benchmarks/ruler/method_scripts/adakv-overview.sh

# Budget: 64k only, budget=128/384/1024/4096
bash benchmarks/ruler/method_scripts/adakv-budget.sh
```

### 1.3 LongBench-v2

**Models**: llama-3.1-8b, qwen-2.5-7b-1m

```bash


bash benchmarks/longbenchv2/method_script/adakv-overview.sh
```

### 1.4 GSM8K

**Model**: ds-qwen-1.5b

```bash


bash benchmarks/gsm8k/method_script/adakv-overview.sh
```

---

## 2. Efficiency (Latency and GPU Memory)

Located at `methods/Adakv/experiments/efficiency/`; not included in the unified framework.

### 2.1 Efficiency Overview

```bash
cd methods/Adakv/experiments/efficiency

bash efficiency_overview.sh
```

### 2.2 Efficiency Budget

```bash
cd methods/Adakv/experiments/efficiency

bash efficiency_budget.sh
```

---

## 3. Memory (GPU Memory Measurement)

Located at `methods/Adakv/experiments/efficiency/`. It is adapted from the efficiency code and **measures GPU memory only** (timing code removed).

```bash
cd methods/Adakv/experiments/efficiency

bash memory.sh
```

---

## 4. Recall

Located at `methods/Adakv/experiments/recall/`; not included in the unified framework.

```bash
cd methods/Adakv/experiments/recall

bash run_recall.sh
```

---

## 5. Time (Latency)

Located at `methods/Adakv/experiments/time/`; not included in the unified framework.

```bash
cd methods/Adakv/experiments/time

bash run.sh
```
