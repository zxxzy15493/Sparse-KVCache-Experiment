# ClusterKV Experiment Guide

Model locations for LongBench are configured in `accuracy/config/model2path.json`.
The RULER model names and locations are configured in `RULER/scripts/config_models.sh`.
GSM8K uses the shared dataset at `benchmarks/gsm8k/data/gsm8k_test.jsonl`.

## 1. Accuracy overview

Measures overall long-context task accuracy at the standard experimental
setting. LongBench evaluates Llama-3.1-8B, Qwen-2.5-7B, and GLM-4-9B-1M;
RULER evaluates Llama-3.1-8B and Qwen-2.5-7B-1M across multiple context
lengths.

LongBench files: `run_longbench_overview.sh`,
`accuracy/LongBench/mypred.py`, and `accuracy/LongBench/eval.py`.

```bash
cd methods/ClusterKV
bash run_longbench_overview.sh
```

RULER files: `RULER/scripts/run_ruler_overview.sh`,
`RULER/scripts/run.sh`, and `RULER/scripts/config_models.sh`.

```bash
cd methods/ClusterKV/RULER/scripts
bash run_ruler_overview.sh
```

LongBench predictions are written below `accuracy/LongBench/pred/`. RULER
data, predictions, and metrics are written below `benchmarks/ruler/benchmark_root/`.

## 2. Accuracy budget

Measures how LongBench and RULER task accuracy changes with the available
cache budget. The LongBench run uses representative question answering,
classification, and code tasks; the RULER run uses its long-context benchmark
setting.

LongBench files: `run_longbench_budget.sh`,
`accuracy/LongBench/mypred.py`, and `accuracy/LongBench/eval.py`.

```bash
cd methods/ClusterKV
bash run_longbench_budget.sh
```

RULER files: `RULER/scripts/run_ruler_budget.sh` and
`RULER/scripts/run.sh`.

```bash
cd methods/ClusterKV/RULER/scripts
bash run_ruler_budget.sh
```

Results are written to the same LongBench prediction and RULER benchmark
directories as the accuracy overview experiment.

## 3. Efficiency overview

Measures prefill and decoding latency as input length increases for
Llama-3.1-8B and Qwen-2.5-7B-1M.

Files: `efficiency/run_my_latency_overview.sh`and
`efficiency/my_textgen.py`.

```bash
cd methods/ClusterKV/efficiency
bash run_my_latency_overview.sh
```

Logs and latency CSV files are written to `efficiency/log_latency/`.

## 4. Efficiency budget

Measures latency sensitivity to cache budget at selected short and long input
lengths for Llama-3.1-8B and Qwen-2.5-7B-1M.

Files: `efficiency/run_my_latency_budget.sh` and
`efficiency/my_textgen.py`.

```bash
cd methods/ClusterKV/efficiency
bash run_my_latency_budget.sh
```

Logs and CSV files are written to `efficiency/log_latency/`.

## 5. Memory

Measures peak GPU memory for short generation with long outputs and for a
length sweep with short outputs. Both Llama-3.1-8B and Qwen-2.5-7B-1M are
covered.

Files: `memory/run_my_mem.sh`, `memory/mem_list.sh`, and
`memory/my_mem_test_once.py`.

```bash
cd methods/ClusterKV/memory
bash run_my_mem.sh
bash mem_list.sh
```

Logs and memory CSV files are written to `memory/log_mem/`.

## 6. Recall

Measures whether ClusterKV retrieves the important KV entries selected during
LongBench and RULER inference. It complements accuracy measurements with a
direct analysis of retrieval behavior.

LongBench files: `accuracy/LongBench/recall_all.sh`,
`accuracy/LongBench/recall_pred.py`, and
`accuracy/LongBench/analyze_recall.py`.

```bash
cd methods/ClusterKV/accuracy/LongBench
bash recall_all.sh
```

RULER files: `RULER/scripts/recall_all.sh`, `RULER/scripts/recall.sh`, and
`RULER/scripts/analyze_recall.py`.

```bash
cd methods/ClusterKV/RULER/scripts
bash recall_all.sh
```

The scripts place recall records and analysis output in their configured
recall result directories.

## 7. LongBenchV2

Evaluates long-context reasoning quality on LongBenchV2 with the configured
Qwen-2.5-7B-1M model and ClusterKV setting.

Files: `LongBenchV2/run.sh`, `LongBenchV2/mypred.py`, and
`LongBenchV2/result.py`.

```bash
cd methods/ClusterKV/LongBenchV2
bash run.sh
```

Predictions are written to `LongBenchV2/results/`; runtime logs are written to
`LongBenchV2/logs/`.

## 8. GSM8K

Evaluates mathematical reasoning accuracy using the configured DeepSeek-Qwen
model with ClusterKV. The prediction script loads the shared GSM8K test file,
and the run script evaluates the generated answers automatically.

Files: `gsm8k/run.sh`, `gsm8k/pred.py`, and `gsm8k/evaluate.py`.

```bash
cd methods/ClusterKV/gsm8k
bash run.sh
```

Predictions, evaluation files, and logs are written below `gsm8k/results/`
and `gsm8k/log/`.

## 9. Breakdown

Measures per-component runtime breakdown for ClusterKV.

Files: `breakdown_test/run_myllama_breakdown.sh` and
`breakdown_test/breakdown_myllama.py`.

```bash
cd methods/ClusterKV
bash breakdown_test/run_myllama_breakdown.sh
```

Breakdown logs and CSV files are written below `breakdown_test/log/`.
