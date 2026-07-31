# PQCache Experiment Guide

Method aliases: **full** = `original`, **topk(query per kv head)** = `no_drop_lb`, **topk32(query per kv head)** =`no_drop_lb_32`, **topp** = `no_drop_lb_topp`, and **pq** = `pq_search`.

## 1. Accuracy overview

Measures overall task accuracy across LongBench and RULER for full attention,
TopK, TopP, and PQCache.

```bash
cd methods/pqcache
bash run_longbench_overview.sh
```

```bash
cd methods/pqcache/RULER/scripts
bash run_ruler_overview.sh
```


## 2. Accuracy budget

Measures how task accuracy changes as the cache budget or TopP threshold varies.

```bash
cd methods/pqcache
bash run_longbench_budget.sh
```

```bash
cd methods/pqcache/RULER/scripts
bash run_ruler_budget.sh
```

## 3. Efficiency overview

Measures prefill and decoding latency across increasing input lengths.

```bash
cd methods/pqcache/efficiency_test
bash run_latency_overview_full.sh   # full attention with no repeatkv function 
bash run_latency_overview.sh        # PQCache
```

Logs and CSV files are written under `methods/pqcache/efficiency_test/log_latency/`.

## 4. Efficiency budget

Measures PQCache latency at several cache budgets and selected input lengths.

```bash
cd methods/pqcache/efficiency_test
bash run_latency_budget.sh
```

Results are written to `log_latency/`.

## 5. Memory

Measures peak GPU memory for short generation and for increasing input lengths.

```bash
cd methods/pqcache/memory
bash run_mem_pq.sh                  # short-input PQCache memory test
bash run_mem_4k_to_128k_all.sh      # full-attention and PQCache length sweep
```

Results are written to `methods/pqcache/memory/log_mem/`.

## 6. Recall

Measures whether TopK, TopK32, and PQCache retrieve the important KV entries.

```bash
cd methods/pqcache
bash run_longbench_recall.sh         # PQCache
bash run_longbench_recall_topk.sh    # TopK
bash run_longbench_recall_topk32.sh  # TopK32
```

```bash
cd methods/pqcache/RULER/scripts
bash recall_all.sh          # PQCache
bash recall_all_topk.sh     # TopK
bash recall_all_topk32.sh   # TopK32
```

Recall CSV files and analysis are written under `recall_list/`.

## 7. LongBenchV2

Evaluates long-context reasoning quality with full attention, TopK, TopP, and
PQCache.

```bash
cd methods/pqcache/LongBenchv2
bash run.sh              # full attention
bash cot_run_topk.sh
bash cot_run_topp.sh
bash cot_run_pq.sh
```

Predictions are written to `results/`; logs are written to `logs/`.

## 8. GSM8K

Evaluates mathematical reasoning accuracy with full attention, TopK, TopP, and
PQCache.

```bash
cd methods/pqcache/gsm8k
bash run_pred_original.sh
bash run_pred_topk.sh
bash run_pred_topp.sh
bash run_pred_pqcache.sh
```

All four GSM8K scripts evaluate automatically. To re-evaluate a prediction:

```bash
cd methods/pqcache/gsm8k
python evaluate.py --force --input pred/gsm8k/<model>/<compressor>/gsm8k.jsonl
```

Predictions and evaluation files are written below `pred/gsm8k/`.

## 9. Breakdown

Measures per-component runtime breakdown for full attention and PQCache.

```bash
cd methods/pqcache
bash breakdown_test/run_native_breakdown.sh   # full attention
bash breakdown_test/run_breakdown.sh          # PQCache
```

Breakdown logs and CSV files are written under `methods/pqcache/breakdown_test/log/`.
