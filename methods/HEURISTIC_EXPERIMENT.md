# HERU Experiment Guide

This document summarizes the experiment entry points for `methods/x-attention`, `methods/quest`, `methods/sparq`, and `methods/flexPrefill`, so readers can locate the relevant folders and scripts by experiment type.

All commands are intended to be run from the repository root. Each command block returns to the repository root with a relative `cd`; if you run only part of a block, make sure your current working directory is correct.

## Method Directories

| Method | Code Directory | Core Implementation | Main Experiment Script Directories |
| --- | --- | --- | --- |
| Quest | `methods/quest` | `quest/`, `evaluation/` | `method_scripts/`, `breakdown/`, `longbenchV2/`, `gsm8k/` |
| SparQ | `methods/sparq` | `llminference/` | `method_scripts/`, `breakdown/`, `longbenchV2/`, `gsm8k/` |
| FlexPrefill | `methods/flexPrefill` | `flex_prefill/` | `method_scripts/`, `breakdown/`, `longbenchV2/`, `gsm8k/` |
| XAttention | `methods/x-attention` | `xattn/`, `threshold/` | `method_scripts/`, `breakdown/`, `longbenchV2/`, `gsm8k/` |

Budget notation: `1024/0.9` means that Quest and SparQ use token budget `1024`, FlexPrefill uses `fixthreshold=0.9` mapped to gamma, and XAttention uses `fixthreshold=0.9` mapped to its threshold. Similarly, `128,256,512,1024/0.8,0.85,0.9,0.95` means the first two methods use token budgets, while the latter two methods use `fixthreshold`.

## 1. accuracy_overview

This experiment measures overall accuracy. LongBench uses `llama-3.1-8b`, `qwen-2.5-7b`, and `glm-4-9b-1m` with budget `1024/0.9`. RULER uses context lengths 4k, 8k, 16k, 32k, and 64k, models `llama-3.1-8b` and `qwen-2.5-7b-1m`, and budget `1024/0.9`.

Unified entry points are under `benchmarks/longbench/method_script` and `benchmarks/ruler/method_scripts`:

```bash
cd benchmarks/longbench/method_script
bash quest-overview.sh
bash sparq-overview.sh
bash flexprefill-overview.sh
bash xattention-overview.sh
cd ../../..

cd benchmarks/ruler/method_scripts
bash quest-overview.sh
bash sparq-overview.sh
bash flexprefill-overview.sh
bash xattention-overview.sh
cd ../../..
```

Additional LongBench/RULER scripts are also kept inside each method directory, mainly under `method_scripts/ACC/`:

```bash
cd methods/quest
bash method_scripts/ACC/llama/longbench.sh
bash method_scripts/ACC/llama/ruler.sh
bash method_scripts/ACC/qwen/longbench.sh
bash method_scripts/ACC/qwen/ruler.sh
bash method_scripts/ACC/glm/longbench.sh
cd ../sparq
bash method_scripts/ACC/llama/longbench.sh
bash method_scripts/ACC/llama/ruler.sh
bash method_scripts/ACC/qwen/longbench.sh
bash method_scripts/ACC/qwen/ruler.sh
bash method_scripts/ACC/glm/longbench.sh
cd ../flexPrefill
bash method_scripts/ACC/llama/longbench.sh
bash method_scripts/ACC/llama/ruler.sh
bash method_scripts/ACC/qwen/longbench.sh
bash method_scripts/ACC/qwen/ruler.sh
bash method_scripts/ACC/glm/longbench.sh
cd ../x-attention
bash method_scripts/ACC/llama/longbench.sh
bash method_scripts/ACC/llama/ruler.sh
bash method_scripts/ACC/qwen/longbench.sh
bash method_scripts/ACC/qwen/ruler.sh
bash method_scripts/ACC/glm/longbench.sh
cd ../..
```

## 2. accuracy_budget

This experiment measures accuracy under different budgets. LongBench uses `llama-3.1-8b` and `qwen-2.5-7b` with budgets `128,256,512,1024/0.8,0.85,0.9,0.95`. RULER uses 64k context length, models `llama-3.1-8b` and `qwen-2.5-7b-1m`, and budgets `128,384,1024,4096/0.8,0.85,0.9,0.95`.

Unified entry points:

```bash
cd benchmarks/longbench/method_script
bash quest-budget.sh
bash sparq-budget.sh
bash flexprefill-budget.sh
bash xattention-budget.sh
cd ../../..

cd benchmarks/ruler/method_scripts
bash quest-budget.sh
bash sparq-budget.sh
bash flexprefill-budget.sh
bash xattention-budget.sh
cd ../../..
```

Method-specific scripts are under `method_scripts/ACC_budget/`:

```bash
cd methods/quest
bash method_scripts/ACC_budget/llama/longbench.sh
bash method_scripts/ACC_budget/llama/ruler.sh
bash method_scripts/ACC_budget/qwen/longbench.sh
bash method_scripts/ACC_budget/qwen/ruler.sh
cd ../sparq
bash method_scripts/ACC_budget/llama/longbench.sh
bash method_scripts/ACC_budget/llama/ruler.sh
bash method_scripts/ACC_budget/qwen/longbench.sh
bash method_scripts/ACC_budget/qwen/ruler.sh
cd ../flexPrefill
bash method_scripts/ACC_budget/llama/longbench.sh
bash method_scripts/ACC_budget/llama/ruler.sh
bash method_scripts/ACC_budget/qwen/longbench.sh
bash method_scripts/ACC_budget/qwen/ruler.sh
cd ../x-attention
bash method_scripts/ACC_budget/llama/longbench.sh
bash method_scripts/ACC_budget/llama/ruler.sh
bash method_scripts/ACC_budget/qwen/longbench.sh
bash method_scripts/ACC_budget/qwen/ruler.sh
cd ../..
```

## 3. efficiency_overview

This experiment measures end-to-end generation efficiency across input lengths. It uses `llama-3.1-8b` and `qwen-2.5-7b-1m`, input lengths from 4k to 128k, output length 32, and budget `1024/0.9`.

All scripts are under each method's `method_scripts/efficiency/` directory:

```bash
cd methods/quest
bash method_scripts/efficiency/latency.sh
bash method_scripts/efficiency/qwen_latency.sh
cd ../sparq
bash method_scripts/efficiency/latency.sh
bash method_scripts/efficiency/qwen_latency.sh
cd ../flexPrefill
bash method_scripts/efficiency/latency.sh
bash method_scripts/efficiency/qwen_latency.sh
cd ../x-attention
bash method_scripts/efficiency/latency.sh
bash method_scripts/efficiency/qwen_latency.sh
cd ../..
```

## 4. efficiency_budget

This experiment measures efficiency under different budgets. It uses `llama-3.1-8b` and `qwen-2.5-7b-1m`, with output length 32. At 4k input length, it sweeps `128,256,512,1024/0.8,0.85,0.9,0.95`; at 64k input length, it sweeps `128,384,1024,4096,16384/0.8,0.85,0.9,0.95`.

All scripts are under each method's `method_scripts/efficiency/` directory:

```bash
cd methods/quest
bash method_scripts/efficiency/budget_latency.sh
bash method_scripts/efficiency/budget_qwen_latency.sh
cd ../sparq
bash method_scripts/efficiency/budget_latency.sh
bash method_scripts/efficiency/budget_qwen_latency.sh
cd ../flexPrefill
bash method_scripts/efficiency/budget_latency.sh
bash method_scripts/efficiency/budget_qwen_latency.sh
cd ../x-attention
bash method_scripts/efficiency/budget_latency.sh
bash method_scripts/efficiency/budget_qwen_latency.sh
cd ../..
```

## 5. memory

This experiment measures peak GPU memory. It uses `llama-3.1-8b` and `qwen-2.5-7b-1m`. One setting tests 1k input length with budgets 64 and 512, output lengths 2 and 4k. Another setting tests input lengths from 4k to 256k with output length 2 and budget `1024/0.9`.

All scripts are under each method's `method_scripts/VRAM/` directory:

```bash
cd methods/quest
bash method_scripts/VRAM/llamaVRAM.sh
bash method_scripts/VRAM/qwenVRAM.sh
cd ../sparq
bash method_scripts/VRAM/llamaVRAM.sh
bash method_scripts/VRAM/qwenVRAM.sh
cd ../flexPrefill
bash method_scripts/VRAM/llamaVRAM.sh
bash method_scripts/VRAM/qwenVRAM.sh
cd ../x-attention
bash method_scripts/VRAM/llamaVRAM.sh
bash method_scripts/VRAM/qwenVRAM.sh
cd ../..
```

## 6. recall

This experiment measures the overlap between sparse selection results and important tokens under full attention. LongBench uses `llama-3.1-8b` and `qwen-2.5-7b` with budgets `128,256,512,1024/0.8,0.85,0.9,0.95`. RULER uses 64k context length, models `llama-3.1-8b` and `qwen-2.5-7b-1m`, and budgets `128,384,1024,4096/0.8,0.85,0.9,0.95`.

Quest, SparQ, and XAttention split recall scripts by model under `method_scripts/recall/llama/` and `method_scripts/recall/qwen/`:

```bash
cd methods/quest
bash method_scripts/recall/llama/longbench.sh
bash method_scripts/recall/qwen/longbench.sh
bash method_scripts/recall/llama/ruler.sh
bash method_scripts/recall/qwen/ruler.sh
cd ../sparq
bash method_scripts/recall/llama/longbench.sh
bash method_scripts/recall/qwen/longbench.sh
bash method_scripts/recall/llama/ruler.sh
bash method_scripts/recall/qwen/ruler.sh
cd ../x-attention
bash method_scripts/recall/llama/longbench.sh
bash method_scripts/recall/qwen/longbench.sh
bash method_scripts/recall/llama/ruler.sh
bash method_scripts/recall/qwen/ruler.sh
cd ../..
```

FlexPrefill recall scripts do not use a separate model-level directory:

```bash
cd methods/flexPrefill
bash method_scripts/recall/longbench.sh
bash method_scripts/recall/ruler.sh
cd ../..
```

## 7. longbenchv2

This experiment measures long-context question answering on LongBenchV2. It uses `qwen-2.5-7b-1m`; Quest and SparQ use token budget `4096`, while FlexPrefill and XAttention use `fixthreshold=0.9`.

Unified entry points are under `benchmarks/longbenchv2/method_script/`:

```bash
cd benchmarks/longbenchv2/method_script
bash quest-overview.sh
bash sparq-overview.sh
bash flexprefill-overview.sh
bash xattention-overview.sh
cd ../../..
```

Independent method-level entry points are also available:

```bash
cd methods/quest
bash longbenchV2/run.sh
cd ../sparq
bash longbenchV2/run.sh
cd ../flexPrefill
bash longbenchV2/run.sh
cd ../x-attention
bash longbenchV2/run.sh
cd ../..
```

## 8. gsm8k

This experiment measures GSM8K mathematical reasoning accuracy. It uses `ds-qwen-1.5b`; Quest and SparQ use token budget `360`, while FlexPrefill and XAttention use `fixthreshold=0.8`.

Unified entry points are under `benchmarks/gsm8k/method_script/`:

```bash
cd benchmarks/gsm8k/method_script
bash quest-overview.sh
bash sparq-overview.sh
bash flexprefill-overview.sh
bash xattention-overview.sh
cd ../../..
```

Independent method-level entry points are also available:

```bash
cd methods/quest
bash gsm8k/gsm.sh
cd ../sparq
bash gsm8k/gsm.sh
cd ../flexPrefill
bash gsm8k/gsm.sh
cd ../x-attention
bash gsm8k/gsm.sh
cd ../..
```

## 9. breakdown

This experiment records component-level inference time, such as prefill, decode, retrieval, and sparse attention computation. The setting mainly uses `llama-3.1-8b`. The paper setting includes `1024/0.9` for 4k input and `1024/0.9` for 64k input.

Current script entry points are under each method's `breakdown/` directory:

```bash
cd methods/quest
bash breakdown/breakdown.sh
cd ../sparq
bash breakdown/breakdown.sh
cd ../flexPrefill
bash breakdown/breakdown.sh
cd ../x-attention
bash breakdown/breakdown.sh
cd ../..
```

## Output Locations

Output directories differ across scripts. Common locations are:

| Experiment | Common Output Directory |
| --- | --- |
| LongBench | Unified entry points: `benchmarks/longbench/outputs/`; method-level entry points: `methods/quest/evaluation/LongBench/pred/`, `methods/sparq/experiments/longbench/pred/`, `methods/flexPrefill/experiments/benchmark/longbench/pred/`, `methods/x-attention/eval/LongBench/pred/` or `methods/x-attention/eval/LongBench/budget_pred/` |
| RULER | Unified entry points: `benchmarks/ruler/benchmark_root_pred/`; method-level entry points: Quest/SparQ/XAttention default to their own `output/ruler/`, while FlexPrefill defaults to `methods/flexPrefill/outputs/ruler/` or `methods/flexPrefill/outputs_budget/` |
| efficiency | Each method's `efficiency/latency-results/` and `efficiency/budget/latency-results/` |
| memory | Quest/SparQ/FlexPrefill/XAttention write under `methods/` |
| recall | Quest: `efficiency/recall_topkrate/`; SparQ: `efficiency/recall_attnscores/`; FlexPrefill: `efficiency/attn_rate-results/`; XAttention: `efficiency/attn_score/` |
| LongBenchV2 | Unified entry points: `benchmarks/longbenchv2/outputs/`; method-level entry points: Quest/FlexPrefill/XAttention/SparQ write to their own `longbenchV2/res/` |
| GSM8K | Unified entry points: `benchmarks/gsm8k/outputs/`; method-level entry points: each method's `gsm8k/res/` |
| breakdown | SparQ/FlexPrefill/XAttention/Quest: each method's `breakdown_results/` |

## Running Notes

1. Unified benchmark scripts usually define models, datasets, budgets, or thresholds at the top of each script. If you only need the settings listed in this document, adjust those top-level arrays first.
2. The method-level `method_scripts/` directories provide supplementary entry points for per-method reproduction and ad hoc debugging.
3. These experiments require CUDA GPUs and depend on the repository's model path configuration, dataset path configuration, and HuggingFace access.
4. Long-context experiments require substantial GPU memory. It is recommended to validate the environment with a single model and a shorter context length before running the full sweep.
