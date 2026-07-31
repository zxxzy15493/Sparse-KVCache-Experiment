# MInference Experiments

This document describes how to reproduce the experiments for MInference. Each section covers one experiment: its purpose, the corresponding scripts, and how to run them.

## Directory Structure

```
methods/minference/
├── minference/              # Core library (accuracy/recall/memory experiments)
├── minference_time/         # Instrumented library (efficiency/timing experiments)
├── csrc/                    # CUDA kernels (build via build.sh)
└── experiments/
    ├── AccuracyOverview/    # Exp 1: Accuracy on LongBench & RULER (default budget)
    ├── EfficencyOverview/   # Exp 2: End-to-end latency across input lengths (default budget)
    ├── VramOverview/        # Exp 3: Peak GPU memory across input lengths (default budget)
    ├── RecallOverview/      # Exp 4: KV cache recall of sparse attention
    ├── LongBenchV2/         # Exp 5: LongBenchV2 evaluation
    ├── GSM8K/               # Exp 6: GSM8K math reasoning
    └── SelectTimeBreakDown/ # Exp 7: Prefill/decode time breakdown
```

Each experiment folder contains:
- A `run.sh` — top-level orchestrator that calls the per-task script with model/dataset combinations.
- A per-task shell script (e.g., `Accuracy.sh`) — sets up paths and launches `pred.py`.
- `pred.py` — the main Python driver that loads the model, applies the MInference patch, and runs inference.
- `config/` — JSON files mapping model short names to HuggingFace paths and max lengths.

**Note on budgets:** MInference does not include separate budget-sweep experiments. All overview experiments (accuracy, efficiency, memory) use the default sparsity budget configuration encoded in the pattern configuration JSON files under `minference/configs/` and `minference_time/configs/`. The scripts below assume the default budget config is already placed at the path resolved by `model2path.py`.

---

## Prerequisites

First, build the CUDA kernels and ensure the environment is set up:

```bash
cd methods/minference/csrc
bash build.sh
cd ../..
```

All experiments below are run from their respective directories under `methods/minference/experiments/`.

---

## 1. Accuracy Overview

**Purpose:** Evaluate end-to-end accuracy of MInference on standard long-context benchmarks using the default sparsity budget.

**Datasets & Models:**

| Benchmark | Models | Context Lengths |
|-----------|--------|-----------------|
| LongBench (12 tasks) | llama-3.1-8b, qwen-2.5-7b, glm-4-9b-chat-1m | native |
| RULER (synthetic) | llama-3.1-8b, qwen-2.5-7b-1m | 4k, 8k, 16k, 32k, 64k |

**Scripts:** `AccuracyOverview/`

**How to run:**

```bash
cd methods/minference/experiments/AccuracyOverview

# LongBench — edit run.sh to uncomment the desired model/task lines, then:
bash run.sh

# Or run individual tasks directly:
bash Accuracy.sh llama-3.1-8b LongBench narrativeqa
bash Accuracy.sh qwen-2.5-7b LongBench qasper
# ... (see run.sh for the full list of 12 LongBench tasks)

# RULER synthetic tasks (example for qwen-2.5-7b-1m):
bash Accuracy.sh qwen-2.5-7b-1m Synthetic niah_single_1
bash Accuracy.sh qwen-2.5-7b-1m Synthetic vt
```

Each task produces predictions in `results/pred/<model>/<benchmark>/` and logs in `log/pred/<model>/<benchmark>/`. After prediction, `eval.py` runs automatically and writes a `result.json` summary.

---

## 2. Efficiency Overview

**Purpose:** Measure end-to-end generation latency (TTFT and TPOT) across input lengths from 4k to 128k tokens using the default sparsity budget, with output length fixed at 32 tokens.

**Models:** llama-3.1-8b, qwen-2.5-7b-1m

**Scripts:** `EfficencyOverview/`

**How to run:**

```bash
cd methods/minference/experiments/EfficencyOverview

# Edit run.sh to uncomment the desired model/length lines, then:
bash run.sh

# Or run individual lengths directly:
bash efficencyOverview.sh llama-3.1-8b 6 4096 32
bash efficencyOverview.sh qwen-2.5-7b 6 131072 32
# Arguments: <model> <warmup_runs> <input_length> <output_length>
```

This uses the instrumented `minference_time` library. Results are written as JSONL to `results/EfficencyOverview/<model>/`. Each run records TTFT, TPOT (ms), and per-step decode latency.

---

## 3. Peak Memory

**Purpose:** Measure peak GPU memory usage during generation. Two sub-experiments:
- **Varying output length:** input fixed at 1k tokens, budgets 64 and 512, output lengths 2 and 4k.
- **Varying input length:** input 4k–256k, output fixed at 2 tokens, using the default budget.

**Models:** llama-3.1-8b, qwen-2.5-7b-1m

**Scripts:** `VramOverview/`

**How to run:**

```bash
cd methods/minference/experiments/VramOverview

# Edit run.sh to uncomment the desired model/length lines, then:
bash run.sh

# Or run individual configurations directly:
bash efficencyOverview.sh llama-3.1-8b 2 4096 2
bash efficencyOverview.sh qwen-2.5-7b 2 131072 2
# Arguments: <model> <warmup_runs> <input_length> <output_length>
```

Results record `peak_memory_MB` in `results/VramOverview/<model>/VramOverview_<input>_<output>.jsonl`.

---

## 4. KV Cache Recall

**Purpose:** Measure the recall of the sparse attention pattern — i.e., what fraction of the top-k dense attention scores are retained by MInference's vertical-and-slash selection.

**Datasets & Models:**

| Benchmark | Models | Budgets |
|-----------|--------|---------|
| LongBench | llama-3.1-8b, qwen-2.5-7b | tokens: 128, 256, 512, 1024 / ratios: 0.8, 0.85, 0.9, 0.95 |
| RULER (64k) | llama-3.1-8b, qwen-2.5-7b-1m | tokens: 128, 384, 1024, 4096 / ratios: 0.8, 0.85, 0.9, 0.95 |

**Scripts:** `RecallOverview/`

**How to run:**

```bash
cd methods/minference/experiments/RecallOverview

# Edit run.sh to uncomment model/task lines, then:
bash run.sh

# Or run individually:
bash RecallOverview.sh llama-3.1-8b LongBench narrativeqa
bash RecallOverview.sh qwen-2.5-7b-1m synthetic niah_single_3
```

After prediction, evaluate recall statistics:

```bash
python evaluate.py --results_dir ./results --granularity all
```

This computes recall at multiple granularities: overall, per sample, per layer, and per head.

---

## 5. LongBenchV2

**Purpose:** Evaluate MInference on the challenging LongBenchV2 benchmark (64k–192k multi-choice questions) with chain-of-thought reasoning. Budget: 4096 tokens.

**Model:** qwen-2.5-7b-1m

**Scripts:** `LongBenchV2/`

**How to run:**

```bash
cd methods/minference/experiments/LongBenchV2

# Default: CoT enabled
bash run.sh qwen-2.5-7b-1M

# Without chain-of-thought:
bash run.sh qwen-2.5-7b-1M --no-cot

# Limit to N samples:
bash run.sh qwen-2.5-7b-1M --num-samples 100
```

Predictions are saved to `results/<model>/pred_cot.jsonl`. After prediction, `result.py` runs automatically and breaks down accuracy by difficulty (easy/hard) and length (short/medium/long).

---

## 6. GSM8K

**Purpose:** Evaluate MInference on the GSM8K math reasoning benchmark (8-shot chain-of-thought). Budget: 360 tokens.

**Model:** deepseek-r1-distill-qwen-1.5b

**Scripts:** `GSM8K/`

**How to run:**

```bash
cd methods/minference/experiments/GSM8K

# Default: 8-shot CoT
bash gsm8k_run.sh deepseek-r1-distill-qwen-1.5b

# Customize shots and CoT type:
bash gsm8k_run.sh <model> <num_shots> <cot_type>
```

Results are saved to `results/pred/<model>/<cot_type>/gsm8k.jsonl`. Evaluation runs automatically after prediction.

---

## 7. Select-Time Breakdown

**Purpose:** Fine-grained CUDA-level timing breakdown of the MInference prefill and decode phases. Separately measures pattern allocation time, attention computation, KV-cache writes, and FFN time. Compares MInference sparse mode against a full attention baseline.

**Datasets & Models:**

| Dataset | Model | Input Lengths |
|---------|-------|---------------|
| Synthetic | llama-3.1-8b | 4k,64k |

**Scripts:** `SelectTimeBreakDown/`

**How to run:**

```bash
cd methods/minference/experiments/SelectTimeBreakDown

# MInference sparse mode (default):
bash run.sh

# Or run with explicit arguments:
bash SelectTimeBreakDown.sh llama-3.1-8b 5 4096 32 0
# Arguments: <model> <warmup_runs> <input_length> <output_length> <full_attn_flag>
#   full_attn_flag: 0 = sparse MInference, 1 = full attention baseline

# Full attention baseline:
bash SelectTimeBreakDown.sh llama-3.1-8b 5 4096 32 1
```

Results are saved under `results/SelectTimeBreakDown/`. Each run records prefill time, per-step decode time, and the breakdown of attention kernel time vs. FFN time.

---
