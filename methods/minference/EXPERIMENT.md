# MInference Experiments

This document describes how to reproduce the experiments for MInference (SIGMOD submission). Each section covers one experiment: its purpose, the corresponding scripts, and how to run them.

## Directory Structure

```
methods/minference/
├── minference/              # Core library (accuracy/recall/memory experiments)
├── minference_time/         # Instrumented library (efficiency/timing experiments)
├── csrc/                    # CUDA kernels (build via build.sh)
└── experiments/
    ├── AccuracyOverview/    # Exp 1: Accuracy on LongBench & RULER (fixed budget)
    ├── EfficencyOverview/   # Exp 3: End-to-end latency across input lengths
    ├── VramOverview/        # Exp 5: Peak GPU memory across input lengths
    ├── RecallOverview/      # Exp 6: KV cache recall of sparse attention
    ├── LongBenchV2/         # Exp 7: LongBenchV2 evaluation
    ├── GSM8K/               # Exp 8: GSM8K math reasoning
    └── SelectTimeBreakDown/ # Supplementary: Prefill/decode time breakdown
```

Each experiment folder contains:
- A `run.sh` — top-level orchestrator that calls the per-task script with model/dataset combinations.
- A per-task shell script (e.g., `Accuracy.sh`) — sets up paths and launches `pred.py`.
- `pred.py` — the main Python driver that loads the model, applies the MInference patch, and runs inference.
- `config/` — JSON files mapping model short names to HuggingFace paths and max lengths.

**Note on budgets:** The sparsity budget (e.g., 1024 tokens, ratio 0.9) is encoded in the pattern configuration JSON files under `minference/configs/` and `minference_time/configs/`. Different budget settings use different config files. The scripts below assume the desired budget config is already placed at the path resolved by `model2path.py`.

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

## 1. Accuracy Overview (accuracy_overview)

**Purpose:** Evaluate end-to-end accuracy of MInference on standard long-context benchmarks at a fixed budget (1024 tokens / ratio 0.9).

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
bash Accuracy.sh llama3.1-8b LongBench narrativeqa
bash Accuracy.sh qwen2.5-7b LongBench qasper
# ... (see run.sh for the full list of 12 LongBench tasks)

# RULER synthetic tasks (example for qwen2.5-7b-1m):
bash Accuracy.sh qwen2.5-7b-1m Synthetic niah_single_1
bash Accuracy.sh qwen2.5-7b-1m Synthetic vt
```

Each task produces predictions in `results/pred/<model>/<benchmark>/` and logs in `log/pred/<model>/<benchmark>/`. After prediction, `eval.py` runs automatically and writes a `result.json` summary.

---

## 2. Accuracy vs. Budget (accuracy_budget)

**Purpose:** Sweep over different sparsity budgets to measure the accuracy–efficiency trade-off. Vary the budget token count and ratio on LongBench and RULER (64k).

**Datasets & Models:**

| Benchmark | Models | Budgets |
|-----------|--------|---------|
| LongBench | llama-3.1-8b, qwen-2.5-7b | tokens: 128, 256, 512, 1024 / ratios: 0.8, 0.85, 0.9, 0.95 |
| RULER (64k) | llama-3.1-8b, qwen-2.5-7b-1m | tokens: 128, 384, 1024, 4096 / ratios: 0.8, 0.85, 0.9, 0.95 |

**Scripts:** Same as Accuracy Overview — reuse `AccuracyOverview/Accuracy.sh` after switching the budget config file.

**How to run:**

```bash
cd methods/minference/experiments/AccuracyOverview

# For each budget setting, update the config path in minference/configs/model2path.py
# to point to the corresponding budget-specific JSON, then run:
bash Accuracy.sh <model> LongBench <task>
bash Accuracy.sh <model> Synthetic <task>
```

---

## 3. Efficiency Overview (efficiency_overview)

**Purpose:** Measure end-to-end generation latency (TTFT and TPOT) across input lengths from 4k to 128k tokens at a fixed budget (1024/0.9), with output length fixed at 32 tokens.

**Models:** llama-3.1-8b, qwen-2.5-7b-1m

**Scripts:** `EfficencyOverview/`

**How to run:**

```bash
cd methods/minference/experiments/EfficencyOverview

# Edit run.sh to uncomment the desired model/length lines, then:
bash run.sh

# Or run individual lengths directly:
bash efficencyOverview.sh llama3.1-8b-instruct 6 4096 32
bash efficencyOverview.sh qwen2.5-7b-instruct 6 131072 32
# Arguments: <model> <warmup_runs> <input_length> <output_length>
```

This uses the instrumented `minference_time` library. Results are written as JSONL to `results/EfficencyOverview/<model>/`. Each run records TTFT, TPOT (ms), and per-step decode latency.

---

## 4. Efficiency vs. Budget (efficiency_budget)

**Purpose:** Sweep over sparsity budgets to measure end-to-end latency at two representative input lengths (4k and 64k).

**Models:** llama-3.1-8b, qwen-2.5-7b-1m

**Budgets:**
- 4k input: tokens 128, 256, 512, 1024 / ratios 0.8, 0.85, 0.9, 0.95
- 64k input: tokens 128, 384, 1024, 4096 / ratios 0.8, 0.85, 0.9, 0.95

**Scripts:** Same as Efficiency Overview — reuse `EfficencyOverview/efficencyOverview.sh` after switching the budget config.

**How to run:**

```bash
cd methods/minference/experiments/EfficencyOverview

# For each budget setting, update the config, then:
bash efficencyOverview.sh <model> <warmup> <input_length> 32
```

---

## 5. Peak Memory (memory)

**Purpose:** Measure peak GPU memory usage during generation. Two sub-experiments:
- **Varying output length:** input fixed at 1k tokens, budgets 64 and 512, output lengths 2 and 4k.
- **Varying input length:** input 4k–128k, output fixed at 2 tokens, budget 1024/0.9.

**Models:** llama-3.1-8b, qwen-2.5-7b-1m

**Scripts:** `VramOverview/`

**How to run:**

```bash
cd methods/minference/experiments/VramOverview

# Edit run.sh to uncomment the desired model/length lines, then:
bash run.sh

# Or run individual configurations directly:
bash efficencyOverview.sh llama3.1-8b-instruct 2 4096 2
bash efficencyOverview.sh qwen2.5-7b-instruct 2 131072 2
# Arguments: <model> <warmup_runs> <input_length> <output_length>
```

Results record `peak_memory_MB` in `results/VramOverview/<model>/VramOverview_<input>_<output>.jsonl`.

---

## 6. KV Cache Recall (recall)

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
bash RecallOverview.sh llama3.1-8b LongBench narrativeqa
bash RecallOverview.sh qwen2.5-7b-1m synthetic niah_single_3
```

After prediction, evaluate recall statistics:

```bash
python evaluate.py --results_dir ./results --granularity all
```

This computes recall at multiple granularities: overall, per sample, per layer, and per head.

---

## 7. LongBenchV2

**Purpose:** Evaluate MInference on the challenging LongBenchV2 benchmark (64k–192k multi-choice questions) with chain-of-thought reasoning. Budget: 4096 tokens.

**Model:** qwen-2.5-7b-1m

**Scripts:** `LongBenchV2/`

**How to run:**

```bash
cd methods/minference/experiments/LongBenchV2

# Default: CoT enabled
bash run.sh Qwen2.5-7B-Instruct-1M

# Without chain-of-thought:
bash run.sh Qwen2.5-7B-Instruct-1M --no-cot

# Limit to N samples:
bash run.sh Qwen2.5-7B-Instruct-1M --num-samples 100
```

Predictions are saved to `results/<model>/pred_cot.jsonl`. After prediction, `result.py` runs automatically and breaks down accuracy by difficulty (easy/hard) and length (short/medium/long).

---

## 8. GSM8K

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

## Supplementary: Select-Time Breakdown

**Purpose:** Fine-grained CUDA-level timing breakdown of the MInference prefill and decode phases. Separately measures pattern allocation time, attention computation, KV-cache writes, and FFN time.

**Scripts:** `SelectTimeBreakDown/`

**How to run:**

```bash
cd methods/minference/experiments/SelectTimeBreakDown

# MInference sparse mode:
bash run.sh

# Full attention baseline (adds --full flag):
bash SelectTimeBreakDown.sh llama3.1-8b-instruct 5 4096 32 1
# Last argument: 1 = full attention, 0 = sparse MInference
```

Results are saved under `results/SelectTimeBreakDown/`.

---

## Notes

- **GPU selection:** Set `CUDA_VISIBLE_DEVICES` in `run.sh` or before running.
- **Resume support:** Most `pred.py` scripts skip already-predicted samples (based on output JSONL), so interrupted runs can be safely resumed.
- **Config files:** Model-to-path and model-to-maxlen mappings live in each experiment's `config/` directory. Update `model2path.json` if your model weights are stored at custom paths.
- **Budget switching:** The sparsity budget is determined by the pattern JSON referenced in `minference/configs/model2path.py` (or `minference_time/configs/model2path.py` for efficiency experiments). To change budgets, point to a different pattern JSON file.
