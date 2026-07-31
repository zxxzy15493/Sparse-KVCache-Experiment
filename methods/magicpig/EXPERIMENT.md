# MagicPIG Experiment Guide

This directory contains the complete experiment code for the MagicPIG method. All experiment scripts are organized under the `experiments/` subdirectory, grouped by experiment type.

## Directory Structure

```
methods/magicpig/
├── experiments/            # Experiment scripts
│   ├── AccuracyOverview/    # Experiments 1-2: Accuracy evaluation
│   ├── EfficencyOverview/   # Experiments 3-4: Efficiency evaluation
│   ├── VRAMOverview/        # Experiment 5: Memory (VRAM) evaluation
│   ├── RECALLOverview/      # Experiment 6: Recall evaluation
│   ├── LongBench-v2/        # Experiment 7: LongBenchV2 evaluation
│   ├── GSM8K/               # Experiment 8: GSM8K evaluation
├── models_single/           # Single-GPU model implementations
├── model_recall/            # Recall-specific model implementations
├── library/                 # C++ extension libraries (sparse_attention, lsh)
└── install.sh               # Library installation script
```

---

## Experiment 1: Accuracy Overview

**Purpose**: Evaluate the end-to-end accuracy of MagicPIG on standard long-context benchmarks, compared against Full Attention.

**Directory**: `experiments/AccuracyOverview/`

**Datasets & Models**:
| Dataset | Models | Budget |
|---------|--------|--------|
| LongBench (all tasks) | Llama-3.1-8B, Qwen-2.5-7B, GLM-4-9B-1M | 1024 |
| RULER (4k/8k/16k/32k/64k) | Llama-3.1-8B, Qwen-2.5-7B-1M | 1024 |

**How to run**:

```bash
cd methods/magicpig/experiments/AccuracyOverview

# LongBench (single task example)
bash Accuracy.sh llama-3.1-8b LongBench narrativeqa

# RULER / Synthetic (single task example, 64k length)
bash Accuracy.sh llama-3.1-8b Synthetic niah_single_1 -1 65536

# Batch run: uncomment desired task lines in run.sh, then execute
bash run.sh
```

**Key scripts**:
- `Accuracy.sh` — Single-task prediction + evaluation entry point
- `pred.py` — Model inference
- `eval.py` — Metric computation
- `metrics.py` — Metric definitions
- `run.sh` — Contains all LongBench and RULER task commands (commented out by default)

**Output**: Results saved in `results/pred/<model>/<benchmark>/<K_L>/`, logs in `log/pred/...`

---

## Experiment 2: Accuracy Budget

**Purpose**: Evaluate how different KV cache budgets affect accuracy, validating MagicPIG's accuracy retention under varying compression rates.

**Directory**: `experiments/AccuracyOverview/` (same scripts as Experiment 1; switch budgets via `K` and `L` parameters)

**Datasets & Models**:
| Dataset | Models | Budgets |
|---------|--------|---------|
| LongBench | Llama-3.1-8B, Qwen-2.5-7B | 128, 256, 512, 1024 |
| RULER (64k) | Llama-3.1-8B, Qwen-2.5-7B-1M | 128, 384, 1024, 4096 |

**How to run**:

```bash
cd methods/magicpig/experiments/AccuracyOverview

# LongBench with different budgets (adjust K and L accordingly)
bash Accuracy.sh llama-3.1-8b LongBench narrativeqa -1 -1 <K> <L>

# RULER with different budgets (64k input length)
bash Accuracy.sh llama-3.1-8b Synthetic niah_single_3 -1 65536 <K> <L>
```

---

## Experiment 3: Efficiency Overview

**Purpose**: Evaluate MagicPIG's inference latency and throughput at different input lengths, demonstrating its speedup over Full Attention.

**Directory**: `experiments/EfficencyOverview/`

**Configuration**:
| Models | Input Lengths | Output Length | Budget |
|--------|---------------|---------------|--------|
| Llama-3.1-8B, Qwen2.5-7B-1M | 4k, 8k, 16k, 32k, 64k, 128k | 32 | 1024 |

**How to run**:

```bash
cd methods/magicpig/experiments/EfficencyOverview

# Single input length (e.g., 4k)
bash efficencyOverview.sh llama-3.1-8b-Instruct <K> <L> 4096 32 1024

# Batch run: uncomment desired lines in run.sh, then execute
bash run.sh
```

**Key scripts**:
- `efficencyOverview.sh` — Single-configuration efficiency evaluation
- `efficency.py` — Inference + timing
- `run.sh` — Batch commands covering all (model × length) combinations

**Output**: Results saved in `results/efficencyOverview/`

---

## Experiment 4: Efficiency Budget

**Purpose**: Evaluate how different KV cache budgets affect inference efficiency, analyzing the trade-off between budget and speedup.

**Directory**: `experiments/EfficencyOverview/`

**Configuration**:
| Models | Input Length | Budgets |
|--------|-------------|---------|
| Llama-3.1-8B, Qwen2.5-7B-1M | 4k | 128, 256, 512, 1024 |
| Llama-3.1-8B, Qwen2.5-7B-1M | 64k | 128, 384, 1024, 4096, 16384 |

**How to run**:

```bash
cd methods/magicpig/experiments/EfficencyOverview

# 4k input + various budgets
bash efficencyBudget.sh llama-3.1-8b-Instruct <K> <L> 4096 32 <except_budget>

# 64k input + various budgets
bash efficencyBudget.sh llama-3.1-8b-Instruct <K> <L> 65536 32 <except_budget>
```

**Key scripts**:
- `efficencyBudget.sh` — Same as `efficencyOverview.sh`, saves to `results/efficencyBudget/`

---

## Experiment 5: Memory (VRAM)

**Purpose**: Evaluate MagicPIG's peak GPU memory usage under different input/output lengths and budgets.

**Directory**: `experiments/VRAMOverview/`

**Configuration**:
| Models | Input Length | Output Length | Budgets |
|--------|-------------|---------------|---------|
| Llama-3.1-8B, Qwen2.5-7B-1M | 1k | 2, 4096 | 64, 512 |
| Llama-3.1-8B, Qwen2.5-7B-1M | 4k ~ 256k | 2 | 1024 |

**How to run**:

```bash
cd methods/magicpig/experiments/VRAMOverview

# 1k input + short output + budget 64
bash VRAMOverview.sh llama-3.1-8b-Instruct <K> <L> 1024 2 64

# 1k input + long output + budget 512
bash VRAMOverview.sh llama-3.1-8b-Instruct <K> <L> 1024 4096 512

# 64k input + short output + budget 512
bash VRAMOverview.sh llama-3.1-8b-Instruct <K> <L> 65536 2 512
```

**Key scripts**:
- `VRAMOverview.sh` — Memory evaluation entry point
- `pred.py` — Inference + VRAM recording
- `run.sh` — Batch commands for all (model × length × budget) combinations

**Output**: Results saved in `results/VRAMOverview/`

---

## Experiment 6: Recall

**Purpose**: Evaluate MagicPIG's KV cache retrieval recall rate — i.e., how well the LSH-based selection preserves the key-value pairs needed for exact attention computation.

**Directory**: `experiments/RECALLOverview/`

**Datasets & Models**:
| Dataset | Models | Budgets |
|---------|--------|---------|
| LongBench (narrativeqa, qasper) | Llama-3.1-8B, Qwen2.5-7B | 128, 256, 512, 1024 |
| RULER (64k, niah_single_3/vt/fwe) | Llama-3.1-8B, Qwen2.5-7B-1M | 128, 384, 1024, 4096 |

**How to run**:

```bash
cd methods/magicpig/experiments/RECALLOverview

# LongBench recall (e.g., narrativeqa)
bash RECALLOverview.sh llama-3.1-8b-Instruct <K> <L> 80 LongBench narrativeqa

# RULER recall (64k, e.g., niah_single_3)
bash RECALLOverview.sh llama-3.1-8b-Instruct <K> <L> 80 synthetic niah_single_3
```

**Key scripts**:
- `RECALLOverview.sh` — Recall evaluation entry point
- `pred.py` — Inference + recall statistics (uses `model_recall/` implementations)
- `evaluate.py` — Recall rate computation
- `run.sh` — Batch run commands

---

## Experiment 7: LongBenchV2

**Purpose**: Evaluate MagicPIG's reasoning capability on the more challenging LongBenchV2 benchmark.

**Directory**: `experiments/LongBench-v2/`

**Configuration**: Qwen2.5-7B-1M, budget 4096, Chain-of-Thought reasoning

**How to run**:

```bash
cd methods/magicpig/experiments/LongBench-v2

bash LongBenchV2.sh Qwen2.5-7B-Instruct-1M <K> <L> 4096
```

**Key scripts**:
- `LongBenchV2.sh` — Prediction entry point
- `pred.py` — Inference (with CoT prompting)
- `result.py` — Result processing
- `config.sh` — Parameter configuration
- `prompts/` — Prompt templates

---

## Experiment 8: GSM8K

**Purpose**: Evaluate MagicPIG's impact on mathematical reasoning using the GSM8K dataset.

**Directory**: `experiments/GSM8K/`

**Configuration**: DeepSeek-R1-Distill-Qwen-1.5B, budget 360, 8-shot CoT

**How to run**:

```bash
cd methods/magicpig/experiments/GSM8K

bash gsm8k_run.sh deepseek-r1-distill-qwen-1.5b <K> <L>
```

**Key scripts**:
- `gsm8k_run.sh` — Prediction + evaluation entry point
- `pred.py` — Inference with few-shot prompting
- `evaluate.py` — Evaluation logic
- `examples.py` — Few-shot examples
- `data/` — GSM8K dataset
- `tool/` — Data processing utilities

---

## Environment

For environment setup, model weight paths, and dependencies:

- **Library installation**: Run `bash install.sh` from the `methods/magicpig/` directory to compile and install the C++ extensions (`sparse_attention` and `lsh`).
- **Model paths**: Configured in each experiment's `config/model2path.json`.
- **Dataset paths**: Referenced relative to `../../../../benchmarks/` from each experiment directory.
- **ENVIRONMENT.md** in the project root — Conda environment and dependency instructions.
