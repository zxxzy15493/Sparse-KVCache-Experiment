# RetroInfer Experiment Guide

This directory contains the complete experiment code for the RetroInfer method. All experiment scripts are organized under the `benchmark/` subdirectory, grouped by experiment type.

## Directory Structure

```
methods/retroinfer/
├── attn_hub/          # Attention mechanism implementations (flash_attn, minfer, retroinfer)
├── cache_hub/         # KV cache implementations (retroinfer cache, kmeans clustering, etc.)
├── model_hub/         # Model adaptation layers (llama, qwen, glm, deepseek)
├── config/            # Model configuration files (JSON)
├── config.py          # RetroInfer configuration generation logic
├── library/           # C++ extension library
└── benchmark/         # Experiment scripts
    ├── AccuracyOverview/    # Experiments 1-2: Accuracy evaluation
    ├── EfficencyOverview/   # Experiments 3-4: Efficiency evaluation
    ├── VRAMOverview/        # Experiment 5: Memory (VRAM) evaluation
    ├── RECALLOverview/      # Experiment 6: Recall evaluation
    ├── LongBenchV2/         # Experiment 7: LongBenchV2 evaluation
    ├── GSM8k/               # Experiment 8: GSM8K evaluation
    └── SelectTimeBreakdown/ # Auxiliary: Time breakdown analysis
```

---

## Experiment 1: Accuracy Overview

**Purpose**: Evaluate the end-to-end accuracy of RetroInfer on standard long-context benchmarks, compared against Full Attention.

**Directory**: `benchmark/AccuracyOverview/`

**Datasets & Models**:
| Dataset | Models | Budget |
|---------|--------|--------|
| LongBench (all tasks) | Llama-3.1-8B, Qwen2.5-7B, GLM-4-9B-1M | 1024 |
| RULER (4k/8k/16k/32k/64k) | Llama-3.1-8B, Qwen2.5-7B-1M | 1024 |

**How to run**:

```bash
cd methods/retroinfer/benchmark/AccuracyOverview

# ===== LongBench (single task example) =====
# Models: llama3.1-8b / qwen2.5-7b / glm-4-9b-chat-1m
# Using RetroInfer attention
bash Accuracy.sh llama3.1-8b RetroInfer LongBench narrativeqa -1 1024

# Using Full Flash Attention (baseline)
bash Accuracy.sh llama3.1-8b Full_Flash_Attn LongBench narrativeqa -1 1024

# ===== RULER / Synthetic (single task example) =====
bash Accuracy.sh qwen2.5-7b-1m RetroInfer Synthetic niah_single_1 -1 1024

# Batch run: edit run.sh to uncomment the desired task lines, then execute
bash run.sh
```

**Key scripts**:
- `Accuracy.sh` — Single-task prediction + evaluation entry point
- `pred.py` — Model inference
- `eval.py` — Metric computation
- `run.sh` — Contains all LongBench and RULER task commands (commented out by default)

**Output**: Results saved in `results/pred/<model>/<attn_type>/<benchmark>/`, logs in `log/pred/...`

---

## Experiment 2: Accuracy Budget

**Purpose**: Evaluate how different KV cache budgets affect accuracy, validating RetroInfer's accuracy retention under varying compression rates.

**Directory**: `benchmark/AccuracyOverview/` (same scripts as Experiment 1; switch budgets via the `--budget` parameter)

**Datasets & Models**:
| Dataset | Models | Budgets |
|---------|--------|---------|
| LongBench | Llama-3.1-8B, Qwen2.5-7B | 128, 256, 512, 1024 |
| RULER (64k) | Llama-3.1-8B, Qwen2.5-7B-1M | 128, 384, 1024, 4096 |

**How to run**:

```bash
cd methods/retroinfer/benchmark/AccuracyOverview

# LongBench with different budgets (e.g., narrativeqa)
bash Accuracy.sh llama3.1-8b RetroInfer LongBench narrativeqa -1 128
bash Accuracy.sh llama3.1-8b RetroInfer LongBench narrativeqa -1 256
bash Accuracy.sh llama3.1-8b RetroInfer LongBench narrativeqa -1 512
bash Accuracy.sh llama3.1-8b RetroInfer LongBench narrativeqa -1 1024

# RULER with different budgets (e.g., niah_single_1)
bash Accuracy.sh qwen2.5-7b-1m RetroInfer Synthetic niah_single_1 -1 128
bash Accuracy.sh qwen2.5-7b-1m RetroInfer Synthetic niah_single_1 -1 384
bash Accuracy.sh qwen2.5-7b-1m RetroInfer Synthetic niah_single_1 -1 1024
bash Accuracy.sh qwen2.5-7b-1m RetroInfer Synthetic niah_single_1 -1 4096
```

---

## Experiment 3: Efficiency Overview

**Purpose**: Evaluate RetroInfer's inference latency and throughput at different input lengths, demonstrating its speedup over Full Attention.

**Directory**: `benchmark/EfficencyOverview/`

**Configuration**:
| Models | Input Lengths | Output Length | Budget |
|--------|---------------|---------------|--------|
| Llama-3.1-8B, Qwen2.5-7B-1M | 4k, 8k, 16k, 32k, 64k, 128k | 32 | 1024 |

**How to run**:

```bash
cd methods/retroinfer/benchmark/EfficencyOverview

# Run efficiency evaluation (iterates over multiple input lengths automatically)
bash efficencyOverview.sh llama3.1-8b RetroInfer 0.018 0.232 1024 1 0 0 32 Full_Flash_Attn
bash efficencyOverview.sh qwen2.5-7b RetroInfer 0.018 0.232 1024 1 0 0 32 Full_Flash_Attn

# Or use the batch commands in run.sh
bash run.sh
```

**Key scripts**:
- `efficencyOverview.sh` — Efficiency evaluation entry point, loops over multiple input lengths
- `pred.py` — Inference + timing
- `run.sh` — Complete batch commands for all models and parameters

**Output**: Results saved in `results/efficencyOverview/`

---

## Experiment 4: Efficiency Budget

**Purpose**: Evaluate how different KV cache budgets affect inference efficiency, analyzing the trade-off between budget and speedup.

**Directory**: `benchmark/EfficencyOverview/`

**Configuration**:
| Models | Input Length | Budgets |
|--------|-------------|---------|
| Llama-3.1-8B, Qwen2.5-7B-1M | 4k | 128, 256, 512, 1024 |
| Llama-3.1-8B, Qwen2.5-7B-1M | 64k | 128, 384, 1024, 4096 |

**How to run**:

```bash
cd methods/retroinfer/benchmark/EfficencyOverview

# 4k input length + various budgets (Llama)
bash efficencyBudget.sh llama-3.1-8b RetroInfer 0.018 0.232 128 1 0 0 32 Full_Flash_Attn 4096
bash efficencyBudget.sh llama-3.1-8b RetroInfer 0.018 0.232 256 1 0 0 32 Full_Flash_Attn 4096
bash efficencyBudget.sh llama-3.1-8b RetroInfer 0.018 0.232 512 1 0 0 32 Full_Flash_Attn 4096
bash efficencyBudget.sh llama-3.1-8b RetroInfer 0.018 0.232 1024 1 0 0 32 Full_Flash_Attn 4096

# 64k input length + various budgets (Llama)
bash efficencyBudget.sh llama-3.1-8b RetroInfer 0.018 0.232 128 1 0 0 32 Full_Flash_Attn 65536
bash efficencyBudget.sh llama-3.1-8b RetroInfer 0.018 0.232 384 1 0 0 32 Full_Flash_Attn 65536
bash efficencyBudget.sh llama-3.1-8b RetroInfer 0.018 0.232 1024 1 0 0 32 Full_Flash_Attn 65536
bash efficencyBudget.sh llama-3.1-8b RetroInfer 0.018 0.232 4096 1 0 0 32 Full_Flash_Attn 65536

# Or use the pre-arranged batch commands in run.sh
bash run.sh
```

**Key scripts**:
- `efficencyBudget.sh` — Single-configuration efficiency evaluation
- `run.sh` — Batch commands covering all (model × length × budget) combinations

---

## Experiment 5: Memory (VRAM)

**Purpose**: Evaluate RetroInfer's peak GPU memory usage under different input/output lengths and budgets.

**Directory**: `benchmark/VRAMOverview/`

**Configuration**:
| Models | Input Length | Output Length | Budgets |
|--------|-------------|---------------|---------|
| Llama-3.1-8B, Qwen2.5-7B-1M | 1k | 2, 4096 | 64, 512 |
| Llama-3.1-8B, Qwen2.5-7B-1M | 4k ~ 128k | 2 | 1024 |

**How to run**:

```bash
cd methods/retroinfer/benchmark/VRAMOverview

# 1k input + short output (output_length=2) + different budgets
bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 64 1024 2
bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 512 1024 2

# 1k input + long output (output_length=4096) + different budgets
bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 64 1024 4096
bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 512 1024 4096

# Sweep over input lengths (output_length=2, budget=1024)
bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 1024 4096 2
bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 1024 8192 2
bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 1024 16384 2
bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 1024 32768 2
bash VRAMOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 1024 65536 2

# Or use the pre-arranged batch commands in run.sh
bash run.sh
```

**Key scripts**:
- `VRAMOverview.sh` — Memory evaluation entry point
- `pred.py` — Inference + VRAM recording
- `config.sh` — Common parameter configuration
- `run.sh` — Batch commands for all (model × length × budget) combinations

**Output**: Results saved in `results/VRAMOverview/`

---

## Experiment 6: Recall

**Purpose**: Evaluate RetroInfer's KV cache retrieval recall rate — i.e., how well the coarse-grained selection stage preserves the key-value pairs needed for exact attention computation.

**Directory**: `benchmark/RECALLOverview/`

**Datasets & Models**:
| Dataset | Models | Budgets |
|---------|--------|---------|
| LongBench (narrativeqa, qasper) | Llama-3.1-8B, Qwen2.5-7B | 128, 256, 512, 1024 |
| RULER (64k, niah_single_3/vt/fwe) | Llama-3.1-8B, Qwen2.5-7B-1M | 128, 384, 1024, 4096 |

**How to run**:

```bash
cd methods/retroinfer/benchmark/RECALLOverview

# LongBench recall (e.g., narrativeqa)
bash RECALLOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 128 LongBench narrativeqa
bash RECALLOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 256 LongBench narrativeqa
bash RECALLOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 512 LongBench narrativeqa
bash RECALLOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 1024 LongBench narrativeqa

# RULER recall (64k, e.g., niah_single_3)
bash RECALLOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 128 synthetic niah_single_3
bash RECALLOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 384 synthetic niah_single_3
bash RECALLOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 1024 synthetic niah_single_3
bash RECALLOverview.sh llama-3.1-8b Full_Flash_Attn RetroInfer 4096 synthetic niah_single_3

# Or use the pre-arranged batch commands in run.sh
bash run.sh
```

**Key scripts**:
- `RECALLOverview.sh` — Recall evaluation entry point (`--recall` mode)
- `pred.py` — Inference + recall statistics
- `evaluate.py` — Recall rate computation
- `run.sh` — Batch run commands

---

## Experiment 7: LongBenchV2

**Purpose**: Evaluate RetroInfer's reasoning capability on the more challenging LongBenchV2 benchmark.

**Directory**: `benchmark/LongBenchV2/`

**Configuration**: Qwen2.5-7B-1M, budget 4096, Chain-of-Thought reasoning

**How to run**:

```bash
cd methods/retroinfer/benchmark/LongBenchV2

# RetroInfer mode
bash LongBenchV2.sh Qwen2.5-7B-Instruct-1M Full_Flash_Attn RetroInfer 4096

# Or simply execute run.sh
bash run.sh
```

**Key scripts**:
- `LongBenchV2.sh` — Prediction entry point
- `pred.py` — Inference
- `result.py` — Result processing
- `config.sh` — Parameter configuration
- `prompts/` — Prompt templates

---

## Experiment 8: GSM8K

**Purpose**: Evaluate RetroInfer's impact on mathematical reasoning using the GSM8K dataset.

**Directory**: `benchmark/GSM8k/`

**Configuration**: DeepSeek-R1-Distill-Qwen-1.5B, budget 360, 8-shot CoT

**How to run**:

```bash
cd methods/retroinfer/benchmark/GSM8k

# RetroInfer mode
bash gsm8k_run.sh deepseek-r1-distill-qwen-1.5b RetroInfer Full_Flash_Attn 360 8 gsm8k-cot

# Full Attention baseline
bash gsm8k_run.sh deepseek-r1-distill-qwen-1.5b Full_Flash_Attn minfer 360 8 gsm8k-cot

# Or simply execute run.sh
bash run.sh
```

**Key scripts**:
- `gsm8k_run.sh` — Prediction + evaluation entry point
- `evaluation_gsm8k.py` / `evaluate.py` — Evaluation logic
- `examples.py` — Few-shot examples
- `config.sh` — Parameter configuration
- `data/` — GSM8K dataset
- `tool/` — Data processing utilities

---

## Environment

For environment setup, model weight paths, and dependencies, refer to the following configuration files:
- `methods/retroinfer/config/` — Model configs (JSON), defining RetroInfer parameters for each model
- `benchmarks/local_paths.json` — Dataset path configuration
- `ENVIRONMENT.md` in the project root — Conda environment and dependency instructions

## Quick Parameter Reference

| Parameter | Meaning | Common Values |
|-----------|---------|---------------|
| `attn_type` | Attention type | `RetroInfer`, `Full_Flash_Attn` |
| `prefill_method` | Attention implementation for prefill | `Full_Flash_Attn`, `minfer` |
| `budget` | KV cache budget (fixed mode) | 128, 256, 512, 1024, 4096 |
| `ratio_or_fixed` | Budget mode: 1=fixed, 0=ratio | 1 |
| `benchmark` | Dataset name | `LongBench`, `Synthetic` (RULER) |
| `task` | Specific task name | See `config/dataset2maxlen.json` |
| `dtype` | Inference precision | `bf16` |
