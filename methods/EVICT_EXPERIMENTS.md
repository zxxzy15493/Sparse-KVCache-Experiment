# EVICT — KV Cache Eviction Experiment Guide

This project contains complete experiment code for four KV Cache eviction methods evaluated on LongBench / LongBenchV2 / RULER / GSM8K benchmarks.

## Directory Structure

```
method/
├── streaming/          # StreamingLLM
│   ├── experiment/     # Experiment scripts (LongBench / RULER / Recall / Efficiency)
│   └── LongBenchV2/    # LongBenchV2 experiment
├── SnapKV/             # SnapKV
│   ├── experiment/
│   └── LongBenchV2/
├── h2o/                # H2O (Heavy-Hitter Oracle)
│   └── experiment/
└── keyformer/          # Keyformer
    └── experiment/
```

Each method's `experiment/` directory follows the same structure:

```
experiment/
├── run_<method>.sh          # LongBench inference script
├── run_recall.sh            # Recall script
├── run_time.sh              # Component timing script
├── run_ef.sh                # Efficiency script
├── run_GPUm.sh              # GPU memory script
├── run_eval.sh              # LongBench evaluation script (calls evalJson.py)
├── RULER-main/              # RULER benchmark
├── Gsm8k/                   # GSM8K inference test
├── <method>.py              # LongBench inference
├── recall.py                # Recall, Recall@100, Attention coverage across budget levels on LongBench
├── breaktime.py             # Component timing across different input lengths
├── Efficency.py             # Efficiency across different input lengths and budgets
├── GPUm.py                  # Peak GPU memory across different input/output lengths
├── evalJson.py / metrics.py # LongBench result evaluation
└── config/                  # Model paths, dataset paths, prompt templates, config files (SnapKV)
```

## Supported Models

- Llama-3.1-8B-Instruct
- Qwen2.5-7B-Instruct / Qwen2.5-7B-Instruct-1M
- glm-4-9b-chat-1m

---

## 1. LongBench Evaluation

Each method has its own entry script and Python file; the rest is shared:

| Method | Directory | Entry Script | Python File | Enable Flag |
|------|------|----------|------------|----------|
| **StreamingLLM** | `streaming/experiment` | `run_streaming.sh` | `streaming.py` | `--enable_streaming` |
| **SnapKV** | `SnapKV/experiment` | `run_snapkv.sh` | `pred_snap.py` | (via `--compress_args_path` JSON config) |
| **H2O** | `h2o/experiment` | `run_h2o.sh` | `h2o.py` | `--enable_h2o_cache` |
| **Keyformer** | `keyformer/experiment` | `run_keyformer.sh` | `keyformer.py` | `--keyformer` |

```bash
cd method/<method>/experiment
bash run_<method>.sh    
```

### StreamingLLM Parameters

| Parameter | Description | Default |
|------|------|--------|
| `--start_size` | Number of sink tokens | 16 |
| `--recent_size` | Recent window size | 1008 |

Cache budget = `start_size + recent_size`.

### SnapKV Parameters

Loaded via JSON config through `--compress_args_path`:

| Field | Description |
|------|------|
| `window_sizes` | Window size per layer |
| `max_capacity_prompts` | Max prompt cache capacity per layer |
| `kernel_sizes` | Pooling kernel size |
| `pooling` | Pooling method |

### H2O Parameters

| Parameter | Description | Default |
|------|------|--------|
| `--heavy_hitter_size` | Heavy-hitter retention count | 992 |
| `--recent_size` | Recent token retention count | 32 |

Cache budget = `heavy_hitter_size + recent_size`.

### Keyformer Parameters

| Parameter | Description | Default |
|------|------|--------|
| `--key_size` | Key retention count | 992 |
| `--recent_size` | Recent token retention count | 32 |
| `--tau_init` | Initial threshold | 1.0 |
| `--tau_delta` | Threshold decay step | 0.01 |

Common parameters: `--model_name_or_path` (model name), `--data_root` .

---

## 2. Other Experiments

Shared across all methods, run under each `experiment/` directory:

```bash
bash run_recall.sh      # KV Cache recall, recall@100, attention coverage
bash run_time.sh        # Component timing
bash run_ef.sh          # Efficiency
bash run_GPUm.sh        # GPU memory
bash run_eval.sh        # LongBench evaluation
```

---

## 3. RULER

Shared across all methods, under each `experiment/RULER-main/scripts/`:

```bash
bash run.sh <model_name> <benchmark_name>   # RULER dataset inference
bash run_recall.sh                          # RULER dataset recall
```

Parameters: `HEAVY_HITTER_SIZE`, `RECENT_SIZE` are set directly in the script.

---

## 4. GSM8K

Shared across all methods, under each `experiment/Gsm8k/`:

```bash
bash run.sh              # Inference
bash run_eval.sh         # Evaluation
```

---

## 5. LongBenchV2

Supported by StreamingLLM and SnapKV:

```bash
cd method/<method>/LongBenchV2
bash run.sh
```

| Parameter | Description |
|------|------|
| `--enable_streaming` | Enable StreamingLLM |
| `--start_size` | Sink size, default 16 |
| `--recent_size` | Window size, default 4080 |
| `--cot` | Enable Chain-of-Thought |
| `--rag N` | Enable RAG, retrieve top N chunks |
| `--no_context` | No-context mode |

---

## Configuration Files

Under each `experiment/config/` directory:

| File | Content |
|------|------|
| `model2path.json` | Model name → local path mapping |
| `model2maxlen.json` | Model name → max context length |
| `dataset2prompt.json` | Dataset name → prompt template |
| `dataset2maxlen.json` | Dataset name → max generation length |
