# Data Preparation and Preselection

## LongBench

LongBench datasets are downloaded on demand through Hugging Face Datasets; no manual data preparation is required.  Replace `dataset` with the desired LongBench subset name and load its test split directly:

```python
from datasets import load_dataset

data = load_dataset("THUDM/LongBench", f"{dataset}", split="test")
```

## RULER Data Generation

Before running any RULER experiment—either through `benchmarks/ruler/` or a method's internal RULER runner—generate the required synthetic data first:

```bash
bash benchmarks/ruler/generate_datasets.sh
```

Run this command from any directory. By default, it generates 50 samples for every RULER synthetic task at 4K, 8K, 16K, 32K, 64K, 128K, and 192K tokens for `llama-3.1-8b` and `qwen-2.5-7b-1m`.

Models and lengths can be selected as needed. For example, the following command prepares only 64K and 128K data for Qwen:

```bash
bash benchmarks/ruler/generate_datasets.sh \
  --models qwen-2.5-7b-1m \
  --lengths 65536 131072
```

The script reads the only retained RULER source assets from `benchmarks/ruler/data/synthetic/json/` and writes generated datasets to `benchmarks/ruler/benchmark_root/<model>/`. Method-local RULER runners use the same generated data, so do not copy the source JSON files into `methods/`.

RULER predictions are kept separate from generated data at `benchmarks/ruler/benchmark_root_pred/synthetic/<model>/<method>/<parameter-signature>/<length>/`. For example, a StreamingLLM run with a budget of 1024 and a context length of 64K writes to `benchmark_root_pred/synthetic/llama-3.1-8b/streaming/budget-1024/65536/`.

The generic RULER runner takes a method followed directly by its option flags; it does not take an `overview` or `budget` positional argument. The `*-overview.sh` and `*-budget.sh` scripts select their respective experiment presets through the arrays they define.

## LongBenchV2

The repository evaluates LongBenchV2 with a prefiltered 64K--192K-token subset, stored at `benchmarks/longbenchv2/filtered_longbench_v2_64k-192k.jsonl`.  Build (or rebuild) this file from the Hugging Face dataset by running:

```bash
bash benchmarks/longbenchv2/filter_64k_192k.sh
```

The script downloads the `train` split of `THUDM/LongBench-v2`, constructs each full multiple-choice input (context, question, and choices), tokenizes it with `Qwen/Qwen2.5-7B-Instruct-1M`, and retains inputs with lengths in `[65536, 196608)` tokens.  It writes the retained records, including their tokenized lengths, to the path above.  The unified LongBenchV2 runner reads this file directly.

## GSM8K

Download the GSM8K test data and place it at the following exact path:

```
benchmarks/gsm8k/data/gsm8k_test.jsonl
```

The GSM8K benchmark scripts load this local JSONL file; keep the filename and directory unchanged.

## Efficiency and Memory Input

`benchmarks/myinput.txt` is the synthetic long input used by efficiency and GPU-memory experiments.  It supports input lengths up to 256K tokens.

## HeadKV Head Score (Retrieval Head Detection)

HeadKV pre-computes per-head importance scores via needle-in-a-haystack detection (`methods/HeadKV/Important_Head/`). A "needle" sentence is embedded at variable depths in a long document; the model's per-layer per-head attention to the needle is aggregated into a score, where higher values indicate stronger retrieval/reasoning contribution.

Two scripts handle two modes:
- `retrieval_head_detection.py` → `*_retrieval_heads.json` for `head_choice=copy`
- `retrieval_head_detection_r2.py` → `*_retrieval_reasoning_heads.json` for `head_choice=reason`

Pre-computed scores are in `methods/HeadKV/Important_Head/head_score/` for Llama-3-8B, Qwen2.5-7B, Mistral-7B-v0.2, GLM-4-9B, and DeepSeek-R1-Distill-Qwen-1.5B.

```bash
cd methods/HeadKV/Important_Head
bash start.sh
```

At runtime, `ReasonSnapKVCluster` normalizes the scores into per-head capacity:

```
head_capacity = round(score_normalized * (base_capacity // beta) * num_layers * num_heads + (base_capacity - base_capacity // beta))
```

## DuoAttention Attention Pattern Training

DuoAttention learns which attention heads need full KV cache (vs. streaming sink+recent only) by training a binary `full_attention_heads` mask. Only the mask parameters are updated—all model weights are frozen. The training script (`train/duo_attn/train.py`) loads the model, injects a dual-route attention forward, and optimizes the mask via L1-regularized gradient descent on a multi-passkey retrieval dataset. Output is a `.tsv` file per model under `attn_patterns/`.

```bash
cd methods/duo-attention/train

# Single model (Llama/Mistral):
CUDA_VISIBLE_DEVICES=0 bash duo_attn/train.sh meta-llama/Llama-3.1-8B-Instruct

# Batch training (run_train.sh — edit model list first):
bash scripts/run_train.sh
```

Key hyperparameters in `train.sh`:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--sink_size` | 128 | Number of initial sink tokens |
| `--recent_size` | 256 | Number of recent window tokens |
| `--max_length` | 131072 | Training context length |
| `--num_steps` | 2000 | Optimization steps |
| `--lr` | 0.02 | Learning rate for full_attention_heads |
| `--reg_weight` | 0.05 | L1 regularization strength |
| `--num_passkey` | 5 | Number of passkeys per sample |
| `--dataset_format` | multiple_passkey | Dataset type |
| `--context_length_min/max` | 1000/131072 | Context length sampling range |

### Runtime Integration

At inference time, `duo_attention_load.py` (or the eval pipeline) reads the TSV file from `attn_patterns/`, converts it to a `full_attention_heads` list, and calls `enable_duo_attention_eval(model, full_attention_heads, sink_size, recent_size)`.  The runtime `budget` parameter is decomposed into `sink_size + recent_size`; heads not in the full-attention list only store these sink + recent tokens.









# Threshold Training Guide

This directory contains the threshold profiling code used by XAttention. The training script profiles a long-context model on a JSON list of prompt strings, estimates per-layer/per-head XAttention thresholds, and writes the resulting threshold table to a Python or JSON file.

## Directory Layout

```text
threshold/
├── text.json                         # Default profiling texts
├── train_scripts/
│   ├── train_threshold.py            # Main threshold profiling entry point
│   ├── run_llama_threshold.sh        # Wrapper for Llama models
│   ├── run_qwen_threshold.sh         # Wrapper for Qwen2 models
│   ├── run_ds_qwen_threshold.sh      # Wrapper for DeepSeek-Distill-Qwen models
│   └── run_glm_threshold.sh          # Wrapper for GLM/chatglm models
└── threshold/
    ├── llama_threshold.py            # Existing Llama threshold tables
    ├── qwen_threshold.py             # Existing Qwen threshold tables
    ├── ds_qwen.py                    # Existing DeepSeek-Distill-Qwen threshold table
    └── glm_threshold.py              # Existing GLM threshold tables
```

## Requirements

Run the commands from `methods/x-attention` unless stated otherwise.

The scripts assume:

- The `xattn` conda environment exists.
- The XAttention package is installed or importable from `methods/x-attention`.
- `flash-attn` and the block sparse attention dependency used by `xattn.src.Xattention` are installed.
- The target Hugging Face model is accessible locally or from the Hugging Face Hub.
- CUDA is available.

Basic environment check:

```bash
cd methods/x-attention
conda run -n xattn python -c "import torch; import flash_attn; import xattn; print(torch.cuda.is_available())"
```

## Input Format

`train_threshold.py` expects `--text_path` to point to a JSON file containing a non-empty list of strings:

```json
[
  "First profiling prompt ...",
  "Second profiling prompt ..."
]
```

The default wrapper scripts use:

```text
threshold/text.json
```

Use long prompts that resemble the workloads where the generated thresholds will be used. The profiler only supports prefill-style full-sequence attention for threshold collection.

## Quick Start

From `methods/x-attention`, run one of the wrapper scripts.

Llama:

```bash
bash threshold/train_scripts/run_llama_threshold.sh 0.9
```

Qwen2:

```bash
bash threshold/train_scripts/run_qwen_threshold.sh 0.9
```

DeepSeek-Distill-Qwen:

```bash
bash threshold/train_scripts/run_ds_qwen_threshold.sh 0.8
```

GLM/chatglm:

```bash
bash threshold/train_scripts/run_glm_threshold.sh 0.9
```

The first positional argument is `p`, the target cumulative attention mass. Supported common values are `0.8`, `0.85`, `0.9`, and `0.95`.

## Wrapper Script Configuration

Each wrapper can be configured with environment variables:

```bash
MODEL_PATH=/path/to/local/model \
TEXT_PATH=/path/to/profile_texts.json \
OUTPUT_DIR=xattn/threshold/threshold \
bash threshold/train_scripts/run_qwen_threshold.sh 0.9
```

Supported variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `P` | Threshold mass target. Overrides the first positional argument. | Model-specific |
| `MODEL_PATH` | Hugging Face repo ID or local model path. | Set by each wrapper |
| `TEXT_PATH` | JSON list of profiling prompts. | `threshold/text.json` |
| `OUTPUT_DIR` | Directory for the generated threshold file. | `xattn/threshold/threshold` |
| `OUTPUT_PATH` | Full output file path. Overrides `OUTPUT_DIR`. | Derived from model and `p` |
| `CHUNK_TEXTS` | Number of texts per profiling chunk. `0` disables chunking. | `0` |
| `BATCH_OUTPUT_DIR` | Directory for per-chunk threshold JSON files. | Derived from `OUTPUT_PATH` |

Examples:

```bash
P=0.95 bash threshold/train_scripts/run_llama_threshold.sh
```

```bash
MODEL_PATH=/path/to/Qwen2.5-7B-Instruct-1M \
TEXT_PATH=/path/to/profile_texts.json \
OUTPUT_PATH=xattn/threshold/threshold/qwen_threshold_p90.py \
bash threshold/train_scripts/run_qwen_threshold.sh 0.9
```

## Direct Python Usage

The wrappers call `train_threshold.py`. You can also run it directly:

```bash
conda run -n xattn python threshold/train_scripts/train_threshold.py \
  --name_or_path Qwen/Qwen2.5-7B-Instruct-1M \
  --model_type qwen2 \
  --p 0.9 \
  --text_path threshold/text.json \
  --output_path xattn/threshold/threshold/qwen_threshold_p90.py \
  --output_format py
```

Important arguments:

| Argument | Description |
| --- | --- |
| `--name_or_path` | Hugging Face model ID or local model directory. Required. |
| `--model_type` | One of `auto`, `llama`, `qwen2`, or `glm`. |
| `--p` | Target cumulative attention mass in `(0, 1]`. Required. |
| `--text_path` | JSON file containing profiling strings. |
| `--output_path` | Output threshold table path. |
| `--output_format` | `py`, `json`, or `auto`. |
| `--output_var` | Variable name used when writing a Python file. |
| `--stride` | XAttention stride. Defaults to `16` for DeepSeek models and `8` otherwise. |
| `--block_size` | Attention block size. Default: `128`. |
| `--device_map` | Transformers device map. Defaults to `balanced` for Llama and `auto` otherwise. |
| `--torch_dtype` | `float32`, `float16`, `bfloat16`, or `auto`. Default: `bfloat16`. |
| `--attn_implementation` | Optional Transformers attention implementation. Qwen2 defaults to `flash_attention_2`. |
| `--chunk_texts` | Process profiling texts in chunks to reduce peak memory pressure. |

## Chunked Profiling

For large profiling sets, use chunked profiling:

```bash
CHUNK_TEXTS=8 \
BATCH_OUTPUT_DIR=outputs/qwen_threshold_chunks \
bash threshold/train_scripts/run_qwen_threshold.sh 0.9
```

In chunked mode, the script:

1. Profiles each chunk independently.
2. Saves each chunk threshold as JSON.
3. Merges chunks by taking the element-wise maximum threshold.
4. Writes the merged threshold table to `OUTPUT_PATH`.

## Output Files

Python output files define a threshold table and usually alias it as `max`:

```python
qwen_fuse_90 = [[...], ...]
max = qwen_fuse_90
```

Default wrapper outputs:

| Wrapper | Default output |
| --- | --- |
| `run_llama_threshold.sh 0.9` | `xattn/threshold/threshold/llama_threshold_p90.py` |
| `run_qwen_threshold.sh 0.9` | `xattn/threshold/threshold/qwen_threshold_p90.py` |
| `run_ds_qwen_threshold.sh 0.8` | `xattn/threshold/threshold/ds_qwen_p80.py` |
| `run_glm_threshold.sh 0.9` | `xattn/threshold/threshold/glm_threshold_p90.py` |

If an evaluation script imports a specific existing module such as `threshold.threshold.qwen_threshold`, either update that module to import the generated table, or copy the generated variable into the module expected by the evaluation code.

## Troubleshooting

`TEXT_PATH does not exist`

Set `TEXT_PATH` to a valid JSON file:

```bash
TEXT_PATH=/path/to/profile_texts.json bash threshold/train_scripts/run_qwen_threshold.sh 0.9
```

`Failed to import xattn`

Run from `methods/x-attention`, install the package, or export `PYTHONPATH`:

```bash
cd methods/x-attention
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

CUDA out of memory

Use fewer or shorter profiling texts, reduce batch chunk size, or enable chunked profiling:

```bash
CHUNK_TEXTS=1 bash threshold/train_scripts/run_llama_threshold.sh 0.9
```

Unsupported model type

The profiler currently supports Llama, Qwen2-compatible models, and GLM/chatglm. Use `--model_type` explicitly if automatic detection picks the wrong family.

Slow model download

Set `MODEL_PATH` to a local model directory:

```bash
MODEL_PATH=/path/to/Qwen2.5-7B-Instruct-1M bash threshold/train_scripts/run_qwen_threshold.sh 0.9
```
