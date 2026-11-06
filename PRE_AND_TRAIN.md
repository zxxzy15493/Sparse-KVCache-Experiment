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
│   ├── run_qwen3_threshold_64k.sh    # Qwen3 prompts up to 64K, YaRN 2x
│   ├── run_qwen3_threshold_128k.sh   # Qwen3 prompts from 64K to 128K, YaRN 4x
│   ├── run_qwen3_threshold_both.sh   # Run both Qwen3 calibrations sequentially
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

Qwen3-32B, prompts up to 64K:

```bash
MODEL_PATH=Qwen/Qwen3-32B \
bash threshold/train_scripts/run_qwen3_threshold_64k.sh 0.9
```

This reads `threshold/qwen3_64k.json` and uses `max_position_embeddings=65536`
with YaRN `factor=2.0`.

Qwen3-32B, prompts from 64K to 128K:

```bash
MODEL_PATH=Qwen/Qwen3-32B \
bash threshold/train_scripts/run_qwen3_threshold_128k.sh 0.9
```

This reads `threshold/qwen3_128k.json` and uses Qwen3's maximum validated
context: `max_position_embeddings=131072` with YaRN `factor=4.0`.

Both Qwen3 wrappers use `stride=16`, `block_size=128`, BF16, and write JSON.
For Qwen3-32B, the resulting table must have shape `[64, 64]`.
The default `CHUNK_TEXTS=1` saves a recoverable JSON result for each profiling
text before merging them by element-wise maximum.

Run all four target masses (`0.8`, `0.85`, `0.9`, and `0.95`) for both Qwen3
ranges sequentially with one command:

```bash
MODEL_PATH=Qwen/Qwen3-32B \
bash threshold/train_scripts/run_qwen3_threshold_both.sh
```

Pass one or more positional values to run only a subset, for example:

```bash
bash threshold/train_scripts/run_qwen3_threshold_both.sh 0.85 0.9
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
| `MAX_TEXTS` | Profile only the first N texts for a smoke run; `0` means all. | `0` |
| `CUDA_VISIBLE_DEVICES` | GPUs used by the Qwen3 wrappers. | `0,1` |
| `DEVICE_MAP` | Device placement for the Qwen3 wrappers. | `balanced` |
| `RESERVE_MEMORY_GIB` | Free memory reserved per GPU for activations by the Qwen3 wrappers. | `8` |
| `MAX_MEMORY` | Explicit JSON weight budgets for the Qwen3 wrappers; overrides automatic budgets. | Empty |
| `HEAD_CHUNK_SIZE` | Attention heads profiled together by the Qwen3 wrappers; `0` means all. | `4` |
| `ESTIMATE_CHUNK_SIZE` | Query tokens per estimate chunk in the Qwen3 wrappers. | `16384` |
| `RESUME_CHUNKS` | Reuse existing chunk JSON files in the Qwen3 wrappers. | `true` |

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
| `--model_type` | One of `auto`, `llama`, `qwen2`, `qwen3`, or `glm`. |
| `--p` | Target cumulative attention mass in `(0, 1]`. Required. |
| `--text_path` | JSON file containing profiling strings. |
| `--output_path` | Output threshold table path. |
| `--output_format` | `py`, `json`, or `auto`. |
| `--output_var` | Variable name used when writing a Python file. |
| `--stride` | XAttention stride. Defaults to `16` for Qwen3/DeepSeek and `8` otherwise. |
| `--block_size` | Attention block size. Default: `128`. |
| `--device_map` | Transformers device map. Defaults to `balanced` across visible GPUs. |
| `--max_memory` | JSON weight budgets, e.g. `'{"0":"40GiB","1":"40GiB"}'`. Indices are relative to `CUDA_VISIBLE_DEVICES`. |
| `--reserve_memory_gib` | With an inferred device map, subtract this many GiB from each GPU's free memory for weight placement. Default: `8`; explicit `--max_memory` overrides it. |
| `--head_chunk_size` | Heads profiled together per layer. Default: `4`; `0` uses all heads. Smaller values reduce temporary memory. |
| `--estimate_chunk_size` | Query tokens per XAttention estimate chunk. Default: `16384`; must be divisible by both `block_size` and `stride*64`. |
| `--torch_dtype` | `float32`, `float16`, `bfloat16`, or `auto`. Default: `bfloat16`. |
| `--attn_implementation` | Optional Transformers attention implementation. Qwen2/Qwen3 default to `flash_attention_2`. |
| `--rope_scaling_json` | Optional JSON object assigned to `config.rope_scaling` before loading the model. |
| `--max_position_embeddings` | Optional context limit assigned before model construction; `0` preserves the model config. |
| `--chunk_texts` | Number of texts per saved profiling chunk; does not split an individual prompt or lower its attention memory requirement. |
| `--max_texts` | Profile only the first N input texts; `0` means all. |

## Multi-GPU Profiling

Calibration runs without gradients and uses model parallelism: complete decoder layers
are placed on different GPUs. Both Qwen3 wrappers default to GPUs `0,1` and `balanced`
placement. Other model wrappers accept the same Python options through their trailing
arguments. This is a single Python process; launch with `python` or the shell wrappers.

For example, from `methods/x-attention`, using an environment with the dependencies installed:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
HEAD_CHUNK_SIZE=4 \
RESERVE_MEMORY_GIB=12 \
bash threshold/train_scripts/run_qwen3_threshold_128k.sh 0.9
```

For GPUs with different capacities or other memory usage, set weight budgets explicitly:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
MAX_MEMORY='{"0":"40GiB","1":"60GiB"}' \
HEAD_CHUNK_SIZE=2 \
ESTIMATE_CHUNK_SIZE=2048 \
bash threshold/train_scripts/run_qwen3_threshold_64k.sh 0.9
```

The budgets apply to logical GPU indices after `CUDA_VISIBLE_DEVICES` remapping and
limit model weight placement, not total runtime memory. Leave room for hidden states,
MLP activations, and attention statistics. Automatic budgets reserve 8 GiB per visible
GPU by default; the appropriate reserve depends on model size and prompt length.
An explicit placement such as `--device_map cuda:0` bypasses automatic weight budgets;
`--max_memory` requires an inferred map such as `balanced` or `auto`.

The profiler skips the LM output head, keeps GQA K/V in their compact form for
FlashAttention, computes statistics a few query heads at a time, and allocates masks
on each layer's device. Head chunking does not divide a single layer across GPUs.
It preserves the per-head threshold formula and the element-wise maximum across texts.
Floating-point kernel differences may cause small numerical differences. Use
`--head_chunk_size 0` to compare against processing all heads together.

Run the regression tests from `methods/x-attention`:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_qwen3_threshold_profile.py \
  tests/test_threshold_memory.py \
  tests/test_transformers_457_compat.py
```

The tests cover threshold statistics against a dense reference, compact GQA,
output lifetime, device hooks, weight budgets, and both launchers. CUDA tests
compare FP16/BF16 head chunking and a randomly initialized two-layer Qwen3 model
on one GPU versus two GPUs. They use small inputs and skip when the required
GPUs are unavailable; they do not load the 32B checkpoint.

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
| `run_qwen3_threshold_64k.sh 0.9` | `threshold/threshold/qwen3_32b_yarn2_64k_p90_s16.json` |
| `run_qwen3_threshold_128k.sh 0.9` | `threshold/threshold/qwen3_32b_yarn4_128k_p90_s16.json` |
| `run_ds_qwen_threshold.sh 0.8` | `xattn/threshold/threshold/ds_qwen_p80.py` |
| `run_glm_threshold.sh 0.9` | `xattn/threshold/threshold/glm_threshold_p90.py` |

If an evaluation script imports a specific existing module such as `threshold.threshold.qwen_threshold`, either update that module to import the generated table, or copy the generated variable into the module expected by the evaluation code.

For the unified Qwen3 loader, pass the generated JSON matching the inference
context configuration. For example, the 64K table is selected with:

```bash
python benchmarks/longbench/qwen3_32b_method_smoke.py \
  --method xattention \
  --set threshold_path=methods/x-attention/threshold/threshold/qwen3_32b_yarn2_64k_p90_s16.json
```

The loader verifies the table shape against the selected model and rejects
non-finite values or values outside `[0, 1]`. Do not reuse a table trained with
a different stride, model size, RoPE/YaRN setting, or substantially different
context-length distribution.

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

For a single long prompt, reduce the number of heads and estimate query tokens
processed together, and leave more free memory for activations:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
HEAD_CHUNK_SIZE=1 \
ESTIMATE_CHUNK_SIZE=2048 \
RESERVE_MEMORY_GIB=16 \
bash threshold/train_scripts/run_qwen3_threshold_128k.sh 0.9
```

`CHUNK_TEXTS=1` saves progress after each text. Texts are already processed one at a
time, so this setting does not resolve a single prompt's attention or MLP OOM.

Unsupported model type

The profiler currently supports Llama, Qwen2-compatible models, and GLM/chatglm. Use `--model_type` explicitly if automatic detection picks the wrong family.

Slow model download

Set `MODEL_PATH` to a local model directory:

```bash
MODEL_PATH=/path/to/Qwen2.5-7B-Instruct-1M bash threshold/train_scripts/run_qwen_threshold.sh 0.9
```
