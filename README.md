# Anonymous Artifact Repository

This repository contains the code and experimental artifacts prepared for an anonymous double-blind review. To preserve anonymity, it contains no author, institutional, or machine-specific information.

## Changelog

| Date | Update |
| --- | --- |
| 20260723 | Uploaded the initial code. |
|  |  |

## Supported Methods and Models

The repository implements and compares the following methods; each name links to its implementation directory:

The Full Attention, TopK, and TopP baselines are implemented in [`methods/pqcache/`](methods/pqcache/) alongside PQCache.

- [`full_attention`](methods/pqcache/), [`topk`](methods/pqcache/), [`topk32`](methods/pqcache/), [`topp`](methods/pqcache/), [`topp32`](methods/pqcache/)
- [`h2o`](methods/h2o/), [`keyformer`](methods/keyformer/), [`snapkv`](methods/SnapKV/), [`streaming`](methods/streaming/)
- [`quest`](methods/quest/), [`sparq`](methods/sparq/), [`xattention`](methods/x-attention/), [`flexprefill`](methods/flexPrefill/), [`retroinfer`](methods/retroinfer/), [`magicpig`](methods/magicpig/), [`pqcache`](methods/pqcache/),[`clusterkv`](methods/ClusterKV/),
- [`minference`](methods/minference/), [`headkv`](methods/HeadKV/), [`adakv`](methods/Adakv/), [`cakekv`](methods/cakekv/), [`duo-attention`](methods/duo-attention/), [`pyramidkv`](methods/PyramidKV/)

The unified entry points use the following model abbreviations:

| Abbreviation | Hugging Face identifier |
| --- | --- |
| `llama-3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` |
| `qwen-2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` |
| `qwen-2.5-7b-1m` | `Qwen/Qwen2.5-7B-Instruct-1M` |
| `glm-4-9b-1m` | `zai-org/glm-4-9b-chat-1m` |
| `ds-qwen-1.5b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |

See [ENVIRONMENT.md](ENVIRONMENT.md) for environment setup and [PRE_AND_TRAIN.md](PRE_AND_TRAIN.md) for data preparation and preselection. After setup, activate the environment with:

```bash
conda activate kv
```

To override the default Hugging Face model location with a local path, use only [`benchmarks/local_paths.json`](benchmarks/local_paths.json).

## Codebase Architecture

```
KVCache-SIGMOD/
├── README.md                 # Repository overview and unified benchmark usage
├── ENVIRONMENT.md            # Environment setup
├── PRE_AND_TRAIN.md          # Dataset preparation and preselection
├── infer.py                  # Unified single-input inference entry point
├── run.sh                    # Examples for invoking infer.py
├── method_configs/           # Per-method default parameters (YAML)
├── loaders/                  # Unified method loaders and model patching
│   ├── __init__.py           # Method name → loader-module dispatch
│   ├── full_attention_load.py
│   ├── pqcache_load.py       # TopK / TopP / PQCache family
│   └── <method>_load.py      # Loader for each remaining supported method
├── benchmarks/               # Unified evaluation framework
│   ├── common.py             # Model paths, shared arguments, result-path utilities
│   ├── local_paths.json      # Optional local model-path overrides (user-created)
│   ├── run_runtest.py        # Run LongBench, LongBenchV2, and GSM8K smoke tests
│   ├── longbench/            # LongBench: prediction, evaluation, prompts, scripts
│   ├── ruler/                # RULER: generation, prediction, evaluation, scripts
│   ├── longbenchv2/          # LongBenchV2: filtering, prediction, evaluation, scripts
│   ├── gsm8k/                # GSM8K: prediction, evaluation, data, scripts
│   ├── Longbench_recall/     # LongBench recall-analysis data
│   ├── Ruler_recall/         # RULER recall-analysis data
│   └── myinput.txt           # Up-to-256K-token input for efficiency and memory tests
└── methods/                  # Implementations and method-specific paper experiments
    ├── pqcache/              # Full attention, TopK, TopP, and PQCache
    ├── ClusterKV/  HeadKV/  PyramidKV/  cakekv/  Adakv/
    ├── SnapKV/  streaming/  h2o/  keyformer/
    ├── quest/  sparq/  x-attention/  flexPrefill/  minference/
    ├── retroinfer/  magicpig/  duo-attention/
    ├── EVICT_EXPERIMENTS.md      # Experiment guide for eviction methods
    └── Heuristic_EXPERIMENTS.md  # Experiment guide for heuristic selection methods
```

`infer.py` and the benchmark prediction scripts share the same loader interface: a method name selects a loader in [`loaders/`](loaders/), which loads and patches its implementation in [`methods/`](methods/).  The benchmark-specific evaluation scripts then score the generated predictions.

## `benchmarks/`: Unified Evaluation Framework

Each unified benchmark follows the same organization: a prediction script, an evaluation script, and method-specific runner scripts. Predictions are stored under each benchmark's `outputs/` directory, organized by model, method, and parameter signature; the evaluation script reads these predictions and produces scores. Default method parameters are defined in the YAML files under [`method_configs/`](method_configs/), and individual options can be overridden with `--set KEY=VALUE`. The model, dataset, context-length, and budget settings in `method_script/` or `method_scripts/` follow the paper's experimental configurations. To change an experiment, edit the arrays at the beginning of the relevant method script.

Run every command in this section from the repository root; none of the benchmark commands requires `cd`.

### LongBench

[`benchmarks/longbench/`](benchmarks/longbench/) contains the unified LongBench implementation:

- `config/` stores dataset prompts and maximum generation lengths. LongBench is loaded through its original Hugging Face loading logic.
- [`longbench_pred.py`](benchmarks/longbench/longbench_pred.py) runs prediction and accepts `--datasets`, `--model`, `--method`, `--budget`, and `--experiment`.
- [`longbench_eval.py`](benchmarks/longbench/longbench_eval.py) reads prediction files and evaluates them.
- Predictions are stored at `outputs/<model>/<experiment>/<method>/<dataset>/<parameter-signature>.jsonl`; evaluation results are written to the corresponding method directory.

To change evaluated datasets, models, or budgets, edit the `models`, `datasets`, and `budgets` arrays in the appropriate script under [`method_script/`](benchmarks/longbench/method_script/). `<method>-overview.sh` runs the paper's overview experiment, while `<method>-budget.sh` runs its budget ablation. Each script runs prediction followed by evaluation. For example:

```bash
bash benchmarks/longbench/method_script/snapkv-overview.sh
bash benchmarks/longbench/method_script/snapkv-budget.sh
```

To run one LongBench prediction and evaluate it directly, use matching model, method, budget, dataset, and experiment arguments in both commands:

```bash
python benchmarks/longbench/longbench_pred.py \
  --method snapkv --model llama-3.1-8b --budget 1024 \
  --datasets qasper --experiment overview
python benchmarks/longbench/longbench_eval.py \
  --method snapkv --model llama-3.1-8b --budget 1024 \
  --datasets qasper --experiment overview
```

### RULER

[`benchmarks/ruler/`](benchmarks/ruler/) is a self-contained RULER synthetic-task framework. [`synthetic.yaml`](benchmarks/ruler/synthetic.yaml) defines the synthetic tasks. Data are stored at `benchmark_root/<model>/synthetic/<length>/data` and shared by all methods for the same model. Predictions and their `summary.csv` files are stored at `benchmark_root_pred/synthetic/<model>/<method>/<parameter-signature>/<length>/`; for example, `benchmark_root_pred/synthetic/llama-3.1-8b/streaming/budget-1024/65536/`. Set `RULER_BENCHMARK_ROOT` or `RULER_BENCHMARK_PRED_ROOT` to override the data or prediction root, respectively.

To customize a RULER run, edit the `models`, `tasks`, `lengths`, and `budgets` arrays in the relevant script under [`method_scripts/`](benchmarks/ruler/method_scripts/). `overview` covers the synthetic tasks and lengths reported in the paper, while `budget` runs the budget study. These names identify the preset script only; the generic `run_experiment.sh` runner accepts a method followed by option flags, then prepares data, predicts, and evaluates in sequence.

```bash
bash benchmarks/ruler/method_scripts/snapkv-overview.sh
bash benchmarks/ruler/method_scripts/snapkv-budget.sh
```

For a direct RULER run, invoke the generic runner with a method, models, tasks, lengths, and budgets:

```bash
bash benchmarks/ruler/run_experiment.sh snapkv \
  --models llama-3.1-8b \
  --tasks niah_single_1 \
  --lengths 4096 \
  --budgets 1024
```

### LongBenchV2

[`benchmarks/longbenchv2/`](benchmarks/longbenchv2/) provides the LongBenchV2 evaluation:

- [`filtered_longbench_v2_64k-192k.jsonl`](benchmarks/longbenchv2/filtered_longbench_v2_64k-192k.jsonl) contains the 64K--192K subset used in this repository. Use [`filter_64k_192k.sh`](benchmarks/longbenchv2/filter_64k_192k.sh) to recreate the filtered subset.
- [`longbenchv2_pred.py`](benchmarks/longbenchv2/longbenchv2_pred.py) runs prediction, and [`longbenchv2_eval.py`](benchmarks/longbenchv2/longbenchv2_eval.py) aggregates results.
- Predictions are written to `outputs/<model>/<method>/longbenchv2/<parameter-signature>.jsonl`, with evaluation results stored alongside them.

This benchmark currently provides overview experiments. Edit the `models` or `budgets` arrays in the relevant [`method_script/`](benchmarks/longbenchv2/method_script/) file, then run the script. The default settings match the paper's LongBenchV2 experiments.

```bash
bash benchmarks/longbenchv2/method_script/snapkv-overview.sh
```

To run one prediction and evaluate it directly:

```bash
python benchmarks/longbenchv2/longbenchv2_pred.py \
  --method snapkv --model qwen-2.5-7b-1m --budget 4096
python benchmarks/longbenchv2/longbenchv2_eval.py \
  --method snapkv --model qwen-2.5-7b-1m --budget 4096
```

### GSM8K

[`benchmarks/gsm8k/`](benchmarks/gsm8k/) contains GSM8K inference and evaluation:

- The bundled dataset is [`data/gsm8k_test.jsonl`](benchmarks/gsm8k/data/gsm8k_test.jsonl).
- [`gsm8k_pred.py`](benchmarks/gsm8k/gsm8k_pred.py) predicts with a fixed few-shot prompt; [`gsm8k_eval.py`](benchmarks/gsm8k/gsm8k_eval.py) extracts final answers and computes accuracy.
- Predictions are written to `outputs/<model>/<method>/gsm8k/<parameter-signature>.jsonl`, with evaluation results stored in the same directory.

This benchmark currently provides overview experiments. Change the `models` and `budgets` arrays in the selected script under [`method_script/`](benchmarks/gsm8k/method_script/), then run that script. Its default parameters follow the paper's GSM8K setting.

```bash
bash benchmarks/gsm8k/method_script/snapkv-overview.sh
```

To run one prediction and evaluate it directly:

```bash
python benchmarks/gsm8k/gsm8k_pred.py \
  --method snapkv --model ds-qwen-1.5b --budget 360
python benchmarks/gsm8k/gsm8k_eval.py \
  --method snapkv --model ds-qwen-1.5b --budget 360
```

### Efficiency, Memory, and Recall Data

[`benchmarks/myinput.txt`](benchmarks/myinput.txt) is a long input sampled from LongBenchV2 and is used for efficiency and GPU-memory measurements. The relevant commands are documented in the experiment guide for each method.

[`benchmarks/Longbench_recall/`](benchmarks/Longbench_recall/) contains LongBench recall data, currently for `narrativeqa` and `qasper`. [`benchmarks/Ruler_recall/`](benchmarks/Ruler_recall/) contains RULER recall data organized by model and length. These datasets support KV-cache recall, Recall@100, and attention-coverage analysis; see the relevant method documentation for scripts and parameters.

## `methods/`: Implementations and Paper Experiments

[`methods/`](methods/) contains the concrete implementations of all methods. Before using a method-specific implementation, enter its directory with `cd methods/<method>/`. Except for environment installation, dataset preparation, and model pretraining, each method's experiment scripts, workflows, and parameter explanations used in the paper are documented in its `EXPERIMENTS.md`.

This includes not only the main accuracy experiments, but also the efficiency, peak GPU memory, sparse-attention quality metrics such as Recall and Attention Coverage, and runtime breakdown experiments. The corresponding scripts and configurations are organized within each method directory, and detailed instructions for running these experiments are provided in `EXPERIMENTS.md`.

The experimental organization, scripts, and parameter descriptions for the eviction methods, including StreamingLLM, SnapKV, H2O, and Keyformer, are collected in [`methods/EVICT_EXPERIMENTS.md`](methods/EVICT_EXPERIMENTS.md).

The experimental organization, scripts, and parameter descriptions for the heuristic retrieval methods, including SparQ, Quest, FlexPrefill, and XAttention, are collected in [`methods/Heuristic_EXPERIMENTS.md`](methods/Heuristic_EXPERIMENTS.md).

Each method directory also retains its original `README.md`. We have not modified these method-specific READMEs; they are provided for reference only.


## Acknowledgements and Upstream Repositories

We sincerely thank the authors and maintainers of the open-source projects that made this comparative implementation and evaluation possible.

| Method | Repository |
| --- | --- |
| H2O | [FMInference/H2O](https://github.com/FMInference/H2O) |
| StreamingLLM | [mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm) |
| Keyformer | [d-matrix-ai/keyformer-llm](https://github.com/d-matrix-ai/keyformer-llm) |
| SnapKV | [FasterDecoding/SnapKV](https://github.com/FasterDecoding/SnapKV) |
| SparQ | [graphcore-research/llm-inference-research](https://github.com/graphcore-research/llm-inference-research/tree/2024-05-sparq) |
| Quest | [mit-han-lab/quest](https://github.com/mit-han-lab/quest) |
| FlexPrefill | [ByteDance-Seed/FlexPrefill](https://github.com/ByteDance-Seed/FlexPrefill) |
| XAttention | [mit-han-lab/x-attention](https://github.com/mit-han-lab/x-attention) |
| ClusterKV | [sjtu-zhao-lab/ClusterKV](https://github.com/sjtu-zhao-lab/ClusterKV) |
| PQCache | [HugoZHL/PQCache](https://github.com/HugoZHL/PQCache) |
| MagicPIG | [Infini-AI-Lab/MagicPiG](https://github.com/Infini-AI-Lab/MagicPiG) |
| RetroInfer | [microsoft/RetrievalAttention](https://github.com/microsoft/RetrievalAttention) |
| CakeKV | [antgroup/cakekv](https://github.com/antgroup/cakekv) |
| PyramidKV | [IsaacRe/PyramidKV](https://github.com/IsaacRe/PyramidKV) |
| AdaKV | [FYYFU/HeadKV](https://github.com/FYYFU/HeadKV) |
| HeadKV | [FYYFU/HeadKV](https://github.com/FYYFU/HeadKV) |
| DuoAttention | [mit-han-lab/duo-attention](https://github.com/mit-han-lab/duo-attention) |
| MInference | [microsoft/MInference](https://github.com/microsoft/MInference) |
