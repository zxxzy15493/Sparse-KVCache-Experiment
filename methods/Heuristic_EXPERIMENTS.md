# Selection Experiment Run Guide

This document summarizes the experiment entry points for the four methods under `methods/sparq`, `methods/quest`, `methods/flexPrefill`, and `methods/x-attention`. Before running a script, enter the corresponding directory first; the commands below include the required `cd` commands explicitly.

## Method Directories

The implementation directories for the four methods are:

| Method | Method directory | Unified loader | Default config |
| --- | --- | --- | --- |
| SparQ | `methods/sparq` | `loaders/sparq_load.py` | `method_configs/sparq.yaml` |
| Quest | `methods/quest` | `loaders/quest_load.py` | `method_configs/quest.yaml` |
| FlexPrefill | `methods/flexPrefill` | `loaders/flexprefill_load.py` | `method_configs/flexprefill.yaml` |
| XAttention | `methods/x-attention` | `loaders/xattention_load.py` | `method_configs/xattention.yaml` |

Quest and SparQ are token-budget methods, using budgets such as 128, 256, 512, 1024, and 4096 in the experiments. FlexPrefill and XAttention are threshold-based methods, using ratio thresholds such as 0.8, 0.85, 0.9, and 0.95. When this document writes `1024/0.9`, it means Quest/SparQ use budget 1024, while FlexPrefill/XAttention use threshold 0.9.

## 1. accuracy_overview (ACC)

This experiment measures overall accuracy on LongBench and RULER.

LongBench uses `benchmarks/longbench`, with scripts located in `benchmarks/longbench/method_script`. The paper setting uses the LongBench dataset, models `llama-3.1-8b`, `qwen-2.5-7b`, and `glm-4-9b-1m`, and budget `1024/0.9`.

```bash
cd benchmarks/longbench/method_script
bash quest-overview.sh
bash sparq-overview.sh
bash flexprefill-overview.sh
bash xattention-overview.sh
```

GLM-specific LongBench runs are also available as method-local scripts. These scripts target `glm-4-9b-chat-1m` from `zai-org/glm-4-9b-chat-1m`. Quest and SparQ use token budget 1024; XAttention uses `xattn`; FlexPrefill uses the default settings in its GLM LongBench script.

```bash
cd methods/sparq
bash buchong_scripts/ACC/glm/longbench.sh

cd methods/quest
bash buchong_scripts/ACC/glm/longbench.sh

cd methods/flexPrefill
bash buchong_scripts/ACC/glm/longbench.sh

cd methods/x-attention
bash buchong_scripts/ACC/glm/longbench.sh
```

RULER uses `benchmarks/ruler`, with scripts located in `benchmarks/ruler/method_scripts`. The paper setting uses sequence lengths 4k, 8k, 16k, 32k, and 64k, models `llama-3.1-8b` and `qwen-2.5-7b-1m`, and budget `1024/0.9`.

```bash
cd benchmarks/ruler/method_scripts
bash quest-overview.sh
bash sparq-overview.sh
bash flexprefill-overview.sh
bash xattention-overview.sh
```

## 2. accuracy_budget (ACC_budget)

This experiment measures how accuracy changes under different budgets.

The LongBench entry point remains `benchmarks/longbench/method_script`. The paper setting uses models `llama-3.1-8b` and `qwen-2.5-7b`. Quest/SparQ use budgets 128, 256, 512, and 1024. FlexPrefill/XAttention use thresholds 0.8, 0.85, 0.9, and 0.95.

```bash
cd benchmarks/longbench/method_script
bash quest-budget.sh
bash sparq-budget.sh
bash flexprefill-budget.sh
bash xattention-budget.sh
```

The RULER entry point is `benchmarks/ruler/method_scripts`. The paper setting uses sequence length 64k and models `llama-3.1-8b` and `qwen-2.5-7b-1m`. Quest/SparQ use budgets 128, 384, 1024, and 4096. FlexPrefill/XAttention use thresholds 0.8, 0.85, 0.9, and 0.95.

```bash
cd benchmarks/ruler/method_scripts
bash quest-budget.sh
bash sparq-budget.sh
bash flexprefill-budget.sh
bash xattention-budget.sh
```

## 3. efficiency_overview

This experiment measures inference efficiency at different input lengths. It uses models `llama-3.1-8b` and `qwen-2.5-7b-1m`, input lengths from 4k to 128k, output length 32, and budget `1024/0.9`.

The scripts are under `buchong_scripts/efficiency` in each method directory:

| Method | Llama script | Qwen script |
| --- | --- | --- |
| SparQ | `buchong_scripts/efficiency/latency.sh` | `buchong_scripts/efficiency/qwen_latency.sh` |
| Quest | `buchong_scripts/efficiency/latency.sh` | `buchong_scripts/efficiency/qwen_latency.sh` |
| FlexPrefill | `buchong_scripts/efficiency/latency.sh` | `buchong_scripts/efficiency/qwen_latency.sh` |
| XAttention | `buchong_scripts/efficiency/latency.sh` | `buchong_scripts/efficiency/qwen_latency.sh` |

Run commands:

```bash
cd methods/sparq
bash buchong_scripts/efficiency/latency.sh
bash buchong_scripts/efficiency/qwen_latency.sh

cd methods/quest
bash buchong_scripts/efficiency/latency.sh
bash buchong_scripts/efficiency/qwen_latency.sh

cd methods/flexPrefill
bash buchong_scripts/efficiency/latency.sh
bash buchong_scripts/efficiency/qwen_latency.sh

cd methods/x-attention
bash buchong_scripts/efficiency/latency.sh
bash buchong_scripts/efficiency/qwen_latency.sh
```

## 4. efficiency_budget

This experiment measures efficiency under different budgets. It uses models `llama-3.1-8b` and `qwen-2.5-7b-1m`, input lengths 4k and 64k, and output length 32. For 4k, it uses 128, 256, 512, 1024 or 0.8, 0.85, 0.9, 0.95. For 64k, it uses 128, 384, 1024, 4096 or 0.8, 0.85, 0.9, 0.95.

The scripts are under `buchong_scripts/efficiency` in each method directory:

| Method | Llama script | Qwen script |
| --- | --- | --- |
| SparQ | `buchong_scripts/efficiency/budget_latency.sh` | `buchong_scripts/efficiency/budget_qwen_latency.sh` |
| Quest | `buchong_scripts/efficiency/budget_latency.sh` | `buchong_scripts/efficiency/budget_qwen_latency.sh` |
| FlexPrefill | `buchong_scripts/efficiency/budget_latency.sh` | `buchong_scripts/efficiency/budget_qwen_latency.sh` |
| XAttention | `buchong_scripts/efficiency/budget_latency.sh` | `buchong_scripts/efficiency/budget_qwen_latency.sh` |

Run commands:

```bash
cd methods/sparq
bash buchong_scripts/efficiency/budget_latency.sh
bash buchong_scripts/efficiency/budget_qwen_latency.sh

cd methods/quest
bash buchong_scripts/efficiency/budget_latency.sh
bash buchong_scripts/efficiency/budget_qwen_latency.sh

cd methods/flexPrefill
bash buchong_scripts/efficiency/budget_latency.sh
bash buchong_scripts/efficiency/budget_qwen_latency.sh

cd methods/x-attention
bash buchong_scripts/efficiency/budget_latency.sh
bash buchong_scripts/efficiency/budget_qwen_latency.sh
```

## 5. vram

This experiment measures peak VRAM usage. It uses models `llama-3.1-8b` and `qwen-2.5-7b-1m`. One group tests budgets 64 and 512 at input length 1k, with output lengths 2 and 4k. Another group tests input lengths from 4k to 128k with output length 2 and budget `1024/0.9`.

The scripts are under `buchong_scripts/VRAM` in each method directory:

| Method | Llama script | Qwen script |
| --- | --- | --- |
| SparQ | `buchong_scripts/VRAM/VRAM.sh` | `buchong_scripts/VRAM/qwenVRAM.sh` |
| Quest | `buchong_scripts/VRAM/llamaVRAM.sh` | `buchong_scripts/VRAM/qwenVRAM.sh` |
| FlexPrefill | `buchong_scripts/VRAM/llamaVRAM.sh` | `buchong_scripts/VRAM/qwenVRAM.sh` |
| XAttention | `buchong_scripts/VRAM/VRAM.sh` | `buchong_scripts/VRAM/qwenVRAM.sh` |

Run commands:

```bash
cd methods/sparq
bash buchong_scripts/VRAM/VRAM.sh
bash buchong_scripts/VRAM/qwenVRAM.sh

cd methods/quest
bash buchong_scripts/VRAM/llamaVRAM.sh
bash buchong_scripts/VRAM/qwenVRAM.sh

cd methods/flexPrefill
bash buchong_scripts/VRAM/llamaVRAM.sh
bash buchong_scripts/VRAM/qwenVRAM.sh

cd methods/x-attention
bash buchong_scripts/VRAM/VRAM.sh
bash buchong_scripts/VRAM/qwenVRAM.sh
```

## 6. recall

This experiment measures how well the selected KV/cache tokens cover the top tokens from full attention. LongBench uses models `llama-3.1-8b` and `qwen-2.5-7b`, with budgets 128, 256, 512, 1024 or thresholds 0.8, 0.85, 0.9, 0.95. RULER uses sequence length 64k and models `llama-3.1-8b` and `qwen-2.5-7b-1m`, with budgets 128, 384, 1024, 4096 or thresholds 0.8, 0.85, 0.9, 0.95.

The recall scripts for SparQ, Quest, and XAttention are split by model under the `llama` and `qwen` subdirectories:

```bash
cd methods/sparq
bash buchong_scripts/recall/llama/longbench.sh
bash buchong_scripts/recall/qwen/longbench.sh
bash buchong_scripts/recall/llama/ruler.sh
bash buchong_scripts/recall/qwen/ruler.sh

cd methods/quest
bash buchong_scripts/recall/llama/longbench.sh
bash buchong_scripts/recall/qwen/longbench.sh
bash buchong_scripts/recall/llama/ruler.sh
bash buchong_scripts/recall/qwen/ruler.sh

cd methods/x-attention
bash buchong_scripts/recall/llama/longbench.sh
bash buchong_scripts/recall/qwen/longbench.sh
bash buchong_scripts/recall/llama/ruler.sh
bash buchong_scripts/recall/qwen/ruler.sh
```

FlexPrefill's recall scripts are in one directory:

```bash
cd methods/flexPrefill
bash buchong_scripts/recall/longbench.sh
bash buchong_scripts/recall/ruler.sh
```

## 7. longbenchv2

This experiment measures multiple-choice accuracy on LongBenchV2. The paper setting uses `qwen-2.5-7b-1m`, with budget 4096 or threshold 0.9. The unified entry point is `benchmarks/longbenchv2/method_script`; the prediction and evaluation scripts are `benchmarks/longbenchv2/longbenchv2_pred.py` and `benchmarks/longbenchv2/longbenchv2_eval.py`.

```bash
cd benchmarks/longbenchv2/method_script
bash quest-overview.sh
bash sparq-overview.sh
bash flexprefill-overview.sh
bash xattention-overview.sh
```

Independent LongBenchV2 scripts are also kept under each method directory:

| Method | Directory | Run script |
| --- | --- | --- |
| SparQ | `methods/sparq/longbenchV2` | `run.sh` |
| Quest | `methods/quest/longbenchV2` | `run.sh` |
| FlexPrefill | `methods/flexPrefill/longbenchV2` | `run.sh` |
| XAttention | `methods/x-attention/longbenchV2` | `run.sh` |

To run these method-local scripts, enter the method directory first:

```bash
cd methods/sparq
bash longbenchV2/run.sh

cd methods/quest
bash longbenchV2/run.sh

cd methods/flexPrefill
bash longbenchV2/run.sh

cd methods/x-attention
bash longbenchV2/run.sh
```

## 8. gsm8k

This experiment measures GSM8K mathematical reasoning accuracy. The paper setting uses `ds-qwen-1.5b`. Quest/SparQ use budget 360, while FlexPrefill/XAttention use the threshold settings in their corresponding scripts. The unified entry point is `benchmarks/gsm8k/method_script`; the prediction and evaluation scripts are `benchmarks/gsm8k/gsm8k_pred.py` and `benchmarks/gsm8k/gsm8k_eval.py`.

```bash
cd benchmarks/gsm8k/method_script
bash quest-overview.sh
bash sparq-overview.sh
bash flexprefill-overview.sh
bash xattention-overview.sh
```

Independent GSM8K scripts are also kept under each method directory:

```bash
cd methods/sparq
bash gsm8k/gsm.sh

cd methods/quest
bash gsm8k/gsm.sh

cd methods/flexPrefill
bash gsm8k/gsm.sh

cd methods/x-attention
bash gsm8k/gsm.sh
```

## Notes

1. Some scripts under `benchmarks/longbench` currently include extra models, such as `qwen-2.5-7b-1m` or `ds-qwen-1.5b`. If you only want to reproduce the settings in the experiment tables, check the scripts before running and narrow the `models` list at the top according to the model list in this document.
2. The scripts under `benchmarks/ruler` are dispatched through `run_experiment.sh`; tasks, lengths, models, and budgets are defined at the top of the corresponding `*-overview.sh` or `*-budget.sh` files.
3. The efficiency, VRAM, and recall scripts under `methods/*/buchong_scripts` usually `cd` into the method root inside the script. This document still recommends manually entering the method directory first, so relative paths and output directories are easier to inspect.
4. LongBenchV2 uses the data file `benchmarks/longbenchv2/filtered_longbench_v2_64k-192k.jsonl`.
