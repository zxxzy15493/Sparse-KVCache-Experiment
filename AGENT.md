# Project Conventions

## Submission Scope

- This repository is prepared for SIGMOD double-blind review. Do not add author,
  organization, machine, or private-path information to committed files.
- Keep only code and small data needed to read, run, and reproduce the reported
  accuracy, efficiency, recall, and component experiments. Do not remove a file
  when its relevance is uncertain; list it for review instead.
- LongBench and LongBenchV2 download through their original Hugging Face
  `load_dataset` calls. Do not add `--data-root` for either benchmark.
- GSM8K uses the bundled file at `benchmarks/gsm8k/data/gsm8k_test.jsonl`.
- Put local absolute model-path overrides only in `benchmarks/local_paths.json`.

## Unified Models And Methods

- Use these exact model names everywhere:
  `llama-3.1-8b`, `qwen-2.5-7b`, `qwen-2.5-7b-1m`,
  `glm-4-9b-1m`, and `ds-qwen-1.5b`.
- The default model for every unified entrypoint is `llama-3.1-8b`.
- Do not introduce method-specific model-name aliases or a ClusterKV-only model
  map. Method loaders receive the unified CLI model name directly.
- Supported methods are: `full_attention`, `topk`, `topk32`, `topp`, `topp32`,
  `pqcache`, and `clusterkv`.
- `full_attention` is PQCache's `original` compressor implementation.
- `topp` and `topp32` use `fixthreshold: 0.9` by default.
- Method defaults belong in one YAML file per method under `method_configs/`.
  `--set KEY=VALUE` must affect the actual method arguments as well as the output
  signature.
- Each method YAML contains its default user-facing `budget`. An explicit
  `--budget` overrides it and is routed to PQCache's `budget` or ClusterKV's
  `token_budget` without changing the output-signature convention.
- PQCache-family YAML files also set `fixbudget: true`. The fixed token budget is
  active only when this flag and `budget` are both set. `fixbudget` is immutable
  and cannot be changed through `--set`.

## Loader Design

- `loaders/full_attention_load.py`, `loaders/pqcache_load.py`, and
  `loaders/clusterkv_load.py` each contain a complete, visible
  `load_model_and_tokenizer` implementation.
- `loaders/__init__.py` owns the shared `load_model_and_tokenizer(args)` dispatch.
  Register every new unified method in `METHOD_LOADERS`; prediction entrypoints and
  `infer.py` must import this function instead of duplicating method branches.
- Do not forward or alias-import `load_model_and_tokenizer` from legacy scripts
  such as `vq_pred.py`, `fullpred.py`, or ClusterKV `pred.py`.
- Imports of model classes, attention patches, and method-local dependencies inside
  a loader are expected.
- PQCache must set `MAX_CPU_IN_USE=16`. Its PQ cache capacity follows the selected
  model context length.
- ClusterKV uses `nlist = max(initial_nlist, ceil(input_length / 80))` for every
  sample and resets its cache after each sample.
- After every sample and after each model run, call `torch.cuda.empty_cache()`.

## Entrypoints And Environment

- Run experiment entrypoints directly, for example:
  `python benchmarks/longbench/longbench_pred.py`.
- Direct experiment entrypoints add the repository root to `sys.path` so their
  absolute `benchmarks.*` imports work. Do not use `GPU_ID`.
  Use the original `CUDA_VISIBLE_DEVICES` variable instead: Python prediction
  entrypoints and shell runners default it to `0`, while a caller-provided value
  remains unchanged. PQCache patches must also handle an absent value safely.
- Shell runners export:
  `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`.
- LongBench's local files are under `benchmarks/longbench/`, including the copied,
  identical original config files in `benchmarks/longbench/config/`.
- RULER is self-contained in `benchmarks/ruler/`. Do not add `config_models.sh` or
  `config_tasks.sh`. Every `method_scripts/<method>-<experiment>.sh` script must
  visibly define its experiment models, tasks, lengths, and budgets, and pass them
  to the generic `run_experiment.sh` executor. The `overview` and `budget` suffixes
  label the preset scripts; they are not positional arguments to that executor.
- RULER uses only local Hugging Face inference through `call_api.py` and
  `model_wrappers.py`; do not add VLLM, remote API, or serving backends.
- RULER generated data and predictions belong below
  `benchmarks/ruler/benchmark_root/` (or `RULER_BENCHMARK_ROOT`). Prediction JSONL
  must omit the `input` field, and evaluation must create `summary.csv`.
- Construct RULER data once per model and place it at
  `benchmark_root/<model-data-name>/synthetic/<length>/data`, shared by that
  model's methods. Only `pred` directories use
  `<model>-<method>-budget-<budget>` prefixes.

## Benchmark Behavior

- LongBench prediction/evaluation format follows the ClusterKV experiment layout:
  `benchmarks/longbench/outputs/<model>/<experiment>/<method>/<dataset>/<parameter-signature>.jsonl`.
- GSM8K and LongBenchV2 do not add an experiment directory:
  `outputs/<model>/<method>/<benchmark>/<parameter-signature>.jsonl`.
- `--runtest` is disabled by default. When enabled it uses the first two rows of
  each selected dataset.
- `benchmarks/longbench/runtest.sh` is the only LongBench runtest entrypoint. It
  runs `qasper` and `narrativeqa`, each limited to its first two rows, under the
  `runtest` experiment and evaluates the generated files.
- LongBench's `method_script/` directory contains one direct shell script for each
  method/experiment pair: `<method>-overview.sh` and `<method>-budget.sh`. Every
  script lists its model, budget, and dataset arrays explicitly, then runs
  `longbench_pred.py` and `longbench_eval.py` for each model/budget pair.
- `methods/pqcache/run_longbench_overview.sh` reproduces the four original
  PQCache compressors using only the internal `vq_pred.py` and `eval.py` code.
  Its four per-method scripts are `run_longbench_original.sh`,
  `run_longbench_no_drop_lb.sh`, `run_longbench_no_drop_lb_topp.sh`, and
  `run_longbench_pqcache.sh`; the data source is `THUDM/LongBench`.
- `methods/pqcache/run_longbench.sh` is parameter-only: it has no method/model
  routing or `case` mapping. Each method script explicitly sets and passes its
  compressor, experiment name, sink/recent sizes, budget, and model names to
  this runner. Do not repeat common runner defaults in multiple method scripts.
- `methods/pqcache/run_budget_method.sh` follows the same parameter-only design
  but invokes `vq_pred_budget.py`. Its budget scripts exclude `original`:
  `no_drop_lb` and `pqcache` sweep 128, 256, 512, and 1024; `no_drop_lb_topp`
  sweeps 0.8, 0.85, 0.9, and 0.95 with budget 1024. They run only `llama-3.1`
  and `qwen-2.5-7b`.
- LongBench evaluation selects its prediction file from `--method`, `--budget`,
  and matching `--set` options. Do not add a free-form filename or `--run`
  selector; TopP/TopP32 use their YAML default budget and must pass the same
  `fixthreshold` value used for prediction.
- GSM8K and LongBenchV2 each have only `<method>-overview.sh` scripts under their
  own `method_script/` directories. Their overview budgets are 360 and 4096,
  respectively; neither benchmark has a budget experiment script. LongBenchV2
  runs only `qwen-2.5-7b-1m`, while GSM8K runs only `ds-qwen-1.5b`.
- TopP and TopP32 scripts use `fixthreshold=0.9` by default. Their LongBench
  overview and budget scripts do not define or pass a budget; they use the method
  YAML default. The budget experiment sweeps `fixthreshold` over `0.8`, `0.85`,
  `0.9`, and `0.95`. GSM8K and LongBenchV2 retain their benchmark-wide fixed
  budgets of 360 and 4096; TopP and TopP32 use `fixthreshold=0.9` for both.
- The overview run is budget 1024 on:
  `narrativeqa`, `qasper`, `2wikimqa`, `musique`, `gov_report`, `multi_news`,
  `triviaqa`, `samsum`, `passage_count`, `passage_retrieval_en`, `lcc`, and
  `repobench-p`.
- The budget run uses 128, 256, 512, and 1024 on `narrativeqa`, `qasper`, `trec`,
  and `lcc`.
- RULER overview runs every synthetic task at lengths 4096, 8192, 16384, 32768,
  and 65536 with budget 1024. Its budget run uses 128, 384, 1024, and 4096
  at length 65536 on `niah_single_3`, `vt`, `cwe`, `fwe`, and `qa_1`. Both only
  run `llama-3.1-8b` and `qwen-2.5-7b-1m`; the fixed default is 50 samples per
  task. The internal PQCache RULER scripts live in
  `methods/pqcache/RULER/scripts/`: `run_ruler_method.sh` invokes the internal
  PQCache prediction/evaluation code, `run_ruler_overview.sh` runs original,
  no-drop, TopP, and PQCache, and `run_ruler_budget.sh` excludes original.
  Keep `config_models.sh` and `config_tasks.sh` in that directory: the PQCache
  RULER recall scripts source both files. The prohibition on these config files
  applies only to the unified `benchmarks/ruler/` runner.
  Their data root is exactly `benchmarks/ruler/benchmark_root`, while their
  predictions stay below `methods/pqcache/RULER/scripts/benchmark_root`.
  The scripts use one `MODEL_NAME` for model/data/result naming and iterate
  `llama-3.1-8b` and `qwen-2.5-7b-1m`; `MODEL_PATH` is only the Hugging Face
  load location. PQCache uses 8 subvectors and 4 bits at length 65536. Do not
  add a per-script or CLI sample-count override.
- Every RULER method script directly lists and passes its models, tasks, lengths,
  and budgets to `run_experiment.sh`. The runner takes only the method as its
  positional argument, followed by option flags. Keep it generic; do not hide those
  experiment settings in it.
- Apply `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` limits only to the exact
  `pqcache` method. Do not set them for `full_attention`, TopK/TopP variants, or
  `clusterkv`.

## Correctness Checks

- TopP/TopP32 must retain the token that crosses the cumulative-probability
  threshold.
- Before finishing changes, run `py_compile`, `bash -n`, `git diff --check`, and a
  qasper/narrativeqa LongBench runtest with evaluation when model resources allow.
