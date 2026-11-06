"""Numerical and dispatch regressions for memory-bounded threshold profiling."""

import math
import os
import subprocess
import weakref
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from accelerate import dispatch_model
from accelerate.hooks import ModelHook, add_hook_to_module
from transformers import BatchEncoding, Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from threshold.train_scripts import train_threshold as profiler
from xattn.src.utils import create_causal_mask


def dense_block_sums(query, key, block_size, causal=True):
  """Independent full-matrix reference, used only with tiny inputs."""
  scores = (query @ (key.transpose(-1, -2) / math.sqrt(query.shape[-1]))).float()
  if causal:
    future = torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool).triu(1)
    scores = scores.masked_fill(future, float("-inf"))
  probabilities = scores.softmax(-1).to(query.dtype)
  length = query.shape[-2]
  padding = (-length) % block_size
  probabilities = F.pad(probabilities, (0, padding, 0, padding))
  blocks = (length + padding) // block_size
  return probabilities.reshape(
    query.shape[0], query.shape[1], blocks, block_size, blocks, block_size,
  ).sum(-1).sum(-2)


def reference_threshold(dense, estimate, p):
  """Select blocks row by row rather than using the profiler's vectorized ranking."""
  result = torch.empty(dense.shape[:2])
  for batch in range(dense.shape[0]):
    for head in range(dense.shape[1]):
      numerator = 0.0
      for row, estimated_row in zip(dense[batch, head], estimate[batch, head]):
        indices = sorted(range(row.numel()), key=lambda i: float(row[i]), reverse=True)
        accumulated = 0.0
        chosen = []
        target = p * float(row.sum())
        for index in indices:
          chosen.append(index)
          accumulated += float(row[index])
          if accumulated >= target:
            break
        minimum = min(float(estimated_row[index]) for index in chosen)
        numerator += float(estimated_row[estimated_row >= minimum].sum())
      result[batch, head] = numerator / (float(estimate[batch, head].sum()) + 1e-8)
  return result


def install_cpu_attention(monkeypatch, calls=None):
  def estimate(query, key, block_size, stride, causal, **kwargs):
    if calls is not None:
      calls.append((query.shape[1], kwargs["chunk_size"], query.device))
    # Different scores from the exact reference, independent across heads.
    values = dense_block_sums(query * 0.7, key + 0.2, block_size, causal) / stride
    return values, torch.ones_like(values, dtype=torch.bool)

  def flash(q, k, v, causal):
    return F.scaled_dot_product_attention(
      q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
      is_causal=causal, enable_gqa=True,
    ).transpose(1, 2)

  monkeypatch.setattr(profiler, "xattn_estimate", estimate)
  monkeypatch.setattr(profiler, "create_causal_mask", create_causal_mask)
  monkeypatch.setattr(profiler, "flash_attn_func", flash)


@pytest.mark.parametrize("length", [1, 7, 8, 11])
@pytest.mark.parametrize("causal", [True, False])
def test_block_statistics_match_dense_reference(monkeypatch, length, causal):
  monkeypatch.setattr(profiler, "create_causal_mask", create_causal_mask)
  torch.manual_seed(length)
  query = torch.randn(2, 3, length, 8)
  key = torch.randn_like(query)
  actual = profiler.chunk_prefill_to_attn_sum(query, key, block_size=4, causal=causal)
  expected = dense_block_sums(query, key, block_size=4, causal=causal)
  torch.testing.assert_close(actual, expected)
  assert torch.isfinite(actual).all()


@pytest.mark.parametrize("statistic", ["exact", "estimate"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_block_statistics_cannot_generate_thresholds(monkeypatch, statistic, value):
  install_cpu_attention(monkeypatch)
  query = torch.randn(1, 2, 8, 8)
  key = torch.randn_like(query)
  if statistic == "exact":
    compute = profiler.chunk_prefill_to_attn_sum

    def invalid_exact(*args, **kwargs):
      sums = compute(*args, **kwargs)
      sums[0, 0, 0, 0] = value
      return sums

    monkeypatch.setattr(profiler, "chunk_prefill_to_attn_sum", invalid_exact)
  else:
    estimate = profiler.xattn_estimate

    def invalid_estimate(*args, **kwargs):
      sums, mask = estimate(*args, **kwargs)
      sums[0, 0, 0, 0] = value
      return sums, mask

    monkeypatch.setattr(profiler, "xattn_estimate", invalid_estimate)
  with pytest.raises(RuntimeError, match="Non-finite.*refusing to generate a threshold"):
    profiler.profile_head_threshold(query, key, 4, 2, 128, True, 0.85)


@pytest.mark.parametrize("kv_heads,head_chunk_size", [(2, 1), (2, 4), (6, 4), (2, 0)])
@pytest.mark.parametrize("p", [0.8, 1.0])
@pytest.mark.parametrize("causal", [True, False])
def test_head_chunks_preserve_thresholds_and_gqa(monkeypatch, kv_heads, head_chunk_size, p, causal):
  calls = []
  install_cpu_attention(monkeypatch, calls)
  torch.manual_seed(17)
  query = torch.randn(1, 6, 11, 8)
  key = torch.randn(1, kv_heads, 11, 8)
  value = torch.randn_like(key)
  repeated_key = key.repeat_interleave(6 // kv_heads, dim=1)
  exact = dense_block_sums(query, repeated_key, 4, causal)
  estimate = dense_block_sums(query * 0.7, repeated_key + 0.2, 4, causal)
  expected = reference_threshold(exact, estimate, p)
  config = profiler.ProfileConfig(p, 2, causal, 4, head_chunk_size, 128)
  module = SimpleNamespace(profile_layer_idx=0, profile_config=config)
  actual_output = profiler.xattn_prefill_profile(module, query, key, value, 4, 2, causal=causal)
  torch.testing.assert_close(config.history_threshold[0], expected)
  torch.testing.assert_close(actual_output, F.scaled_dot_product_attention(
    query, key, value, is_causal=causal, enable_gqa=True,
  ))
  assert config.history_threshold[0].device.type == "cpu"
  limit = head_chunk_size or 6
  assert [call[0] for call in calls] == [min(limit, 6 - start) for start in range(0, 6, limit)]
  assert all(call[1] == 128 and call[2].type == "cpu" for call in calls)


def tiny_qwen3():
  config = Qwen3Config(
    vocab_size=64, hidden_size=32, intermediate_size=64,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    head_dim=8, max_position_embeddings=64,
  )
  config._attn_implementation = "eager"
  return Qwen3ForCausalLM(config).eval()


def tokenizer(text, **kwargs):
  return BatchEncoding({"input_ids": torch.tensor([[1, 2, 3, 4, int(text)]])})


@pytest.mark.parametrize("chunk_size", [0, 1, 4, 32])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_mlp_chunking_matches_dense_with_uneven_noncontiguous_inputs(chunk_size, batch_size):
  model = tiny_qwen3()
  mlp = model.model.layers[0].mlp
  hidden = torch.randn(batch_size, 22, model.config.hidden_size)[:, ::2]
  with torch.no_grad():
    expected = mlp(hidden)
    profiler.patch_mlp_for_profiling(mlp, chunk_size)
    actual = mlp(hidden)
  torch.testing.assert_close(actual, expected)


def test_mlp_chunking_preserves_thresholds_and_accelerate_hooks(monkeypatch):
  install_cpu_attention(monkeypatch)
  dense = tiny_qwen3()
  chunked = deepcopy(dense)
  hook_calls = []

  class CountingHook(ModelHook):
    def pre_forward(self, module, *args, **kwargs):
      hook_calls.append(module.profile_mlp_chunk_size)
      return args, kwargs

  wrappers = []
  for layer in chunked.model.layers:
    add_hook_to_module(layer.mlp, CountingHook())
    wrappers.append(layer.mlp.forward)

  thresholds, hidden_states = [], []
  for model, chunk_size in [(dense, 0), (chunked, 3)]:
    config = profiler.ProfileConfig(0.85, 2, True, 4, estimate_chunk_size=128, mlp_chunk_size=chunk_size)
    # Reapplying must keep the real original forward and avoid wrapper recursion.
    profiler.patch_model_for_profiling(model, config, "qwen3")
    profiler.patch_model_for_profiling(model, config, "qwen3")
    hook = model.model.register_forward_hook(
      lambda module, args, output: hidden_states.append(output.last_hidden_state.detach()))
    profiler.run_profile_on_texts(model, tokenizer, ["5"])
    hook.remove()
    thresholds.append(profiler.compute_final_threshold_from_history(config))
  torch.testing.assert_close(hidden_states[1], hidden_states[0])
  torch.testing.assert_close(thresholds[1], thresholds[0])
  assert hook_calls == [3, 3]
  assert all(layer.mlp.forward is wrapper for layer, wrapper in zip(chunked.model.layers, wrappers))


def test_negative_mlp_chunk_size_fails_early():
  with pytest.raises(ValueError, match="mlp_chunk_size must be non-negative"):
    profiler.ProfileConfig(0.85, 16, True, 128, mlp_chunk_size=-1)


def test_backbone_profiling_skips_logits_and_releases_outputs(monkeypatch):
  install_cpu_attention(monkeypatch)
  model = tiny_qwen3()
  config = profiler.ProfileConfig(0.9, 2, True, 4, head_chunk_size=1, estimate_chunk_size=128)
  profiler.patch_model_for_profiling(model, config, "qwen3")
  references = []

  def before_forward(module, args, kwargs):
    assert all(ref() is None for ref in references)
    assert kwargs["use_cache"] is False
    assert not torch.is_grad_enabled()

  def after_forward(module, args, output):
    references.append(weakref.ref(output))

  def forbidden_logits(*args, **kwargs):
    pytest.fail("Threshold profiling must not execute the LM head")

  monkeypatch.setattr(model.lm_head, "forward", forbidden_logits)
  model.model.register_forward_pre_hook(before_forward, with_kwargs=True)
  model.model.register_forward_hook(after_forward)
  profiler.run_profile_on_texts(model, tokenizer, ["5", "6"])
  assert len(references) == len(config.history_threshold) == 2
  threshold = profiler.compute_final_threshold_from_history(config)
  assert threshold.shape == (2, 4)
  assert torch.isfinite(threshold).all()
  torch.testing.assert_close(threshold, torch.stack(config.history_threshold).amax(0))


def test_profiling_preserves_accelerate_attention_hooks(monkeypatch):
  install_cpu_attention(monkeypatch)
  calls = []

  class CountingHook(ModelHook):
    def pre_forward(self, module, *args, **kwargs):
      calls.append(module.profile_layer_idx)
      return args, kwargs

  model = tiny_qwen3()
  for layer in model.model.layers:
    add_hook_to_module(layer.self_attn, CountingHook())
  config = profiler.ProfileConfig(0.9, 2, True, 4, estimate_chunk_size=128)
  profiler.patch_model_for_profiling(model, config, "qwen3")
  profiler.run_profile_on_texts(model, tokenizer, ["5"])
  assert calls == [0, 1]
  assert profiler.compute_final_threshold_from_history(config).shape == (2, 4)


def test_input_device_uses_embedding_execution_hook():
  embedding = torch.nn.Embedding(4, 4, device="meta")
  embedding._hf_hook = SimpleNamespace(execution_device=1)
  model = SimpleNamespace(get_input_embeddings=lambda: embedding, device=torch.device("cuda:0"))
  assert profiler.get_profile_input_device(model) == torch.device("cuda:1")


def test_memory_budgets_reserve_activations_on_each_visible_gpu(monkeypatch):
  gib = 1024**3
  monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
  monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
  monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: ((80, 40)[index] * gib, 80 * gib))
  assert profiler.resolve_max_memory("", 12, "balanced") == {0: 68 * gib, 1: 28 * gib}
  assert profiler.resolve_max_memory('{"0":"20GiB","1":1073741824}', 12, "balanced") == {
    0: 20 * gib, 1: gib,
  }
  assert profiler.resolve_max_memory("", 12, "cuda:0") is None
  with pytest.raises(ValueError, match="inferred device_map"):
    profiler.resolve_max_memory('{"0":"20GiB"}', 12, "cuda:0")
  with pytest.raises(ValueError, match="GPU 1"):
    profiler.resolve_max_memory("", 48, "balanced")


@pytest.mark.parametrize("value", ['[]', '{}', '{"cuda:0":"1GiB"}', '{"0":true}', '{"0":-1}'])
def test_invalid_memory_budgets_fail_early(value):
  with pytest.raises(ValueError):
    profiler.resolve_max_memory(value, 8, "balanced")


def test_empty_cache_visits_all_model_gpus(monkeypatch):
  devices, cleared = [], []
  monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
  monkeypatch.setattr(torch.cuda, "device", lambda device: devices.append(device) or nullcontext())
  monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleared.append(True))
  model = SimpleNamespace(hf_device_map={"embed": 0, "layer0": "cuda:0", "layer1": 2, "out": "cpu"})
  profiler.empty_model_cuda_cache(model)
  assert devices == [0, 2]
  assert len(cleared) == 2


@pytest.mark.parametrize("context", ["64k", "128k"])
def test_qwen3_launchers_forward_memory_options(tmp_path, context):
  script = Path(profiler.__file__).parent / f"run_qwen3_threshold_{context}.sh"
  texts = tmp_path / "texts.json"
  texts.write_text('["test"]', encoding="utf-8")
  env = dict(os.environ, CUDA_VISIBLE_DEVICES="2,3", DEVICE_MAP="balanced_low_0",
    MAX_MEMORY='{"0":"20GiB","1":"30GiB"}', HEAD_CHUNK_SIZE="2",
    ESTIMATE_CHUNK_SIZE="2048", MLP_CHUNK_SIZE="1024", RESERVE_MEMORY_GIB="12", RESUME_CHUNKS="false",
    TEXT_PATH=str(texts), OUTPUT_DIR=str(tmp_path / "output"), CHUNK_TEXTS="1")
  env.pop("P", None)
  # Intercept the interpreter to exercise shell quoting without loading a model.
  result = subprocess.run([
    "bash", "-c", 'python() { printf "%s\\n" "$@"; }; export -f python; bash "$1" 0.85',
    "threshold-launch-test", str(script),
  ], env=env, capture_output=True, text=True, check=True)
  arguments = result.stdout.splitlines()
  expected = {"--device_map": "balanced_low_0", "--max_memory": env["MAX_MEMORY"],
    "--head_chunk_size": "2", "--estimate_chunk_size": "2048", "--reserve_memory_gib": "12",
    "--mlp_chunk_size": "1024", "--resume_chunks": "false", "--p": "0.85"}
  for option, value in expected.items():
    assert arguments[arguments.index(option) + 1] == value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA memory measurements")
def test_cuda_mlp_chunking_reduces_peak_memory():
  config = Qwen3Config(hidden_size=256, intermediate_size=2048, num_attention_heads=4)
  mlp = Qwen3MLP(config).eval().to(device="cuda:0", dtype=torch.bfloat16)
  hidden = torch.randn(1, 32771, 256, device="cuda:0", dtype=torch.bfloat16)
  outputs, peaks = [], []
  with torch.no_grad():
    for chunk_size in [0, 4096]:
      profiler.patch_mlp_for_profiling(mlp, chunk_size)
      torch.cuda.synchronize(0)
      baseline = torch.cuda.memory_allocated(0)
      torch.cuda.reset_peak_memory_stats(0)
      outputs.append(mlp(hidden))
      torch.cuda.synchronize(0)
      peaks.append(torch.cuda.max_memory_allocated(0) - baseline)
  torch.testing.assert_close(outputs[1], outputs[0], rtol=5e-3, atol=5e-3)
  assert peaks[1] < peaks[0] / 2
  print(f"\nCUDA MLP peak increment (dense / chunked): "
        f"{peaks[0] / 1024**2:.2f} / {peaks[1] / 1024**2:.2f} MiB")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA and the actual attention kernels")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_head_chunks_match_all_heads(dtype):
  profiler.ensure_xattn_imports()
  torch.manual_seed(17)
  query = torch.randn(1, 6, 8195, 64, device="cuda:0", dtype=dtype)
  key = torch.randn(1, 2, 8195, 64, device="cuda:0", dtype=dtype)
  value = torch.randn_like(key)
  outputs, thresholds, peaks = [], [], []
  with torch.no_grad():
    for head_chunk_size in [0, 2]:
      config = profiler.ProfileConfig(0.9, 16, True, 128, head_chunk_size, 1024)
      module = SimpleNamespace(profile_layer_idx=0, profile_config=config)
      torch.cuda.synchronize(0)
      baseline = torch.cuda.memory_allocated(0)
      torch.cuda.reset_peak_memory_stats(0)
      outputs.append(profiler.xattn_prefill_profile(module, query, key, value, 128, 16))
      torch.cuda.synchronize(0)
      peaks.append(torch.cuda.max_memory_allocated(0) - baseline)
      thresholds.append(config.history_threshold[0])
  torch.testing.assert_close(thresholds[1], thresholds[0], rtol=5e-3, atol=5e-3)
  torch.testing.assert_close(outputs[1], outputs[0])
  assert peaks[1] < peaks[0]
  print(f"\nCUDA {dtype} profiling peak increment (all heads / 2 heads): "
        f"{peaks[0] / 1024**2:.2f} / {peaks[1] / 1024**2:.2f} MiB")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_two_gpu_qwen3_profiling_matches_single_gpu(dtype):
  profiler.ensure_xattn_imports()
  torch.manual_seed(17)
  config = Qwen3Config(vocab_size=64, hidden_size=256, intermediate_size=512,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    head_dim=64, max_position_embeddings=2048)
  config._attn_implementation = "eager"
  single = Qwen3ForCausalLM(config).eval().to(dtype=dtype)
  split = deepcopy(single)
  single.to("cuda:0")
  split = dispatch_model(split, device_map={"model.embed_tokens": 0,
    "model.layers.0": 0, "model.layers.1": 1, "model.norm": 1,
    "model.rotary_emb": 0, "lm_head": 1})
  thresholds = []
  hidden_states = []
  for model in [single, split]:
    profile_config = profiler.ProfileConfig(0.9, 16, True, 128, 2, 1024, mlp_chunk_size=256)
    profiler.patch_model_for_profiling(model, profile_config, "qwen3")
    hook = model.model.register_forward_hook(
      lambda module, args, output: hidden_states.append(output.last_hidden_state.detach().cpu()))
    profiler.run_profile_on_texts(model, lambda *args, **kwargs: BatchEncoding({
      "input_ids": (torch.arange(1031) % 64).unsqueeze(0),
    }), ["test"])
    hook.remove()
    thresholds.append(profiler.compute_final_threshold_from_history(profile_config))
  torch.testing.assert_close(thresholds[1], thresholds[0], rtol=5e-3, atol=5e-3)
  torch.testing.assert_close(hidden_states[1], hidden_states[0], rtol=5e-3, atol=5e-3)
