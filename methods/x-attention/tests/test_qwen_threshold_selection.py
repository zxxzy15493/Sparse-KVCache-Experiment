"""Model-specific XAttention presets and benchmark parameter forwarding."""

from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  Qwen2Config,
  Qwen2ForCausalLM,
  Qwen3Config,
  Qwen3ForCausalLM,
)

from threshold.threshold import qwen_threshold, qwen3_threshold
from xattn.src import load_qwen
from xattn.src import budget_load_qwen_recall


def make_model(kind="qwen3"):
  config_cls, model_cls = {
    "qwen2": (Qwen2Config, Qwen2ForCausalLM),
    "qwen3": (Qwen3Config, Qwen3ForCausalLM),
  }[kind]
  config = config_cls(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
  )
  config._attn_implementation = "eager"
  return model_cls(config).eval()


@pytest.mark.parametrize("kind", ["qwen2", "qwen3"])
@pytest.mark.parametrize("p, tag", [(0.8, 80), (0.85, 85), (0.9, 90), (0.95, 95)])
def test_patch_selects_preset_for_actual_model(kind, p, tag):
  config = load_qwen.FastPrefillConfig(p=p)
  model = load_qwen.patch_model(make_model(kind), config)
  module, prefix = (qwen3_threshold, "qwen3") if kind == "qwen3" else (qwen_threshold, "qwen")
  expected = torch.tensor(getattr(module, f"{prefix}_fuse_{tag}"))

  assert config.model_type == kind
  torch.testing.assert_close(config.threshold, expected, rtol=0, atol=0)
  for layer in model.model.layers:
    assert layer.self_attn.fastprefillconfig is config
    row = load_qwen._layer_threshold(config, layer.self_attn.layer_idx, 4, torch.device("cpu"))
    torch.testing.assert_close(row, expected[layer.self_attn.layer_idx, :4], rtol=0, atol=0)
  if kind == "qwen3":
    assert config.threshold.shape == (64, 64)
    last = load_qwen._layer_threshold(config, 63, 64, torch.device("cpu"))
    torch.testing.assert_close(last, expected[63], rtol=0, atol=0)


@pytest.mark.parametrize("threshold", [0.72, torch.tensor(0.72), [[0.7, 0.8], [0.85, 0.9]], torch.full((2, 4), 0.73)])
def test_explicit_threshold_takes_priority_over_preset(threshold):
  config = load_qwen.FastPrefillConfig(p=0.8, threshold=threshold)
  load_qwen.patch_model(make_model(), config)
  torch.testing.assert_close(torch.as_tensor(config.threshold), torch.as_tensor(threshold))


def test_default_qwen3_preset_is_p90():
  model = load_qwen.patch_model(make_model(), load_qwen.FastPrefillConfig())
  expected = torch.tensor(qwen3_threshold.qwen3_fuse_90)
  torch.testing.assert_close(model.model.layers[0].self_attn.fastprefillconfig.threshold, expected)


def test_generic_loader_detects_qwen3(monkeypatch):
  model = make_model()
  tokenizer = SimpleNamespace()
  monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: model)
  monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
  actual_model, actual_tokenizer = load_qwen.load_model(
    load_qwen.FastPrefillConfig(p=0.85), name_or_path="local-qwen3"
  )
  assert actual_model is model
  assert actual_tokenizer is tokenizer
  expected = torch.tensor(qwen3_threshold.qwen3_fuse_85)
  torch.testing.assert_close(model.model.layers[0].self_attn.fastprefillconfig.threshold, expected)


@pytest.fixture
def benchmark_loader(monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
  from loaders import xattention_load

  monkeypatch.setattr(Qwen3ForCausalLM, "from_pretrained", lambda *a, **kw: make_model())
  monkeypatch.setattr(
    AutoTokenizer, "from_pretrained", lambda *a, **kw: SimpleNamespace(pad_token_id=None, eos_token_id=2)
  )
  return xattention_load


def benchmark_args(overrides=(), **kwargs):
  return Namespace(
    model="qwen3-32b", model_path="local-qwen3", method="xattention", set=list(overrides), **kwargs
  )


@pytest.mark.parametrize("parameter", ["p", "fixthreshold", "fixthreshold_arg"])
@pytest.mark.parametrize("p, tag", [(0.8, 80), (0.85, 85), (0.9, 90), (0.95, 95)])
def test_benchmark_parameters_select_qwen3_preset(benchmark_loader, parameter, p, tag):
  args = (benchmark_args(fixthreshold=p) if parameter == "fixthreshold_arg"
          else benchmark_args([f"{parameter}={p}"]))
  model, tokenizer = benchmark_loader.load_model_and_tokenizer(args)
  actual = model.model.layers[0].self_attn.fastprefillconfig.threshold
  expected = torch.tensor(getattr(qwen3_threshold, f"qwen3_fuse_{tag}"))
  torch.testing.assert_close(actual, expected, rtol=0, atol=0)
  assert args._method_args.p == p
  assert tokenizer.pad_token_id == 2


@pytest.mark.parametrize("explicit", ["scalar", "table"])
def test_benchmark_explicit_threshold_takes_priority(benchmark_loader, tmp_path, explicit):
  overrides = ["p=0.85"]
  if explicit == "scalar":
    expected = 0.72
    overrides.append("threshold=0.72")
  else:
    expected = [[0.7, 0.8, 0.9, 0.95], [0.71, 0.81, 0.91, 0.96]]
    path = tmp_path / "threshold.json"
    path.write_text(json.dumps(expected))
    overrides.append(f"threshold_path={path}")
  model, _ = benchmark_loader.load_model_and_tokenizer(benchmark_args(overrides))
  actual = model.model.layers[0].self_attn.fastprefillconfig.threshold
  torch.testing.assert_close(torch.as_tensor(actual), torch.as_tensor(expected))


def test_benchmark_rejects_conflicting_threshold_overrides(benchmark_loader):
  with pytest.raises(ValueError, match="Set only one"):
    benchmark_loader.load_model_and_tokenizer(
      benchmark_args(["threshold=0.72", "threshold_path=unused.json"])
    )


def test_unsupported_preset_is_rejected():
  with pytest.raises(ValueError, match="Unsupported p"):
    load_qwen.FastPrefillConfig(p=0.7, model_type="qwen3")


@pytest.mark.parametrize("p, tag", [(0.8, 80), (0.85, 85), (0.9, 90), (0.95, 95)])
def test_recall_config_selects_qwen3_preset(p, tag):
  config = budget_load_qwen_recall.FastPrefillConfig(p=p, model_type="qwen3")
  expected = torch.tensor(getattr(qwen3_threshold, f"qwen3_fuse_{tag}"))
  torch.testing.assert_close(config.threshold, expected, rtol=0, atol=0)
  last = budget_load_qwen_recall._layer_threshold(config, 63, 64, torch.device("cpu"))
  torch.testing.assert_close(last, expected[63], rtol=0, atol=0)


def test_recall_explicit_scalar_needs_no_cuda():
  config = budget_load_qwen_recall.FastPrefillConfig(p=0.8, threshold=0.72, model_type="qwen3")
  assert budget_load_qwen_recall._layer_threshold(config, 63, 64, torch.device("cpu")) == 0.72
