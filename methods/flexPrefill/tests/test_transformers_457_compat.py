"""Transformers 4.57 compatibility tests for FlexPrefill."""

from copy import deepcopy
import importlib
import inspect

import pytest
import torch
import torch.nn.functional as F
import transformers
from transformers import (
  LlamaConfig,
  LlamaForCausalLM,
  Qwen2Config,
  Qwen2ForCausalLM,
  Qwen3Config,
  Qwen3ForCausalLM,
)

from flex_prefill import patch_model
from flex_prefill.modules.llama.flex_prefill_attention import (
  llama_flex_prefill_attention_forward,
)
from flex_prefill.modules.qwen2.flex_prefill_attention import (
  qwen2_flex_prefill_attention_forward,
)
from flex_prefill.ops.flex_prefill_attention import flex_prefill_attention

attention_module = importlib.import_module("flex_prefill.ops.flex_prefill_attention")


COMMON_CONFIG = dict(
  vocab_size=64,
  hidden_size=32,
  intermediate_size=64,
  num_hidden_layers=2,
  num_attention_heads=4,
  num_key_value_heads=2,
  max_position_embeddings=64,
  bos_token_id=1,
  eos_token_id=2,
  pad_token_id=0,
)

FLEX_CONFIG = {
  "block_size": 16,
  "flex_prefill_gamma": 0.9,
  "flex_prefill_tau": 0.1,
  "flex_prefill_min_budget": 1,
  "flex_prefill_max_budget": None,
}


@pytest.fixture
def cpu_reference_attention(monkeypatch):
  """Validate model/cache contracts with an explicit test-only CPU backend."""
  def reference(model_name, layer_id, q, k, v, *, attention_mask=None, **kwargs):
    query, key, value = (tensor.transpose(1, 2) for tensor in (q, k, v))
    repeats = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    q_len, k_len = query.shape[-2], key.shape[-2]
    if attention_mask is not None and attention_mask.ndim == 2:
      allowed = (torch.arange(k_len)[None, :] <= torch.arange(q_len)[:, None] + k_len - q_len)
      allowed = allowed & attention_mask.bool()[:, None, :]
      attention_mask = (~allowed)[:, None].to(q.dtype) * torch.finfo(q.dtype).min
    return F.scaled_dot_product_attention(
      query, key, value, attn_mask=attention_mask,
      is_causal=attention_mask is None and q_len > 1,
    ).transpose(1, 2)
  for name in ["llama", "qwen2"]:
    module = importlib.import_module(f"flex_prefill.modules.{name}.flex_prefill_attention")
    monkeypatch.setattr(module, "flex_prefill_attention", reference)


def _make_model(kind: str):
  config_cls, model_cls, extra = {
    "llama": (LlamaConfig, LlamaForCausalLM, {}),
    "qwen2": (Qwen2Config, Qwen2ForCausalLM, {}),
    "qwen3": (Qwen3Config, Qwen3ForCausalLM, {"head_dim": 8}),
  }[kind]
  config = config_cls(**(COMMON_CONFIG | extra))
  config._attn_implementation = "eager"
  return model_cls(config).eval()


@pytest.mark.parametrize("kind", ["llama", "qwen2", "qwen3"])
@pytest.mark.usefixtures("cpu_reference_attention")
def test_prefill_decode_and_dynamic_cache_match_dense_model(kind: str) -> None:
  assert transformers.__version__ == "4.57.1"
  torch.manual_seed(31)
  dense = _make_model(kind)
  flex = patch_model(deepcopy(dense), "flex_prefill", FLEX_CONFIG.copy()).eval()
  input_ids = torch.tensor([[0, 0, 5, 6, 7], [0, 8, 9, 10, 11]])
  attention_mask = input_ids.ne(0).long()

  with torch.no_grad():
    dense_prefill = dense(input_ids, attention_mask=attention_mask, use_cache=True)
    flex_prefill = flex(input_ids, attention_mask=attention_mask, use_cache=True)
    decode_mask = torch.cat(
      [attention_mask, torch.ones((2, 1), dtype=attention_mask.dtype)], dim=1
    )
    next_tokens = torch.tensor([[12], [13]])
    dense_decode = dense(
      next_tokens,
      attention_mask=decode_mask,
      past_key_values=dense_prefill.past_key_values,
      use_cache=True,
    )
    flex_decode = flex(
      next_tokens,
      attention_mask=decode_mask,
      past_key_values=flex_prefill.past_key_values,
      use_cache=True,
    )

  assert flex_prefill.logits.shape == (2, 5, 64)
  torch.testing.assert_close(flex_prefill.logits, dense_prefill.logits)
  torch.testing.assert_close(flex_decode.logits, dense_decode.logits)
  assert all(layer.keys.shape[-2] == 6 for layer in flex_decode.past_key_values.layers)


@pytest.mark.parametrize("kind", ["llama", "qwen2", "qwen3"])
@pytest.mark.usefixtures("cpu_reference_attention")
def test_flash_attention_compact_mask_and_generate(kind: str) -> None:
  torch.manual_seed(37)
  dense = _make_model(kind)
  flex = patch_model(deepcopy(dense), "flex_prefill", FLEX_CONFIG.copy()).eval()
  flex.config._attn_implementation = "flash_attention_2"
  input_ids = torch.tensor([[0, 0, 5, 6, 7], [0, 8, 9, 10, 11]])
  attention_mask = input_ids.ne(0).long()

  with torch.no_grad():
    expected = dense(input_ids, attention_mask=attention_mask).logits
    actual = flex(input_ids, attention_mask=attention_mask).logits
    generated = flex.generate(
      input_ids,
      attention_mask=attention_mask,
      max_new_tokens=2,
      do_sample=False,
    )

  torch.testing.assert_close(actual, expected)
  assert generated.shape == (2, 7)


@pytest.mark.parametrize("masked", [False, True])
def test_unsupported_inputs_raise_without_sdpa(monkeypatch, masked) -> None:
  monkeypatch.setattr(attention_module, "flash_attn_func", object())
  query = torch.randn(2, 5, 4, 8)
  key = torch.randn(2, 5, 2, 8)
  value = torch.randn(2, 5, 2, 8)
  attention_mask = torch.ones(2, 5) if masked else None
  error = NotImplementedError if masked else RuntimeError
  with pytest.raises(error, match="attention_mask" if masked else "requires CUDA"):
    flex_prefill_attention("Qwen", 0, query, key, value, gamma=0.9, attention_mask=attention_mask)


def test_missing_flash_attention_raises(monkeypatch):
  monkeypatch.setattr(attention_module, "flash_attn_func", None)
  query = torch.zeros(1, 1, 2, 16)
  with pytest.raises(RuntimeError, match="FlexPrefill requires flash_attn_func"):
    flex_prefill_attention("Qwen", 0, query, query, query, gamma=0.9)


def test_flash_attention_errors_propagate(monkeypatch):
  error = RuntimeError("FlashAttention kernel failed")
  def fail(*args, **kwargs):
    raise error
  monkeypatch.setattr(attention_module, "_validate_attention_inputs", lambda *args: None)
  monkeypatch.setattr(attention_module, "flash_attn_func", fail)
  query = torch.zeros(1, 1, 2, 16)
  with pytest.raises(RuntimeError) as raised:
    flex_prefill_attention("Qwen", 0, query, query, query, gamma=0.9)
  assert raised.value is error


def test_457_signatures_and_qwen3_sliding_layer_routing() -> None:
  expected_parameters = [
    "self",
    "hidden_states",
    "position_embeddings",
    "attention_mask",
    "past_key_values",
    "cache_position",
  ]
  assert list(inspect.signature(llama_flex_prefill_attention_forward).parameters)[:6] == expected_parameters
  assert list(inspect.signature(qwen2_flex_prefill_attention_forward).parameters)[:6] == expected_parameters

  config = Qwen3Config(
    **(
      COMMON_CONFIG
      | {"head_dim": 8, "use_sliding_window": True, "sliding_window": 4}
    ),
    layer_types=["full_attention", "sliding_attention"],
  )
  model = Qwen3ForCausalLM(config).eval()
  native_sliding_forward = model.model.layers[1].self_attn.forward.__func__
  patch_model(model, "flex_prefill", FLEX_CONFIG.copy())
  assert hasattr(model.model.layers[0].self_attn, "num_heads")
  assert model.model.layers[1].self_attn.forward.__func__ is native_sliding_forward
