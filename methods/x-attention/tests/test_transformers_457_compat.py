"""Transformers 4.57 compatibility tests for the active XAttention loaders."""

from copy import deepcopy
import inspect

import pytest
import torch
import torch.nn.functional as F
from torch import nn
import transformers
from transformers import (
  LlamaConfig,
  LlamaForCausalLM,
  Qwen2Config,
  Qwen2ForCausalLM,
  Qwen3Config,
  Qwen3ForCausalLM,
)

from xattn.src import Xattention as xattention_module
from xattn.src.Xattention import Xattention_prefill
from xattn.src.load_llama import (
  FastPrefillConfig as LlamaFastPrefillConfig,
  forward_eval as llama_forward_eval,
  patch_model as patch_llama,
)
from xattn.src.load_qwen import (
  FastPrefillConfig as QwenFastPrefillConfig,
  forward_eval as qwen_forward_eval,
  patch_model as patch_qwen,
)
from xattn.src.load_glm import (
  FastPrefillConfig as GlmFastPrefillConfig,
  glm_core_attention_forward,
)


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


def _make_model(kind: str):
  config_cls, model_cls, extra = {
    "llama": (LlamaConfig, LlamaForCausalLM, {}),
    "qwen2": (Qwen2Config, Qwen2ForCausalLM, {}),
    "qwen3": (Qwen3Config, Qwen3ForCausalLM, {"head_dim": 8}),
  }[kind]
  config = config_cls(**(COMMON_CONFIG | extra))
  config._attn_implementation = "eager"
  return model_cls(config).eval()


def _patch(model, kind: str):
  if kind == "llama":
    return patch_llama(model, LlamaFastPrefillConfig(threshold=0.9))
  return patch_qwen(model, QwenFastPrefillConfig(threshold=0.9))


@pytest.mark.parametrize("kind", ["llama", "qwen2", "qwen3"])
def test_prefill_decode_and_dynamic_cache_match_dense_model(kind: str) -> None:
  assert transformers.__version__ == "4.57.1"
  torch.manual_seed(17)
  dense = _make_model(kind)
  xattn = _patch(deepcopy(dense), kind).eval()
  input_ids = torch.tensor([[0, 0, 5, 6, 7], [0, 8, 9, 10, 11]])
  attention_mask = input_ids.ne(0).long()

  with torch.no_grad():
    dense_prefill = dense(input_ids, attention_mask=attention_mask, use_cache=True)
    xattn_prefill = xattn(input_ids, attention_mask=attention_mask, use_cache=True)
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
    xattn_decode = xattn(
      next_tokens,
      attention_mask=decode_mask,
      past_key_values=xattn_prefill.past_key_values,
      use_cache=True,
    )

  torch.testing.assert_close(xattn_prefill.logits, dense_prefill.logits)
  torch.testing.assert_close(xattn_decode.logits, dense_decode.logits)
  assert all(layer.keys.shape[-2] == 6 for layer in xattn_decode.past_key_values.layers)


@pytest.mark.parametrize("kind", ["llama", "qwen2", "qwen3"])
def test_flash_attention_compact_mask_with_left_padding(kind: str) -> None:
  torch.manual_seed(19)
  dense = _make_model(kind)
  xattn = _patch(deepcopy(dense), kind).eval()
  xattn.config._attn_implementation = "flash_attention_2"
  input_ids = torch.tensor([[0, 0, 5, 6, 7], [0, 8, 9, 10, 11]])
  attention_mask = input_ids.ne(0).long()

  with torch.no_grad():
    expected = dense(input_ids, attention_mask=attention_mask).logits
    actual = xattn(input_ids, attention_mask=attention_mask).logits

  torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("kind", ["llama", "qwen2", "qwen3"])
def test_generate_uses_patched_attention(kind: str) -> None:
  model = _patch(_make_model(kind), kind).eval()
  model.config._attn_implementation = "flash_attention_2"
  input_ids = torch.tensor([[0, 0, 5, 6, 7], [0, 8, 9, 10, 11]])
  attention_mask = input_ids.ne(0).long()
  with torch.no_grad():
    output = model.generate(
      input_ids,
      attention_mask=attention_mask,
      max_new_tokens=2,
      do_sample=False,
    )
  assert output.shape == (2, 7)
  assert all(hasattr(layer.self_attn, "fastprefillconfig") for layer in model.model.layers)


def test_prefill_requires_block_sparse_extension(monkeypatch) -> None:
  monkeypatch.setattr(xattention_module, "block_sparse_attn_func", None)
  query = torch.randn(1, 2, 5, 8)

  with pytest.raises(RuntimeError, match="XAttention requires block_sparse_attn_func"):
    Xattention_prefill(query, query, query, stride=4, threshold=0.9)


def test_cpu_prefill_fallback_and_457_signatures(monkeypatch) -> None:
  # Simulate an installed extension; CPU validation does not invoke its kernel.
  monkeypatch.setattr(xattention_module, "block_sparse_attn_func", object())
  torch.manual_seed(23)
  query = torch.randn(2, 4, 5, 8)
  key = torch.randn(2, 4, 5, 8)
  value = torch.randn(2, 4, 5, 8)
  actual = Xattention_prefill(query, key, value, stride=4, threshold=0.9)
  expected = F.scaled_dot_product_attention(query, key, value, is_causal=True)
  torch.testing.assert_close(actual, expected)

  expected_parameters = [
    "self",
    "hidden_states",
    "position_embeddings",
    "attention_mask",
    "past_key_values",
    "cache_position",
  ]
  assert list(inspect.signature(llama_forward_eval).parameters)[:6] == expected_parameters
  assert list(inspect.signature(qwen_forward_eval).parameters)[:6] == expected_parameters


def test_qwen_fastprefill_config_accepts_per_layer_head_table() -> None:
  table = torch.tensor([[0.8, 0.9], [0.85, 0.95]])
  config = QwenFastPrefillConfig(threshold=table)

  assert config.threshold.device.type == "cpu"
  assert config.threshold.dtype == torch.float32
  torch.testing.assert_close(config.threshold, table)


def test_glm_cpu_fallback_honors_boolean_mask() -> None:
  class CoreAttention(nn.Module):
    def __init__(self):
      super().__init__()
      self.fastprefillconfig = GlmFastPrefillConfig(threshold=0.9)
      self.layer_idx = 0
      self.hidden_size_per_partition = 32
      self.attention_dropout = nn.Dropout(0.0)

  torch.manual_seed(29)
  core = CoreAttention().eval()
  query = torch.randn(2, 4, 5, 8)
  key = torch.randn(2, 4, 5, 8)
  value = torch.randn(2, 4, 5, 8)
  blocked = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
  blocked = blocked[None, None].expand(2, 1, 5, 5)

  actual = glm_core_attention_forward(core, query, key, value, blocked)
  weights = torch.matmul(query, key.transpose(2, 3)) / (query.shape[-1] ** 0.5)
  weights = torch.softmax(weights.masked_fill(blocked, float("-inf")), dim=-1)
  expected = torch.matmul(weights, value).transpose(1, 2).reshape(2, 5, 32)
  torch.testing.assert_close(actual, expected)
