"""Transformers 4.57 compatibility coverage for SparQ model conversion."""

from copy import deepcopy

import pytest
import torch
import transformers
from transformers import (
  GemmaConfig,
  GemmaForCausalLM,
  GPTNeoXConfig,
  GPTNeoXForCausalLM,
  LlamaConfig,
  LlamaForCausalLM,
  MistralConfig,
  MistralForCausalLM,
  Qwen2Config,
  Qwen2ForCausalLM,
  Qwen3Config,
  Qwen3ForCausalLM,
)

from llminference.methods.ann_attention_copy import AnnAttention, Settings, convert
from llminference.models import flash_attention as flash_backend


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


@pytest.fixture
def cpu_reference_backend(monkeypatch):
  """Validate model/cache contracts with an explicit test-only CPU backend."""
  def reference(query, key, value, *, attention_mask=None, causal=False,
                dropout_p=0.0, softmax_scale=None):
    query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
    repeats = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    scores = query @ key.transpose(-1, -2)
    scores *= query.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    if attention_mask is not None:
      if attention_mask.ndim == 5:
        attention_mask = attention_mask.flatten(0, 1)
      scores += attention_mask
    if causal:
      q_len, k_len = query.shape[-2], key.shape[-2]
      blocked = torch.arange(k_len)[None, :] > torch.arange(q_len)[:, None] + k_len - q_len
      scores = scores.masked_fill(blocked, torch.finfo(scores.dtype).min)
    return (scores.softmax(-1) @ value).transpose(1, 2)
  monkeypatch.setattr(flash_backend, "flash_attention", reference)


def _make_model(kind: str):
  config_cls, model_cls, extra = {
    "llama": (LlamaConfig, LlamaForCausalLM, {}),
    "qwen2": (Qwen2Config, Qwen2ForCausalLM, {}),
    "qwen3": (Qwen3Config, Qwen3ForCausalLM, {"head_dim": 8}),
    "mistral": (
      MistralConfig,
      MistralForCausalLM,
      {"sliding_window": 32},
    ),
    "gemma": (GemmaConfig, GemmaForCausalLM, {"head_dim": 8}),
    "gpt_neox": (
      GPTNeoXConfig,
      GPTNeoXForCausalLM,
      {"rotary_pct": 1.0},
    ),
  }[kind]
  kwargs = COMMON_CONFIG | extra
  if kind == "gpt_neox":
    kwargs.pop("num_key_value_heads")
  config = config_cls(**kwargs)
  config._attn_implementation = "eager"
  return model_cls(config).eval()


def _settings(k: int = 64, reallocate_to_mean_value: bool = False) -> Settings:
  return Settings(
    k=k,
    local_k=2,
    reallocate_to_mean_value=reallocate_to_mean_value,
    score="sparse_q",
    rank=4,
  )


@pytest.mark.parametrize(
  "kind", ["llama", "qwen2", "qwen3", "mistral", "gemma", "gpt_neox"]
)
@pytest.mark.usefixtures("cpu_reference_backend")
def test_prefill_decode_and_dynamic_cache_match_dense_model(kind: str) -> None:
  assert transformers.__version__ == "4.57.1"
  torch.manual_seed(7)
  dense = _make_model(kind)
  sparse = convert(deepcopy(dense), _settings()).eval()

  input_ids = torch.tensor([[0, 0, 5, 6, 7], [0, 8, 9, 10, 11]])
  attention_mask = input_ids.ne(0).long()
  with torch.no_grad():
    dense_prefill = dense(input_ids, attention_mask=attention_mask, use_cache=True)
    sparse_prefill = sparse(input_ids, attention_mask=attention_mask, use_cache=True)

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
    sparse_decode = sparse(
      next_tokens,
      attention_mask=decode_mask,
      past_key_values=sparse_prefill.past_key_values,
      use_cache=True,
    )

  torch.testing.assert_close(sparse_prefill.logits, dense_prefill.logits)
  torch.testing.assert_close(sparse_decode.logits, dense_decode.logits)
  assert len(list(sparse.modules())) > 0
  assert all(layer.keys.shape[-2] == 6 for layer in sparse_decode.past_key_values.layers)


@pytest.mark.parametrize("kind", ["llama", "qwen2", "qwen3"])
@pytest.mark.usefixtures("cpu_reference_backend")
def test_generate_uses_sparse_decode(kind: str) -> None:
  torch.manual_seed(11)
  model = convert(
    _make_model(kind), _settings(k=2, reallocate_to_mean_value=True)
  ).eval()
  ann_modules = [module for module in model.modules() if isinstance(module, AnnAttention)]
  assert len(ann_modules) == 2
  for module in ann_modules:
    module.debug_indices = []

  with torch.no_grad():
    output = model.generate(
      torch.tensor([[5, 6, 7, 8]]),
      attention_mask=torch.ones((1, 4), dtype=torch.long),
      max_new_tokens=2,
      do_sample=False,
    )

  assert output.shape == (1, 6)
  assert all(module.debug_indices for module in ann_modules)


@pytest.mark.parametrize("kind", ["llama", "qwen2", "qwen3"])
@pytest.mark.usefixtures("cpu_reference_backend")
def test_flash_attention_mask_contract_with_left_padding(kind: str) -> None:
  """The 4.57 flash path supplies a compact 2D mask to custom attention."""
  torch.manual_seed(13)
  dense = _make_model(kind)
  sparse = convert(deepcopy(dense), _settings()).eval()
  sparse.config._attn_implementation = "flash_attention_2"
  input_ids = torch.tensor([[0, 0, 5, 6, 7], [0, 8, 9, 10, 11]])
  attention_mask = input_ids.ne(0).long()

  with torch.no_grad():
    expected = dense(input_ids, attention_mask=attention_mask).logits
    actual = sparse(input_ids, attention_mask=attention_mask).logits

  torch.testing.assert_close(actual, expected)


def test_conversion_is_idempotent_and_qwen3_keeps_projection_layout() -> None:
  model = convert(_make_model("qwen3"), _settings())
  first_attention = model.model.layers[0].self_attn
  assert first_attention.q_proj.bias is None
  assert first_attention.k_proj.bias is None
  assert convert(model, _settings()) is model
  assert model.model.layers[0].self_attn is first_attention


def test_missing_flash_attention_raises(monkeypatch):
  monkeypatch.setattr(flash_backend, "flash_attn_func", None)
  query = torch.zeros(1, 1, 2, 8)
  with pytest.raises(RuntimeError, match="SparQ requires flash_attn_func"):
    flash_backend.flash_attention(query, query, query)


@pytest.mark.parametrize("masked", [False, True])
def test_unsupported_inputs_raise_without_dense_fallback(monkeypatch, masked):
  monkeypatch.setattr(flash_backend, "flash_attn_func", object())
  query = torch.zeros(1, 1, 2, 8)
  mask = torch.zeros(1, 1, 1, 1) if masked else None
  error = NotImplementedError if masked else RuntimeError
  with pytest.raises(error, match="attention_mask" if masked else "requires CUDA"):
    flash_backend.flash_attention(query, query, query, attention_mask=mask)


def test_ann_backend_errors_propagate(monkeypatch):
  error = RuntimeError("sparse backend failed")
  def fail(*args, **kwargs):
    raise error
  monkeypatch.setattr(flash_backend, "flash_attention", fail)
  ann = AnnAttention(_settings(k=2), n_kv_heads=2, head_size=8)
  query = torch.zeros(1, 2, 2, 1, 8)
  key = torch.zeros(1, 2, 1, 3, 8)
  with pytest.raises(RuntimeError) as raised:
    ann._attention(query, key, key, None, torch.tensor(1.0), torch.zeros_like(query))
  assert raised.value is error
