from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version
import transformers
from transformers import (
  LlamaConfig,
  LlamaForCausalLM,
  Qwen2Config,
  Qwen2ForCausalLM,
  Qwen3Config,
  Qwen3ForCausalLM,
)

from evaluation.quest_attention import enable_quest_attention_eval as patch_llama
from evaluation.quest_qwen_attention import enable_quest_attention_eval as patch_qwen
from evaluation.llama import enable_tuple_kv_cache_for_llama
import evaluation.quest_attention_retrieve as retrieve_patch


pytestmark = pytest.mark.skipif(
  not (Version("4.57") <= Version(transformers.__version__) < Version("4.58")),
  reason="Quest compatibility regression targets Transformers 4.57.x",
)


def _cases():
  common = dict(
    vocab_size=97,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
  )
  return [
    pytest.param(LlamaConfig(**common), LlamaForCausalLM, patch_llama, id="llama"),
    pytest.param(Qwen2Config(**common), Qwen2ForCausalLM, patch_qwen, id="qwen2"),
    pytest.param(
      Qwen3Config(**common, head_dim=8),
      Qwen3ForCausalLM,
      patch_qwen,
      id="qwen3",
    ),
  ]


@pytest.mark.parametrize("config,model_cls,patch", _cases())
def test_prefill_decode_cache_and_full_budget_equivalence(config, model_cls, patch):
  torch.manual_seed(0)
  config._attn_implementation = "eager"
  reference = model_cls(config).eval()
  quest = deepcopy(reference).eval()
  args = SimpleNamespace(token_budget=128, chunk_size=4, save_path=None, task=None)

  # Patching twice must not replace the saved native forward with Quest itself.
  patch(quest, args)
  patch(quest, args)
  assert [layer.self_attn.layer_id for layer in quest.model.layers] == [0, 1, 2, 3]

  input_ids = torch.tensor([[0, 0, 4, 5, 6], [7, 8, 9, 10, 11]])
  attention_mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
  next_ids = torch.tensor([[12], [13]])
  next_mask = torch.cat(
    [attention_mask, torch.ones(2, 1, dtype=attention_mask.dtype)],
    dim=1,
  )

  with torch.no_grad():
    reference_prefill = reference(
      input_ids=input_ids,
      attention_mask=attention_mask,
      use_cache=True,
    )
    quest_prefill = quest(
      input_ids=input_ids,
      attention_mask=attention_mask,
      use_cache=True,
    )
    reference_decode = reference(
      input_ids=next_ids,
      attention_mask=next_mask,
      past_key_values=reference_prefill.past_key_values,
      use_cache=True,
    )
    quest_decode = quest(
      input_ids=next_ids,
      attention_mask=next_mask,
      past_key_values=quest_prefill.past_key_values,
      use_cache=True,
    )

  torch.testing.assert_close(
    quest_decode.logits,
    reference_decode.logits,
    rtol=1e-5,
    atol=1e-6,
  )
  assert [
    quest_decode.past_key_values.get_seq_length(layer_idx)
    for layer_idx in range(config.num_hidden_layers)
  ] == [6, 6, 6, 6]


@pytest.mark.parametrize("config,model_cls,patch", _cases())
def test_generate_uses_transformers_457_attention_contract(config, model_cls, patch):
  torch.manual_seed(1)
  config._attn_implementation = "eager"
  model = model_cls(config).eval()
  patch(
    model,
    SimpleNamespace(token_budget=4, chunk_size=2, save_path=None, task=None),
  )

  input_ids = torch.tensor([[0, 4, 5], [6, 7, 8]])
  attention_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
  with torch.no_grad():
    output = model.generate(
      input_ids=input_ids,
      attention_mask=attention_mask,
      max_new_tokens=2,
      do_sample=False,
      pad_token_id=0,
    )

  assert output.shape == (2, 5)


@pytest.mark.parametrize("config,model_cls,patch", _cases())
def test_decode_accepts_compact_2d_padding_mask(config, model_cls, patch):
  torch.manual_seed(2)
  config._attn_implementation = "eager"
  model = model_cls(config).eval()
  patch(
    model,
    SimpleNamespace(token_budget=4, chunk_size=2, save_path=None, task=None),
  )

  input_ids = torch.tensor([[0, 0, 4, 5, 6], [7, 8, 9, 10, 11]])
  attention_mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
  with torch.no_grad():
    prefill = model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      use_cache=True,
    )

    hidden_states = torch.randn(2, 1, config.hidden_size)
    position_ids = torch.full((2, 1), 5, dtype=torch.long)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    compact_mask = torch.cat(
      [attention_mask, torch.ones(2, 1, dtype=attention_mask.dtype)],
      dim=1,
    )
    output, weights = model.model.layers[2].self_attn(
      hidden_states=hidden_states,
      position_embeddings=position_embeddings,
      attention_mask=compact_mask,
      past_key_values=prefill.past_key_values,
      cache_position=torch.tensor([5]),
      position_ids=position_ids,
      use_cache=True,
    )

  assert output.shape == (2, 1, config.hidden_size)
  assert weights is None
  assert torch.isfinite(output).all()
  assert prefill.past_key_values.get_seq_length(2) == 6


def test_legacy_tuple_cache_patch_is_safely_disabled():
  from transformers.models.llama.modeling_llama import LlamaAttention

  original_forward = LlamaAttention.forward
  with pytest.warns(RuntimeWarning, match="legacy tuple-KV monkey patch is disabled"):
    enable_tuple_kv_cache_for_llama()
  assert LlamaAttention.forward is original_forward


def test_retrieve_instrumentation_patch_uses_457_cache_contract(monkeypatch):
  def dense_selected_attention(query, key, value, selected_valid):
    scores = torch.matmul(query, key.transpose(2, 3)) / (query.shape[-1] ** 0.5)
    scores = scores.masked_fill(~selected_valid, torch.finfo(scores.dtype).min)
    return torch.matmul(torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype), value)

  monkeypatch.setattr(
    retrieve_patch,
    "_flash_attention_selected_tokens",
    dense_selected_attention,
  )
  config = LlamaConfig(
    vocab_size=97,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
  )
  config._attn_implementation = "eager"
  model = LlamaForCausalLM(config).eval()
  retrieve_patch.enable_quest_attention_eval(
    model,
    SimpleNamespace(token_budget=16, chunk_size=2),
  )

  with torch.no_grad():
    prefill = model(input_ids=torch.tensor([[4, 5, 6]]), use_cache=True)
    decode = model(
      input_ids=torch.tensor([[7]]),
      attention_mask=torch.ones(1, 4, dtype=torch.long),
      past_key_values=prefill.past_key_values,
      use_cache=True,
    )

  assert decode.logits.shape == (1, 1, 97)
  assert [decode.past_key_values.get_seq_length(i) for i in range(4)] == [4, 4, 4, 4]


def test_retrieval_rejects_cpu_instead_of_falling_back():
  query = torch.zeros(1, 2, 1, 8)
  valid = torch.ones(1, 2, 1, 1, dtype=torch.bool)
  with pytest.raises(RuntimeError, match="requires CUDA"):
    retrieve_patch._flash_attention_selected_tokens(query, query, query, valid)


@pytest.mark.parametrize("all_valid", [True, False])
def test_retrieval_requires_flash_attention(monkeypatch, all_valid):
  query = SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16, shape=(1, 2, 1, 8))
  key = torch.zeros(1, 2, 2, 8)
  valid = torch.tensor([[[[True, all_valid]], [[True, all_valid]]]])
  monkeypatch.setattr(retrieve_patch, "flash_attn_func", None)
  monkeypatch.setattr(retrieve_patch, "flash_attn_varlen_func", None)
  with pytest.raises(RuntimeError, match="requires flash_attn"):
    retrieve_patch._flash_attention_selected_tokens(query, key, key, valid)


@pytest.mark.parametrize("config,model_cls,patch", _cases())
def test_native_prefill_errors_propagate(config, model_cls, patch):
  config._attn_implementation = "eager"
  model = model_cls(config).eval()
  patch(model, SimpleNamespace(token_budget=4, chunk_size=2, save_path=None, task=None))
  error = RuntimeError("native attention failed")
  def fail(*args, **kwargs):
    raise error
  model.model.layers[0].self_attn.flash_forward = fail
  with pytest.raises(RuntimeError) as raised:
    model(input_ids=torch.tensor([[4, 5, 6]]), use_cache=True)
  assert raised.value is error


@pytest.mark.parametrize("config,model_cls,patch", _cases())
def test_sdpa_configuration_is_rejected(config, model_cls, patch):
  config._attn_implementation = "sdpa"
  model = model_cls(config).eval()
  with pytest.raises(ValueError, match="Quest does not support SDPA"):
    patch(model, SimpleNamespace(token_budget=4, chunk_size=2))
