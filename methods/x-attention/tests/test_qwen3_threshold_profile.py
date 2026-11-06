"""Forward consistency tests for threshold profiling on Hugging Face decoders."""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from transformers import (
  LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM,
  Qwen3Config, Qwen3ForCausalLM,
)

from threshold.train_scripts import train_threshold


@pytest.mark.parametrize("kind", ["llama", "qwen2", "qwen3"])
def test_profiler_preserves_dense_attention(monkeypatch, kind) -> None:
  config_type, model_type = {
    "llama": (LlamaConfig, LlamaForCausalLM),
    "qwen2": (Qwen2Config, Qwen2ForCausalLM),
    "qwen3": (Qwen3Config, Qwen3ForCausalLM),
  }[kind]
  config = config_type(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    max_position_embeddings=64,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
  )
  config._attn_implementation = "eager"
  dense = model_type(config).eval()
  profiled = deepcopy(dense)

  def dense_profile(
    _module,
    query_states,
    key_states,
    value_states,
    block_size,
    stride,
    chunk_size=16384,
    causal=True,
  ):
    del block_size, stride, chunk_size
    return F.scaled_dot_product_attention(
      query_states,
      key_states,
      value_states,
      is_causal=causal,
      enable_gqa=True,
    )

  monkeypatch.setattr(train_threshold, "xattn_prefill_profile", dense_profile)
  profile_config = train_threshold.ProfileConfig(
    p=0.9,
    stride=16,
    causal=True,
    block_size=128,
    mlp_chunk_size=3,
  )
  model_type = train_threshold.patch_model_for_profiling(
    profiled,
    profile_config,
    requested_family=kind,
  )

  input_ids = torch.tensor([[1, 5, 6, 7, 8]])
  with torch.no_grad():
    expected = dense(input_ids, use_cache=False).logits
    actual = profiled(input_ids, use_cache=False).logits

  assert model_type == kind
  torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
