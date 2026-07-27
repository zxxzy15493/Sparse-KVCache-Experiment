#
#
#
"""PyTorch ChatGLM model."""

from flash_attn import flash_attn_func


def glm_flash_attention_forward(
  self, query_states, key_states, value_states, attention_mask
):
  query_states = query_states.transpose(1, 2)
  key_states = key_states.transpose(1, 2)
  value_states = value_states.transpose(1, 2)
  batch_size, query_length = query_states.shape[:2]

  if query_length > 1:
    attn_output = flash_attn_func(
      query_states, key_states, value_states, causal=self.is_causal
    )
  else:
    attn_output = flash_attn_func(query_states, key_states, value_states)
  attn_output = attn_output.reshape(
    batch_size, query_length, self.hidden_size_per_partition
  ).contiguous()
  return attn_output
