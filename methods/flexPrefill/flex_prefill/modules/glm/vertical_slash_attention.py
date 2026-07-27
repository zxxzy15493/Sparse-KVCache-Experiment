#
#
#
"""PyTorch ChatGLM model."""

from flash_attn import flash_attn_func

from flex_prefill.ops.vertical_slash_attention import vertical_slash_attention


def glm_vertical_slash_attention_forward(
  self, query_states, key_states, value_states, attention_mask
):
  query_states = query_states.transpose(1, 2)
  key_states = key_states.transpose(1, 2)
  value_states = value_states.transpose(1, 2)
  batch_size, query_length = query_states.shape[:2]

  if query_states.shape[1] > 1:
    attn_output = vertical_slash_attention(
      query_states,
      key_states,
      value_states,
      block_size=self.config.block_size,
      vertical_size=self.config.vertical_size,
      slash_size=self.config.slash_size,
    )
  else:
    attn_output = flash_attn_func(query_states, key_states, value_states)

  attn_output = attn_output.reshape(
    batch_size, query_length, self.hidden_size_per_partition
  ).contiguous()
  return attn_output
