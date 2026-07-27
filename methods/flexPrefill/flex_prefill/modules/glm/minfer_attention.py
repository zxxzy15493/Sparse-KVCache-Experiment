#
#
#
"""PyTorch ChatGLM model."""

from flash_attn import flash_attn_func

from flex_prefill.ops.minfer.minfer_attention import minfer_attention


def glm_minfer_attention_forward(
  self, query_states, key_states, value_states, attention_mask
):
  batch_size, query_length = query_states.shape[0], query_states.shape[2]
  gqa_groups = query_states.shape[1] // key_states.shape[1]

  if query_states.shape[2] > 1:
    attn_output = minfer_attention(
      query_states,
      key_states,
      value_states,
      self.config.minfer_config[self.layer_number - 1],
    ).transpose(1, 2)
  else:
    attn_output = flash_attn_func(
      query_states.transpose(1, 2),
      key_states.transpose(1, 2),
      value_states.transpose(1, 2),
    )

  attn_output = attn_output.reshape(
    batch_size, query_length, self.hidden_size_per_partition
  ).contiguous()
  return attn_output
