#
#
#
"""PyTorch ChatGLM model."""

from flex_prefill.ops.flex_prefill_attention import flex_prefill_attention


def glm_flex_prefill_attention_forward(
  self, query_states, key_states, value_states, attention_mask
):
  query_states = query_states.transpose(1, 2)
  key_states = key_states.transpose(1, 2)
  value_states = value_states.transpose(1, 2)
  batch_size, query_length = query_states.shape[:2]

  layer_id = int(getattr(self, "layer_idx", -1))
  model_name="glm"

  attn_output = flex_prefill_attention(
    model_name,
    layer_id,
    query_states,
    key_states,
    value_states,
    gamma=self.config.flex_prefill_gamma,
    tau=self.config.flex_prefill_tau,
    block_size=self.config.block_size,
    task=getattr(self.config, "task", None),
    min_budget=getattr(self.config, "flex_prefill_min_budget", None),
    max_budget=getattr(self.config, "flex_prefill_max_budget", None),
    type=getattr(self.config, "type", None),
  )

  attn_output = attn_output.reshape(
    batch_size, query_length, self.hidden_size_per_partition
  ).contiguous()
  return attn_output
