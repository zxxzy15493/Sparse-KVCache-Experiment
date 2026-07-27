#
#
#
import torch
import torch.nn.functional as F


def glm_mlp_forward(self, hidden_states):
  def inner_mlp_forward(x):
    intermediate_parallel = self.dense_h_to_4h(x)
    intermediate_parallel = self.activation_func(intermediate_parallel)
    output = self.dense_4h_to_h(intermediate_parallel)
    return output

  batch_size, seq_len, hidden_dim = hidden_states.shape
  chunk_size = 32768
  output = torch.empty_like(hidden_states)
  for b in range(batch_size):
    for i in range(0, seq_len, chunk_size):
      output[b : b + 1, i : i + chunk_size] = inner_mlp_forward(
        hidden_states[b : b + 1, i : i + chunk_size]
      )
  return output
