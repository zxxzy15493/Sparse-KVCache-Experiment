#
#
#
import torch


def qwen2_mlp_forward(self, hidden_state):
  def inner_mlp_forward(x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

  batch_size, seq_len, hidden_dim = hidden_state.shape
  chunk_size = 32768

  output = torch.empty_like(hidden_state)
  for b in range(batch_size):
    for i in range(0, seq_len, chunk_size):
      output[b : b + 1, i : i + chunk_size] = inner_mlp_forward(
        hidden_state[b : b + 1, i : i + chunk_size]
      )
  return output
