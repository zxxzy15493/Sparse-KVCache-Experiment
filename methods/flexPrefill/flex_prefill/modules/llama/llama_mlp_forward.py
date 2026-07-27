#
#
#
import torch
import torch.nn.functional as F


def llama_mlp_forward(self, x):
  if self.config.pretraining_tp > 1:
    slice = self.intermediate_size // self.config.pretraining_tp
    gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
    up_proj_slices = self.up_proj.weight.split(slice, dim=0)
    down_proj_slices = self.down_proj.weight.split(slice, dim=1)

    gate_proj = torch.cat(
      [
        F.linear(x, gate_proj_slices[i])
        for i in range(self.config.pretraining_tp)
      ],
      dim=-1,
    )
    up_proj = torch.cat(
      [F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)],
      dim=-1,
    )

    intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
    down_proj = [
      F.linear(intermediate_states[i], down_proj_slices[i])
      for i in range(self.config.pretraining_tp)
    ]
    down_proj = sum(down_proj)
  else:

    def inner_mlp_forward(xx):
      return self.down_proj(self.act_fn(self.gate_proj(xx)) * self.up_proj(xx))

    batch_size, seq_len, hidden_dim = x.shape
    chunk_size = 32768
    down_proj = torch.empty_like(x)
    for b in range(batch_size):
      for i in range(0, seq_len, chunk_size):
        down_proj[b : b + 1, i : i + chunk_size] = inner_mlp_forward(
          x[b : b + 1, i : i + chunk_size]
        )
  return down_proj
