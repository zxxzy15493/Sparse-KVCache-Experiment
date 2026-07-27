#
#
#
from typing import List

import torch


@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
  b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
  rot_dim = rope_cache.shape[-2] * 2
  x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
  rope_cache = rope_cache[:, :sq]
  xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
  rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
  x_out2 = torch.stack(
    [
      xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
      xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
    ],
    -1,
  )
  x_out2 = x_out2.flatten(3)
  return torch.cat((x_out2, x_pass), dim=-1)


def split_tensor_along_last_dim(
  tensor: torch.Tensor,
  num_partitions: int,
  contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
  """Split a tensor along its last dimension.

  Arguments:
    tensor: input tensor.
    num_partitions: number of partitions to split the tensor
    contiguous_split_chunks: If True, make each chunk contiguous
                 in memory.

  Returns:
    A list of Tensors
  """
  last_dim = tensor.dim() - 1
  last_dim_size = tensor.size()[last_dim] // num_partitions
  tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
  if contiguous_split_chunks:
    return tuple(chunk.contiguous() for chunk in tensor_list)

  return tensor_list


def glm_self_attention_forward(
  self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
):

  # =================================================
  # =================================================
  # =====================
  # =====================

  mixed_x_layer = self.query_key_value(hidden_states)

  if self.multi_query_attention:
    (query_layer, key_layer, value_layer) = mixed_x_layer.split(
      [
        self.num_attention_heads_per_partition
        * self.hidden_size_per_attention_head,
        self.num_multi_query_groups_per_partition
        * self.hidden_size_per_attention_head,
        self.num_multi_query_groups_per_partition
        * self.hidden_size_per_attention_head,
      ],
      dim=-1,
    )
    query_layer = query_layer.view(
      query_layer.size()[:-1]
      + (
        self.num_attention_heads_per_partition,
        self.hidden_size_per_attention_head,
      )
    )
    key_layer = key_layer.view(
      key_layer.size()[:-1]
      + (
        self.num_multi_query_groups_per_partition,
        self.hidden_size_per_attention_head,
      )
    )
    value_layer = value_layer.view(
      value_layer.size()[:-1]
      + (
        self.num_multi_query_groups_per_partition,
        self.hidden_size_per_attention_head,
      )
    )
  else:
    new_tensor_shape = mixed_x_layer.size()[:-1] + (
      self.num_attention_heads_per_partition,
      3 * self.hidden_size_per_attention_head,
    )
    mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

    (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(
      mixed_x_layer, 3
    )

  query_layer, key_layer, value_layer = [
    k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]
  ]

  if rotary_pos_emb is not None:
    query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
    key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

  if kv_cache is not None:
    cache_k, cache_v = kv_cache
    key_layer = torch.cat((cache_k, key_layer), dim=2)
    value_layer = torch.cat((cache_v, value_layer), dim=2)
  if use_cache:
    if kv_cache is None:
      kv_cache = torch.cat(
        (
          key_layer.unsqueeze(0).unsqueeze(0),
          value_layer.unsqueeze(0).unsqueeze(0),
        ),
        dim=1,
      )
    else:
      kv_cache = (key_layer, value_layer)
  else:
    kv_cache = None

  #   )
  #   )
  #   )
  #   )

  # ==================================

  # ==================================

  context_layer = self.core_attention(
    query_layer, key_layer, value_layer, attention_mask
  )

  # =================
  # =================

  output = self.dense(context_layer)

  return output, kv_cache
