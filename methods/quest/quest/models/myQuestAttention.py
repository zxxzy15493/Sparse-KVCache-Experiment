import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv

import quest.utils
import quest.global_time as global_time

class QuestAttention(nn.Module):
  """Multi-headed attention from 'Attention Is All You Need' paper"""

  def __init__(self, config: LlamaConfig, layer_idx: int):
    super().__init__()
    self.layer_idx = layer_idx
    self.config = config
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.pretraining_tp = getattr(config, "pretraining_tp", 1)
    self.max_position_embeddings = config.max_position_embeddings

    if (self.head_dim * self.num_heads) != self.hidden_size:
      raise ValueError(
        f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
        f" and `num_heads`: {self.num_heads})."
      )
    attn_bias = getattr(config, "attention_bias", "Qwen2" in type(config).__name__)
    self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=attn_bias)
    self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attn_bias)
    self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attn_bias)
    self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
    self._init_rope()

  def _init_rope(self):
    self.rope_theta = getattr(self.config, "rope_theta", 1e4)
    if self.config.rope_scaling is None:
      self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
      self.rope_scale = 1.0
    else:
      scaling_type = self.config.rope_scaling.get("type", self.config.rope_scaling.get("rope_type", "default"))
      if scaling_type in {"default", "linear", "llama3"}:
        self.rope_scale = self.config.rope_scaling.get("factor", 1.0)
      else:
        raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

  def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
    return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

  def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    iController: Optional[quest.utils.InferenceController] = None,
  ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    phase = "prefill" if q_len > 1 else "decode"

    assert bsz == 1, "QuestAttention only supports batch size 1."
    assert hasattr(self, 'layer_idx'), "QuestAttention requires layer_idx to inference."

    if self.pretraining_tp > 1:
      key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
      query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
      key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
      value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

      query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
      query_states = torch.cat(query_states, dim=-1)

      key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
      key_states = torch.cat(key_states, dim=-1)

      value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
      value_states = torch.cat(value_states, dim=-1)

    else:
      torch.cuda.nvtx.range_push("qkv_proj")
      query_states = self.q_proj(hidden_states)
      key_states = self.k_proj(hidden_states)
      value_states = self.v_proj(hidden_states)
      torch.cuda.nvtx.range_pop()
    
    query_states = query_states.view(q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(q_len, self.num_key_value_heads, self.head_dim)

    pad_heads = iController._pad_heads
    if pad_heads > 0:
      query_states = F.pad(query_states, (0, 0, 0, pad_heads))
    kernel_dtype = iController.dtype
    output_dtype = hidden_states.dtype
    if query_states.dtype != kernel_dtype:
      query_states = query_states.to(kernel_dtype)
      key_states = key_states.to(kernel_dtype)
      value_states = value_states.to(kernel_dtype)

    torch.cuda.nvtx.range_push("RoPE")
    quest.utils.apply_rope_in_place(
      query_states,
      key_states,
      iController.kv_cache.seqlen - q_len,
      rope_scale=self.rope_scale,
      rope_theta=self.rope_theta,
    )
    torch.cuda.nvtx.range_pop()
    global_time.stage_end(f"{phase}_pre_ffn", self.layer_idx)
    torch.cuda.nvtx.range_push("append_kv")
    
    quest.utils.append_kv(
      key_states,
      value_states,
      iController,
      self.layer_idx,
    )
    torch.cuda.nvtx.range_pop()
    if q_len > 1:
      global_time.stage_begin("prefill_attn", self.layer_idx)
      torch.cuda.nvtx.range_push("prefill_attn")
      attn_output = quest.utils.prefill_forward(
        query_states,
        iController,
        self.layer_idx,
      )
      torch.cuda.nvtx.range_pop()
      global_time.stage_end("prefill_attn", self.layer_idx)
    else:
      if iController.need_estimate() == False:
        torch.cuda.nvtx.range_push("full_attn")
        attn_output = quest.utils.decode_sparse_attn(
          query_states,
          iController,
          self.layer_idx,
          iController.kv_indices_without_last,
        )
        torch.cuda.nvtx.range_pop()
      else:
        torch.cuda.synchronize()
        global_time.stage_begin("decode_retrieve", self.layer_idx)
        torch.cuda.nvtx.range_push("estimate")
        estimated_attn_score = quest.utils.decode_estimate(
          query_states,
          iController,
          self.layer_idx,
        )
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("topk")
        quest.utils.decode_topk(
          estimated_attn_score,
          iController,
        )
        torch.cuda.nvtx.range_pop()
        global_time.stage_end("decode_retrieve", self.layer_idx)
        torch.cuda.synchronize()
        global_time.stage_begin("decode_attn", self.layer_idx)
        torch.cuda.nvtx.range_push("approx_attn")
        attn_output = quest.utils.decode_sparse_attn(
          query_states,
          iController,
          self.layer_idx,
          iController.topk_dindices_buffer,
        )
        torch.cuda.nvtx.range_pop()
        global_time.stage_end("decode_attn", self.layer_idx)
        torch.cuda.synchronize()
    global_time.stage_begin(f"{phase}_post_ffn", self.layer_idx)
    attn_output = attn_output.unsqueeze(0) # unsqueeze the batch dimension
    if pad_heads > 0:
      attn_output = attn_output[:, :, :self.num_heads, :]
    if attn_output.size() != (bsz, q_len, self.num_heads, self.head_dim):
      raise ValueError(
        f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
        f" {attn_output.size()}"
      )
    if attn_output.dtype != output_dtype:
      attn_output = attn_output.to(output_dtype)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    torch.cuda.nvtx.range_push("o_proj")
    if self.pretraining_tp > 1:
      attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
      o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
      attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
    else:
      attn_output = self.o_proj(attn_output)
    torch.cuda.nvtx.range_pop()

    if not output_attentions:
      attn_weights = None

    return attn_output, attn_weights, past_key_value
