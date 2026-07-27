#
#
#

import math
import warnings
from contextlib import nullcontext
from typing import List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from einops import rearrange





blocks_num=[]
_FLEXPREFILL_BREAKDOWN_TIME_MANAGER = None


def _breakdown_measure(component: str):
  manager = _FLEXPREFILL_BREAKDOWN_TIME_MANAGER
  return manager.measure(component) if manager is not None else nullcontext()

def gpu_info():
  if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0).lower()
    device_capability = torch.cuda.get_device_capability()
    major, minor = device_capability
    return device_name, major
  return None, None


GPU_NAME, GPU_MAJOR = gpu_info()


def get_num_warps_stages(head_dim, block_size, gpu_name):
  """
  Returns recommended num_warps and num_stages for a Sparse Attention kernel in Triton.

  Args:
    head_dim (int): Size of the head dimension.
    block_size (int): Size of the block in the attention matrix.
    gpu_name (str): Name of the GPU.

  Returns:
    tuple: (num_warps, num_stages) recommended values.
  """
  gpu_name = gpu_name.lower()
  head_large = head_dim > 64
  block_large = block_size > 64

  if "h100" in gpu_name:
    if head_large and block_large:
      num_warps = 8
      num_stages = 3
    elif head_large or block_large:
      num_warps = 4
      num_stages = 3
    else:
      num_warps = 2
      num_stages = 2
  elif "a100" in gpu_name:
    if head_large and block_large:
      num_warps = 8
      num_stages = 3
    elif head_large or block_large:
      num_warps = 8
      num_stages = 3
    else:
      num_warps = 2
      num_stages = 2
  elif "4090" in gpu_name:
    if head_large and block_large:
      num_warps = 8
      num_stages = 2
    elif head_large or block_large:
      num_warps = 8
      num_stages = 3
    else:
      num_warps = 2
      num_stages = 2
  else:
    if head_large and block_large:
      num_warps = 8
      num_stages = 2
    elif head_large or block_large:
      num_warps = 4
      num_stages = 3
    else:
      num_warps = 2
      num_stages = 2
  if head_dim > 128:
    num_stages = 2
  return num_warps, num_stages


@triton.jit
def prefill_kernel(
  q_ptr, # Q: b x n x h x d
  k_ptr, # K: b x n x h x d
  v_ptr, # V: b x n x h x d
  o_ptr,
  BATCH_SIZE,
  NUM_HEADS,
  NUM_KV_HEADS,
  NUM_SHARE_Q_HEADS,
  Q_LEN,
  K_LEN,
  HEAD_DIM: tl.constexpr,
  softmax_scale,
  causal,
  gqa_interleave,
  stride_qb,
  stride_qn,
  stride_qh,
  stride_qd,
  stride_kb,
  stride_kn,
  stride_kh,
  stride_kd,
  stride_vb,
  stride_vn,
  stride_vh,
  stride_vd,
  stride_ob,
  stride_on,
  stride_oh,
  stride_od,
  BLOCK_SIZE_Q: tl.constexpr, # q block size
  BLOCK_SIZE_K: tl.constexpr, # k block size
):
  pid_q = tl.program_id(0)
  pid_bh = tl.program_id(1)
  pid_b = pid_bh // NUM_HEADS
  pid_h = pid_bh % NUM_HEADS
  if gqa_interleave:
    pid_kh = pid_h % NUM_KV_HEADS
  else:
    pid_kh = pid_h // NUM_SHARE_Q_HEADS
  q_ptrs = tl.make_block_ptr(
    base=q_ptr + pid_b * stride_qb + pid_h * stride_qh,
    shape=(Q_LEN, HEAD_DIM),
    strides=(stride_qn, stride_qd),
    offsets=(pid_q * BLOCK_SIZE_Q, 0),
    block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
    order=(1, 0),
  )
  k_ptrs = tl.make_block_ptr(
    base=k_ptr + pid_b * stride_kb + pid_kh * stride_kh,
    shape=(HEAD_DIM, K_LEN),
    strides=(stride_kd, stride_kn),
    offsets=(0, 0),
    block_shape=(HEAD_DIM, BLOCK_SIZE_K),
    order=(0, 1),
  )
  v_ptrs = tl.make_block_ptr(
    base=v_ptr + pid_b * stride_vb + pid_kh * stride_vh,
    shape=(K_LEN, HEAD_DIM),
    strides=(stride_vn, stride_vd),
    offsets=(0, 0),
    block_shape=(BLOCK_SIZE_K, HEAD_DIM),
    order=(1, 0),
  )
  q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
  off_m = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q
  off_n = tl.arange(0, BLOCK_SIZE_K)
  m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
  lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
  acc_o = tl.full((BLOCK_SIZE_Q, HEAD_DIM), 0, dtype=tl.float32)
  lo = 0
  if causal:
    hi = min(K_LEN, (pid_q + 1) * BLOCK_SIZE_Q)
  else:
    hi = K_LEN
  for i in range(lo, hi, BLOCK_SIZE_K):
    i = tl.multiple_of(i, BLOCK_SIZE_K)
    k = tl.load(k_ptrs, boundary_check=(1,), padding_option="zero")
    qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
    if causal:
      qk += tl.where(off_m[:, None] >= (i + off_n)[None, :], 0, float("-inf"))
    else:
      qk += tl.where((off_n < K_LEN - i)[None, :], 0, float("-inf"))
    qk += tl.dot(q, k) * softmax_scale
    m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
    p = tl.math.exp2(qk - m_ij[:, None])
    l_ij = tl.sum(p, axis=1)
    acc_o_scale = tl.math.exp2(m_i - m_ij)
    acc_o = acc_o * acc_o_scale[:, None]
    v = tl.load(v_ptrs, boundary_check=(0,), padding_option="zero")
    p = p.to(v.dtype)
    acc_o += tl.dot(p, v)
    m_i = m_ij
    lse_i = m_ij + tl.math.log2(tl.math.exp2(lse_i - m_ij) + l_ij)
    k_ptrs = tl.advance(k_ptrs, (0, BLOCK_SIZE_K))
    v_ptrs = tl.advance(v_ptrs, (BLOCK_SIZE_K, 0))
  acc_o = acc_o * tl.math.exp2(m_i - lse_i)[:, None]
  o_ptrs = tl.make_block_ptr(
    base=o_ptr + pid_b * stride_ob + pid_h * stride_oh,
    shape=(Q_LEN, HEAD_DIM),
    strides=(stride_on, stride_od),
    offsets=(pid_q * BLOCK_SIZE_Q, 0),
    block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
    order=(1, 0),
  )
  tl.store(o_ptrs, acc_o.to(tl.bfloat16), boundary_check=(0,))


def triton_flash_prefill(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  causal: bool = True,
  softmax_scale: Optional[float] = None,
  gqa_interleave: bool = False,
):
  batch_size, q_len, num_q_heads, head_dim = q.shape
  batch_size, k_len, num_kv_heads, head_dim = k.shape
  assert v.shape == k.shape
  assert q.dtype == torch.bfloat16, "only support dtype bfloat16"
  assert head_dim in {16, 32, 64, 128}, "only support head_dim in {16, 32, 64, 128}"
  assert num_q_heads % num_kv_heads == 0
  num_share_q_heads = num_q_heads // num_kv_heads
  if softmax_scale is None:
    softmax_scale = 1 / math.sqrt(head_dim) * math.log2(math.e)
  else:
    softmax_scale = softmax_scale * math.log2(math.e)
  o = torch.zeros_like(q)

  grid = lambda META: (
    triton.cdiv(q_len, META["BLOCK_SIZE_Q"]),
    batch_size * num_q_heads,
  )
  BLOCK_SIZE_Q = min(
    128, max(16, triton.next_power_of_2(q_len))
  ) # min block size of tl.dot: 16
  BLOCK_SIZE_K = 128
  num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_Q, GPU_NAME)
  prefill_kernel[grid](
    q,
    k,
    v,
    o,
    batch_size,
    num_q_heads,
    num_kv_heads,
    num_share_q_heads,
    q_len,
    k_len,
    head_dim,
    softmax_scale,
    causal,
    gqa_interleave,
    q.stride(0),
    q.stride(1),
    q.stride(2),
    q.stride(3),
    k.stride(0),
    k.stride(1),
    k.stride(2),
    k.stride(3),
    v.stride(0),
    v.stride(1),
    v.stride(2),
    v.stride(3),
    o.stride(0),
    o.stride(1),
    o.stride(2),
    o.stride(3),
    BLOCK_SIZE_Q=BLOCK_SIZE_Q,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
    num_warps=num_warps,
    num_stages=num_stages,
  )
  return o


@triton.jit
def decode_kernel(
  q_ptr, # Q: b x 1 x h x d
  k_ptr, # K: b x n x h x d
  v_ptr, # V: b x n x h x d
  acco_ptr, # acc_o: b x c x h x d
  lse_ptr, # lse: b x c x h
  mi_ptr, # mi: b x c x h
  BATCH_SIZE,
  NUM_HEADS,
  NUM_KV_HEADS,
  NUM_SHARE_Q_HEADS,
  K_LEN,
  NUM_CHUNKS,
  HEAD_DIM: tl.constexpr,
  softmax_scale,
  gqa_interleave,
  stride_qb,
  stride_qn,
  stride_qh,
  stride_qd,
  stride_kb,
  stride_kn,
  stride_kh,
  stride_kd,
  stride_vb,
  stride_vn,
  stride_vh,
  stride_vd,
  stride_ob,
  stride_oc,
  stride_oh,
  stride_od,
  stride_lb,
  stride_lc,
  stride_lh,
  stride_mb,
  stride_mc,
  stride_mh,
  BLOCK_SIZE_K: tl.constexpr, # k block size
  CHUNK_SIZE_K: tl.constexpr,
):
  tl.static_assert(CHUNK_SIZE_K % BLOCK_SIZE_K == 0)
  pid_bh = tl.program_id(0)
  pid_b = pid_bh // NUM_HEADS
  pid_h = pid_bh % NUM_HEADS
  if gqa_interleave:
    pid_kh = pid_h % NUM_KV_HEADS
  else:
    pid_kh = pid_h // NUM_SHARE_Q_HEADS
  pid_c = tl.program_id(1)
  q_ptrs = (
    q_ptr
    + pid_b * stride_qb
    + pid_h * stride_qh
    + tl.arange(0, HEAD_DIM) * stride_qd
  )
  k_ptrs = tl.make_block_ptr(
    base=k_ptr + pid_b * stride_kb + pid_kh * stride_kh,
    shape=(HEAD_DIM, K_LEN),
    strides=(stride_kd, stride_kn),
    offsets=(0, pid_c * CHUNK_SIZE_K),
    block_shape=(HEAD_DIM, BLOCK_SIZE_K),
    order=(0, 1),
  )
  v_ptrs = tl.make_block_ptr(
    base=v_ptr + pid_b * stride_vb + pid_kh * stride_vh,
    shape=(K_LEN, HEAD_DIM),
    strides=(stride_vn, stride_vd),
    offsets=(pid_c * CHUNK_SIZE_K, 0),
    block_shape=(BLOCK_SIZE_K, HEAD_DIM),
    order=(1, 0),
  )
  q = tl.load(q_ptrs)
  off_n = tl.arange(0, BLOCK_SIZE_K)
  m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
  lse_i = tl.full((1,), float("-inf"), dtype=tl.float32)
  acc_o = tl.full((HEAD_DIM,), 0, dtype=tl.float32)
  lo = pid_c * CHUNK_SIZE_K
  hi = min(K_LEN, (pid_c + 1) * CHUNK_SIZE_K)
  for i in range(lo, hi, BLOCK_SIZE_K):
    i = tl.multiple_of(i, BLOCK_SIZE_K)
    k = tl.load(k_ptrs, boundary_check=(1,), padding_option="zero")
    qk = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)
    qk += tl.where((off_n < hi - i), 0, float("-inf"))
    qk += tl.sum(q[:, None] * k, axis=0) * softmax_scale
    m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
    p = tl.math.exp2(qk - m_ij)
    l_ij = tl.sum(p, axis=0)
    acc_o_scale = tl.math.exp2(m_i - m_ij)
    acc_o = acc_o * acc_o_scale
    v = tl.load(v_ptrs, boundary_check=(0,), padding_option="zero")
    p = p.to(v.dtype)
    acc_o += tl.sum(p[:, None] * v, axis=0)
    m_i = m_ij
    lse_i = m_ij + tl.math.log2(tl.math.exp2(lse_i - m_ij) + l_ij)
    k_ptrs = tl.advance(k_ptrs, (0, BLOCK_SIZE_K))
    v_ptrs = tl.advance(v_ptrs, (BLOCK_SIZE_K, 0))
  lse_ptr = (
    lse_ptr
    + pid_b * stride_lb
    + pid_h * stride_lh
    + (pid_c + tl.arange(0, 1)) * stride_lc
  )
  tl.store(lse_ptr, lse_i)
  mi_ptr = (
    mi_ptr
    + pid_b * stride_mb
    + pid_h * stride_mh
    + (pid_c + tl.arange(0, 1)) * stride_mc
  )
  tl.store(mi_ptr, m_i)
  off_d = tl.arange(0, HEAD_DIM)
  o_ptrs = (
    acco_ptr
    + pid_b * stride_ob
    + pid_c * stride_oc
    + pid_h * stride_oh
    + off_d * stride_od
  )
  tl.store(o_ptrs, acc_o)


@triton.jit
def rescale_kernel(
  acco_ptr, # acc_o: b x c x h x d
  o_ptr, # o: b x 1 x h x d
  lse_ptr, # lse: b x c x h
  mi_ptr, # mi: b x c x h
  BATCH_SIZE,
  NUM_HEADS,
  NUM_CHUNKS,
  HEAD_DIM: tl.constexpr,
  stride_ab,
  stride_ac,
  stride_ah,
  stride_ad,
  stride_ob,
  stride_on,
  stride_oh,
  stride_od,
  stride_lb,
  stride_lc,
  stride_lh,
  stride_mb,
  stride_mc,
  stride_mh,
  BLOCK_SIZE_D: tl.constexpr,
  BLOCK_SIZE_C: tl.constexpr,
):
  pid_bh = tl.program_id(0)
  pid_b = pid_bh // NUM_HEADS
  pid_h = pid_bh % NUM_HEADS
  off_chunks = tl.arange(0, BLOCK_SIZE_C)
  mi_ptrs = mi_ptr + pid_b * stride_mb + pid_h * stride_mh + off_chunks * stride_mc
  lse_ptrs = lse_ptr + pid_b * stride_lb + pid_h * stride_lh + off_chunks * stride_lc
  acco_ptrs = tl.make_block_ptr(
    base=acco_ptr + pid_b * stride_ab + pid_h * stride_ah,
    shape=(NUM_CHUNKS, HEAD_DIM),
    strides=(stride_ac, stride_ad),
    offsets=(0, 0),
    block_shape=(BLOCK_SIZE_C, BLOCK_SIZE_D),
    order=(1, 0),
  )
  o_ptrs = tl.make_block_ptr(
    base=o_ptr + pid_b * stride_ob + pid_h * stride_oh,
    shape=(1, HEAD_DIM),
    strides=(stride_on, stride_od),
    offsets=(0, 0),
    block_shape=(1, BLOCK_SIZE_D),
    order=(1, 0),
  )
  mi = tl.load(mi_ptrs, mask=off_chunks < NUM_CHUNKS, other=float("-inf"))
  lse = tl.load(lse_ptrs, mask=off_chunks < NUM_CHUNKS, other=float("-inf"))
  m = tl.max(mi, axis=0)
  scale = tl.math.exp2(mi - m) / tl.sum(tl.math.exp2(lse - m), axis=0)
  o = tl.full((HEAD_DIM,), 0, dtype=tl.float32)
  for i in range(0, HEAD_DIM, BLOCK_SIZE_D):
    i = tl.multiple_of(i, BLOCK_SIZE_D)
    acco = tl.load(acco_ptrs, boundary_check=(0, 1), padding_option="zero")
    acco = tl.sum(acco * scale[:, None], axis=0)[None, :]
    tl.store(o_ptrs, acco.to(tl.bfloat16), boundary_check=(0, 1))
    acco_ptrs = tl.advance(acco_ptrs, (0, BLOCK_SIZE_D))
    o_ptrs = tl.advance(o_ptrs, (0, BLOCK_SIZE_D))


def triton_flash_decode(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  softmax_scale: Optional[float] = None,
  gqa_interleave: bool = False,
):
  batch_size, q_len, num_q_heads, head_dim = q.shape
  batch_size, k_len, num_kv_heads, head_dim = k.shape
  assert q_len == 1
  assert v.shape == k.shape
  assert q.dtype == torch.bfloat16, "only support dtype bfloat16"
  assert head_dim in {16, 32, 64, 128}, "only support head_dim in {16, 32, 64, 128}"
  if softmax_scale is None:
    softmax_scale = 1 / math.sqrt(head_dim) * math.log2(math.e)
  else:
    softmax_scale = softmax_scale * math.log2(math.e)
  assert num_q_heads % num_kv_heads == 0
  num_share_q_heads = num_q_heads // num_kv_heads
  grid = lambda META: (
    batch_size * num_q_heads, # batch & head
    triton.cdiv(k_len, META["CHUNK_SIZE_K"]), # k chunks
  )
  BLOCK_SIZE_K = 128
  CHUNK_SIZE_K = 4096
  num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_K, GPU_NAME)
  num_chunks = triton.cdiv(k_len, CHUNK_SIZE_K)
  lse = torch.empty(
    batch_size, num_chunks, num_q_heads, dtype=torch.float32, device=q.device
  )
  mi = torch.empty(
    batch_size, num_chunks, num_q_heads, dtype=torch.float32, device=q.device
  )
  acc_o = torch.empty(
    batch_size,
    num_chunks,
    num_q_heads,
    head_dim,
    dtype=torch.float32,
    device=q.device,
  )
  decode_kernel[grid](
    q,
    k,
    v,
    acc_o,
    lse,
    mi,
    batch_size,
    num_q_heads,
    num_kv_heads,
    num_share_q_heads,
    k_len,
    num_chunks,
    head_dim,
    softmax_scale,
    gqa_interleave,
    q.stride(0),
    q.stride(1),
    q.stride(2),
    q.stride(3),
    k.stride(0),
    k.stride(1),
    k.stride(2),
    k.stride(3),
    v.stride(0),
    v.stride(1),
    v.stride(2),
    v.stride(3),
    acc_o.stride(0),
    acc_o.stride(1),
    acc_o.stride(2),
    acc_o.stride(3),
    lse.stride(0),
    lse.stride(1),
    lse.stride(2),
    mi.stride(0),
    mi.stride(1),
    mi.stride(2),
    BLOCK_SIZE_K=BLOCK_SIZE_K,
    CHUNK_SIZE_K=CHUNK_SIZE_K,
    num_warps=num_warps,
    num_stages=num_stages,
  )
  o = torch.empty(
    batch_size,
    1,
    num_q_heads,
    head_dim,
    dtype=q.dtype,
    device=q.device,
  )
  grid = lambda META: (batch_size * num_q_heads,) # batch & head
  BLOCK_SIZE_C = triton.next_power_of_2(num_chunks)
  BLOCK_SIZE_D = min(head_dim, 128 * 128 // BLOCK_SIZE_C)
  num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_K, GPU_NAME)
  rescale_kernel[grid](
    acc_o,
    o,
    lse,
    mi,
    batch_size,
    num_q_heads,
    num_chunks,
    head_dim,
    acc_o.stride(0),
    acc_o.stride(1),
    acc_o.stride(2),
    acc_o.stride(3),
    o.stride(0),
    o.stride(1),
    o.stride(2),
    o.stride(3),
    lse.stride(0),
    lse.stride(1),
    lse.stride(2),
    mi.stride(0),
    mi.stride(1),
    mi.stride(2),
    BLOCK_SIZE_D=BLOCK_SIZE_D,
    BLOCK_SIZE_C=BLOCK_SIZE_C,
    num_warps=num_warps,
    num_stages=num_stages,
  )
  return o


def triton_flash_attention(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  causal: bool = True,
  softmax_scale: Optional[float] = None,
  gqa_interleave: bool = False,
):
  batch_size, q_len, num_heads, head_dim = q.shape
  batch_size, k_len, num_heads, head_dim = k.shape
  assert v.shape == k.shape
  assert q.dtype == torch.bfloat16, "only support dtype bfloat16"
  assert head_dim in {16, 32, 64, 128}, "only support head_dim in {16, 32, 64, 128}"
  if q_len > 1:
    return triton_flash_prefill(q, k, v, causal, softmax_scale, gqa_interleave)
  else:
    return triton_flash_decode(q, k, v, softmax_scale, gqa_interleave)


def torch_block_wise_attention(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  block_idx: torch.Tensor,
  block_size: int,
  grid_offset: int = 0,
):

  b, n, h, d = q.shape
  assert k.shape == q.shape
  assert v.shape == k.shape
  num_block = math.ceil(grid_offset / block_size) + math.ceil(
    (n - grid_offset) / block_size
  )
  mask = torch.zeros(b, h, num_block, num_block, dtype=torch.bool, device=q.device)
  mask[
    torch.arange(b).view(b, 1, 1).expand(b, h, block_idx.shape[-1]),
    torch.arange(h).view(1, h, 1).expand(b, h, block_idx.shape[-1]),
    block_idx // num_block,
    block_idx % num_block,
  ] = 1
  act_blocks_per_row = torch.tril(mask).sum(-1)
  mask = mask.repeat_interleave(block_size, -2).repeat_interleave(block_size, -1)
  mask = mask[..., grid_offset : grid_offset + n, grid_offset : grid_offset + n]
  mask = torch.tril(mask)
  attn_weight = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(d)
  attn_weight.masked_fill_(~mask, float("-inf"))
  attn_weight = torch.softmax(attn_weight, dim=-1)
  o = torch.einsum("bhij,bjhd->bhid", attn_weight, v)
  o = o.transpose(1, 2)
  return o




@torch.no_grad()
def torch_full_attention_weights(
  q: torch.Tensor,           # [B, Q, Hq, D]
  k: torch.Tensor,           # [B, K, Hkv, D]
  v: Optional[torch.Tensor] = None,   # [B, K, Hkv, D]
  *,
  causal: bool = True,
  gqa_interleave: bool = False,
  softmax_scale: Optional[float] = None,
  q_start: int = 0,
  q_end: Optional[int] = None,
  return_output: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
  """
  Returns:
   attn_weight: [B, Hq, Q', K] (float32 probabilities)
   (optional) attn_out: [B, Q', Hq, D] (same dtype as q)
  where Q' = q_end - q_start

  Notes:
  - Prefill self-attention typically has Q==K and causal=True
  - Supports GQA: Hq may be larger than Hkv, K/V are expanded to Hq per gqa_interleave
  - Attention weight tensors are very large; for long context, compute only the last segment: q_start = Q - block_size
  """
  assert q.dim() == 4 and k.dim() == 4
  B, Q, Hq, D = q.shape
  Bk, K, Hkv, Dk = k.shape
  assert B == Bk and D == Dk, f"shape mismatch: q={q.shape}, k={k.shape}"
  if v is not None:
    assert v.shape == k.shape, f"v must have same shape as k, got v={v.shape}, k={k.shape}"
  assert Hq % Hkv == 0, f"GQA requires Hq%Hkv==0, got Hq={Hq}, Hkv={Hkv}"
  num_share_q_heads = Hq // Hkv

  if q_end is None:
    q_end = Q
  q_start = max(0, q_start)
  q_end = min(Q, q_end)
  assert q_start < q_end, f"invalid slice: q_start={q_start}, q_end={q_end}"

  q_slice = q[:, q_start:q_end, :, :] # [B, Q', Hq, D]
  Qp = q_slice.shape[1]


  if not gqa_interleave:

    k_exp = k.repeat_interleave(num_share_q_heads, dim=2) # [B, K, Hq, D]
    v_exp = v.repeat_interleave(num_share_q_heads, dim=2) if v is not None else None
  else:

    k_exp = k.repeat(1, 1, num_share_q_heads, 1)      # [B, K, Hq, D]
    v_exp = v.repeat(1, 1, num_share_q_heads, 1) if v is not None else None

  scale = (1.0 / math.sqrt(D)) if softmax_scale is None else float(softmax_scale)

  attn_logits = torch.einsum("bqhd,bkhd->bhqk", q_slice, k_exp) * scale

  if causal:

    q_pos = torch.arange(q_start, q_start + Qp, device=q.device) # [Q']
    k_pos = torch.arange(K, device=q.device)           # [K]
    causal_mask = (k_pos[None, :] <= q_pos[:, None])       # [Q', K]
    attn_logits = attn_logits.masked_fill(~causal_mask[None, None, :, :], float("-inf"))


  attn_weight = torch.softmax(attn_logits, dim=-1, dtype=torch.float32) # [B, Hq, Q', K]




  if not return_output:
    return attn_weight

  if v_exp is None:
    raise ValueError("return_output=True requires v is not None")

  out = torch.einsum("bhqk,bkhd->bqhd", attn_weight.to(v_exp.dtype), v_exp)
  out = out.to(q.dtype)
  return attn_weight, out


def attn_weight_k_topk(
  attn_weight: torch.Tensor,
  topk: int,
  *,
  reduce_heads: Optional[str] = None,  # None / "mean" / "sum" / "max"
  reduce_batch: Optional[str] = None,  # None / "mean" / "sum" / "max"
  return_scores: bool = True,
) -> Union[
  torch.Tensor,
  Tuple[torch.Tensor, torch.Tensor],
  Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
  """
  Compute the per-key "average attention score":
   k_score = mean_Q(attn_weight)

  Returns:
   - topk_indices: (..., topk) key positions of top-k in the last dim
   - topk_values: (..., topk) corresponding scores (optional)
   - k_scores:   (..., K)   scores for all keys (optional)

  reduce_heads/reduce_batch:
   - None: no aggregation, output separated by (B,H), each gets topk
   - "mean"/"sum"/"max": aggregate across dimension then take topk
  """
  assert attn_weight.dim() == 4, f"expect [B,H,Q,K], got {attn_weight.shape}"
  B, H, Q, K = attn_weight.shape
  assert 1 <= topk <= K, f"topk must be in [1, K], got topk={topk}, K={K}"


  k_scores = attn_weight.mean(dim=2)


  if reduce_heads is not None:
    if reduce_heads == "mean":
      k_scores = k_scores.mean(dim=1)   # [B, K]
    elif reduce_heads == "sum":
      k_scores = k_scores.sum(dim=1)   # [B, K]
    elif reduce_heads == "max":
      k_scores = k_scores.max(dim=1).values
    else:
      raise ValueError(f"reduce_heads must be None/mean/sum/max, got {reduce_heads}")


  if reduce_batch is not None:
    if reduce_batch == "mean":
      k_scores = k_scores.mean(dim=0)
    elif reduce_batch == "sum":
      k_scores = k_scores.sum(dim=0)
    elif reduce_batch == "max":
      k_scores = k_scores.max(dim=0).values
    else:
      raise ValueError(f"reduce_batch must be None/mean/sum/max, got {reduce_batch}")


  topk_values, topk_indices = torch.topk(k_scores, k=topk, dim=-1, largest=True, sorted=True)

  if not return_scores:
    return topk_indices

  return topk_indices, topk_values, k_scores


def _blocks_to_token_indices(
  block_idx: torch.Tensor,  # [..., n_blocks]
  block_size: int,
  K: int,
) -> torch.Tensor:
  """
  Expand block indices to token indicestoken = block_idx*block_size + offset
  Returns: [..., n_blocks*block_size]
  """
  block_idx = block_idx.to(torch.long)
  offsets = torch.arange(block_size, device=block_idx.device, dtype=torch.long) # [bs]
  token_idx = block_idx[..., None] * block_size + offsets            # [..., n_blocks, bs]
  token_idx = token_idx.reshape(*block_idx.shape[:-1], -1)            # [..., topk]



  token_idx = token_idx.masked_fill((token_idx < 0) | (token_idx >= K), K)
  return token_idx


BlockIdxType = Union[
  torch.Tensor,         # [B, H, Kmax] (padded)
  List[List[torch.Tensor]],   # block_idx[b][h] = 1D LongTensor (ragged)
]


def _as_ragged_block_idx(block_idx: BlockIdxType) -> List[List[torch.Tensor]]:
  """Normalize block_idx to ragged list[b][h] -> 1D LongTensor."""
  if isinstance(block_idx, torch.Tensor):
    assert block_idx.dim() == 3, "block_idx Tensor must be [B, H, Kmax]"
    B, H, _ = block_idx.shape
    out: List[List[torch.Tensor]] = []
    for b in range(B):
      row = []
      for h in range(H):
        row.append(block_idx[b, h].to(dtype=torch.long))
      out.append(row)
    return out

  assert isinstance(block_idx, list) and isinstance(block_idx[0], list)
  return [[t.to(dtype=torch.long) for t in row] for row in block_idx]



def effective_k_token_count_from_block_idx(
  block_idx: BlockIdxType,
  q_len: int,
  block_size: int,
  kv_len: Optional[int] = None,
) -> torch.Tensor:
  """
  Compute for each query token the"effective key token count"satisfying k_pos <= q_pos

  Args:
    block_idx:
     - ragged: block_idx[b][h] = 1D tensor of flatten indices
     - or padded: [B,H,Kmax]
     flatten index: idx = q_block * num_k_blocks + k_block
    q_len: query token length
    block_size: block size
    kv_len: key length (default: same as q_len)

  Returns:
    counts: LongTensor [B, H, q_len]
  """
  if kv_len is None:
    kv_len = q_len

  num_k_blocks = math.ceil(kv_len / block_size)
  num_q_blocks = math.ceil(q_len / block_size)

  ragged = _as_ragged_block_idx(block_idx)
  B = len(ragged)
  H = len(ragged[0]) if B > 0 else 0

  device = ragged[0][0].device if (B > 0 and H > 0) else torch.device("cpu")
  counts = torch.zeros((B, H, q_len), dtype=torch.long, device=device)


  block_lens = torch.full((num_k_blocks,), block_size, dtype=torch.long, device=device)
  last_len = kv_len - (num_k_blocks - 1) * block_size
  if num_k_blocks > 0:
    block_lens[-1] = last_len if last_len > 0 else block_size

  for b in range(B):
    for h in range(H):
      idx = ragged[b][h]
      if idx.numel() == 0:
        continue

      qb = idx // num_k_blocks
      kb = idx % num_k_blocks


      valid = (qb >= 0) & (qb < num_q_blocks) & (kb >= 0) & (kb < num_k_blocks)
      qb = qb[valid]
      kb = kb[valid]
      if qb.numel() == 0:
        continue


      uniq_qb = torch.unique(qb)
      for qbi in uniq_qb.tolist():
        sel_kb = kb[qb == qbi]
        if sel_kb.numel() == 0:
          continue
        sel_kb = torch.unique(sel_kb)


        sel_kb = sel_kb[sel_kb <= qbi]
        if sel_kb.numel() == 0:
          continue


        kb_before = sel_kb[sel_kb < qbi]
        count_before = block_lens[kb_before].sum() if kb_before.numel() > 0 else 0


        has_self = (sel_kb == qbi).any()

        q_start = qbi * block_size
        q_end = min(q_start + block_size, q_len)
        q_block_len = q_end - q_start
        if q_block_len <= 0:
          continue

        if has_self:
          offsets = torch.arange(1, q_block_len + 1, device=device, dtype=torch.long)
          counts[b, h, q_start:q_end] = count_before + offsets
        else:
          counts[b, h, q_start:q_end] = count_before

  return counts


def per_query_true_topk_indices(
  attn_weight: torch.Tensor,   # [B, H, Q, K]
  num_top_k: torch.Tensor,    # [B, H, Q] (int)
  pad_val: int = -1,
) -> torch.Tensor:
  """
  Returns top-k key indices per query (variable budget, padding-aligned).
  Output shape: [B, H, Q, Kmax]where Kmax = num_top_k.max()
  Each query only has effective values in the first num_top_k[b,h,q] positions, rest are pad_val.
  """
  assert attn_weight.dim() == 4, "attn_weight must be [B, H, Q, K]"
  B, H, Q, K = attn_weight.shape

  if num_top_k.dtype != torch.long:
    num_top_k = num_top_k.to(torch.long)
  assert num_top_k.shape == (B, H, Q), "num_top_k must be [B, H, Q]"


  num_top_k = num_top_k.clamp(min=0, max=K)

  Kmax = int(num_top_k.max().item())
  if Kmax == 0:
    return torch.full((B, H, Q, 0), pad_val, device=attn_weight.device, dtype=torch.long)


  _, topk_idx = torch.topk(attn_weight, k=Kmax, dim=-1, largest=True, sorted=True) # [B,H,Q,Kmax]


  ar = torch.arange(Kmax, device=attn_weight.device)[None, None, None, :]     # [1,1,1,Kmax]
  mask = ar < num_top_k[..., None]                         # [B,H,Q,Kmax]

  topk_idx = topk_idx.masked_fill(~mask, pad_val)
  return topk_idx




def caculate_racall(
  q: torch.Tensor,           # [B, Q, Hq, D]
  k: torch.Tensor,           # [B, K, Hkv, D]
  block_size:int,
  flex_block_idx: torch.Tensor,
  v: Optional[torch.Tensor] = None,   # [B, K, Hkv, D]
):
  batch_size, seq_len, num_heads, head_dim = q.shape



  attn_weight=torch_full_attention_weights(q,k,v)

  print("111")



  num_top_k = effective_k_token_count_from_block_idx(block_idx=flex_block_idx,
  q_len=seq_len,
  block_size=block_size,
  kv_len=seq_len,)
  print("111")



def _as_ragged_block_idx(block_idx: BlockIdxType) -> List[List[torch.Tensor]]:
  if isinstance(block_idx, torch.Tensor):
    assert block_idx.dim() == 3, "flex_block_idx tensor must be [B,H,Kmax]"
    B, H, _ = block_idx.shape
    out = []
    for b in range(B):
      row = []
      for h in range(H):
        row.append(block_idx[b, h].to(torch.long))
      out.append(row)
    return out
  return [[t.to(torch.long) for t in row] for row in block_idx]

@torch.no_grad()
def calculate_flexprefill_layer_recall(
  model_name:str,
  layerid:int,
  q: torch.Tensor,           # [B, Q, H, D]
  k: torch.Tensor,           # [B, K, Hkv, D]
  block_size: int,
  flex_block_idx: BlockIdxType,
  block_sparse_mask:torch.Tensor,
  min_budget:int,
  gamma:float,
  tau:float,
  v: Optional[torch.Tensor] = None,   # [B, K, Hkv, D]
  pad_val: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """
  By definition:
   1) Use flex_block_idx to get budget num_top_k per (b,h,q) (= actual visible tokens for this query)
   2) Use full attention attn_weight to get top-num_top_k indices per (b,h,q)
   3) recall(b,h,q) = |indices2 ∩ indices1| / num_top_k
   4) layer_recall = mean_{b,h,q} recall(b,h,q)

  Returns
   layer_recall: scalar
   head_recall: [H], averaged per head over (B,Q)
   recall_per_query: [B,H,Q]
  """


  attn_weight = torch_full_attention_weights(q, k, v)

  B, H, Q, K = attn_weight.shape
  kv_len = K



  num_top_k = effective_k_token_count_from_block_idx(
    block_idx=flex_block_idx,
    q_len=Q,
    block_size=block_size,
    kv_len=K,
  ) # [B,H,Q], long/int

  num_top_k = num_top_k.clamp(min=0, max=K).to(torch.long)
  Kmax = int(num_top_k.max().item())
  Kmax = min(Kmax, K)
  if Kmax == 0:
    recall_per_query = torch.zeros((B, H, Q), device=attn_weight.device, dtype=torch.float32)
    del attn_weight
    torch.cuda.empty_cache()
    head_recall = recall_per_query.mean(dim=(0, 2))
    layer_recall = recall_per_query.mean()
    return layer_recall, head_recall, recall_per_query


  _, true_topk_idx = torch.topk(attn_weight, k=Kmax, dim=-1, largest=True, sorted=True) # [B,H,Q,Kmax]
  del attn_weight
  torch.cuda.empty_cache()
  ar = torch.arange(Kmax, device=true_topk_idx.device)[None, None, None, :]
  true_valid = ar < num_top_k[..., None]          # [B,H,Q,Kmax]
  true_topk_idx = true_topk_idx.masked_fill(~true_valid, pad_val)


  num_k_blocks = math.ceil(K / block_size)
  num_q_blocks = math.ceil(Q / block_size)

  ragged = _as_ragged_block_idx(flex_block_idx)
  selected_mask = torch.zeros((B, H, num_q_blocks, num_k_blocks),
                dtype=torch.bool, device=true_topk_idx.device)



  for b in range(B):
    for h in range(H):
      idx = ragged[b][h].to(true_topk_idx.device)
      if idx.numel() == 0:
        continue
      qb = idx // num_k_blocks
      kb = idx % num_k_blocks
      valid = (qb >= 0) & (qb < num_q_blocks) & (kb >= 0) & (kb < num_k_blocks)
      qb = qb[valid]
      kb = kb[valid]
      if qb.numel() > 0:
        selected_mask[b, h, qb, kb] = True





  q_pos = torch.arange(Q, device=true_topk_idx.device)[None, None, :, None] # [1,1,Q,1]

  q_block = (q_pos // block_size).to(torch.long)              # [1,1,Q,1]



  idx2 = true_topk_idx # [B,H,Q,Kmax]
  idx2_valid = idx2 != pad_val
  k_block = (idx2.clamp(min=0) // block_size).to(torch.long)        # [B,H,Q,Kmax]

  N=num_q_blocks * num_k_blocks



  





  mask_flat = selected_mask.view(B, H, -1) # [B,H,N]

  linear = (q_block * num_k_blocks + k_block)
  linear = linear.clamp(0, num_q_blocks * num_k_blocks - 1)

  linear_flat = linear.view(B, H, -1)           # [B,H,Q*Kmax]
  in_selected_block = mask_flat.gather(-1, linear_flat)  # [B,H,Q*Kmax]
  in_selected_block = in_selected_block.view(B, H, Q, Kmax)




  causal_ok = idx2 <= q_pos.expand_as(idx2)

  hit = in_selected_block & idx2_valid & causal_ok
  hit_count = hit.sum(dim=-1)                       # [B,H,Q]



  denom = num_top_k.clamp(min=1).to(torch.float32)

  recall_per_query = hit_count.to(torch.float32) / denom          # [B,H,Q]


  head_recall = recall_per_query.mean(dim=(0, 2))             # [H]

  layer_recall = recall_per_query.mean()




  from pathlib import Path
  import json

  suffix=f'{model_name}-gamma{gamma}'
  outdir = Path("efficiency/recall-results")
  outdir.mkdir(parents=True, exist_ok=True)


  outpath = outdir / f"{suffix}.jsonl"
  outpath.parent.mkdir(parents=True, exist_ok=True)




  block_cnt = torch.zeros((B, H), dtype=torch.long)
  for b in range(B):
    for h in range(H):
      idx = ragged[b][h]
      if isinstance(idx, torch.Tensor) and idx.numel() > 0:

        block_cnt[b, h] = torch.unique(idx.to("cpu")).numel()
      else:
        block_cnt[b, h] = 0

  block_num_per_head = block_cnt.float().mean(dim=0)


  def mask_to_head_type(block_sparse_mask: torch.Tensor, H: int) -> List[str]:
    m = block_sparse_mask
    if m.dtype != torch.bool:
      m = m.bool()


    if m.dim() == 1:     # [H]
      qa = m
    elif m.dim() == 2:    # [B,H]
      qa = m.any(dim=0)
    else:
      raise ValueError(f"block_sparse_mask shape expected [H] or [B,H], got {tuple(m.shape)}")

    return ["query_aware" if qa[h].item() else "vshead" for h in range(H)]

  head_type_list = mask_to_head_type(block_sparse_mask, H)

  with open(outpath, "a", encoding="utf-8") as f:

    for h in range(H):
      record = {
        "layer": layerid,
        "head_num": int(h),
        "avg_recall_pre_head": float(head_recall[h].item()),
        "head_type": head_type_list[h],     # "vshead" / "query_aware" / "unknown"
        "q_len": int(Q),
        "block_num": float(block_num_per_head[h].item()) if B > 1 else int(block_cnt[0, h].item()),
      }
      json.dump(record, f, ensure_ascii=False)
      f.write("\n")


  del true_topk_idx, selected_mask, mask_flat, linear, linear_flat, in_selected_block, hit, hit_count, num_top_k
  torch.cuda.empty_cache()

  return layer_recall, head_recall, recall_per_query


def calculate_flexprefill_layer_captured_mass(
  save_path:str,
  model_name:str,
  layerid:int,
  q: torch.Tensor,           # [B, Q, H, D]
  k: torch.Tensor,           # [B, K, Hkv, D]
  block_size: int,
  flex_block_idx: BlockIdxType,
  block_sparse_mask:torch.Tensor,
  min_budget:int,
  gamma:float,
  tau:float,
  v: Optional[torch.Tensor] = None,   # [B, K, Hkv, D]
  pad_val: int = -1,
  sample_id: str = "",
  gqa_interleave: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:


  B, Q, H, _ = q.shape
  K = k.shape[1]
  device = q.device

  kv_len = K


  num_k_blocks = math.ceil(K / block_size)
  num_q_blocks = math.ceil(Q / block_size)

  ragged = _as_ragged_block_idx(flex_block_idx)
  selected_mask = torch.zeros((B, H, num_q_blocks, num_k_blocks),
                dtype=torch.bool, device=device)

  for b in range(B):
    for h in range(H):
      idx = ragged[b][h].to(device)
      if idx.numel() == 0:
        continue
      qb = idx // num_k_blocks
      kb = idx % num_k_blocks
      valid = (qb >= 0) & (qb < num_q_blocks) & (kb >= 0) & (kb < num_k_blocks)
      qb = qb[valid]
      kb = kb[valid]
      if qb.numel() > 0:
        selected_mask[b, h, qb, kb] = True




  Hkv = k.shape[2]
  assert H % Hkv == 0, f"GQA requires H%Hkv==0, got H={H}, Hkv={Hkv}"
  num_share_q_heads = H // Hkv
  if not gqa_interleave:
    k_exp = k.repeat_interleave(num_share_q_heads, dim=2) # [B, K, H, D]
  else:
    k_exp = k.repeat(1, 1, num_share_q_heads, 1)      # [B, K, H, D]

  scale = 1.0 / math.sqrt(q.shape[-1])
  captured_mass_sum = torch.zeros((B, H), dtype=torch.float32, device=device)
  top100_recall_sum = torch.zeros((B, H), dtype=torch.float32, device=device)
  top100_recall_count = 0
  top100_k = min(100, K)
  key_chunk_blocks = min(16, num_k_blocks)
  selected_mask_flat = selected_mask.view(B, H, -1)

  for qbi in range(num_q_blocks):
    q_start = qbi * block_size
    q_end = min(q_start + block_size, Q)
    q_block = q[:, q_start:q_end, :, :]          # [B, q_blk_len, H, D]
    q_pos = torch.arange(q_start, q_end, device=device)   # [q_blk_len]
    q_blk_len = q_end - q_start

    m_i = torch.full((B, H, q_blk_len), float("-inf"), dtype=torch.float32, device=device)
    l_i = torch.zeros((B, H, q_blk_len), dtype=torch.float32, device=device)
    selected_l_i = torch.zeros((B, H, q_blk_len), dtype=torch.float32, device=device)
    top100_valid_query = q_pos >= 100
    top100_q_pos = q_pos[top100_valid_query]
    top100_q_len = int(top100_valid_query.sum().item())
    if top100_q_len > 0:
      top100_values = torch.full((B, H, top100_q_len, top100_k), float("-inf"), dtype=torch.float32, device=device)
      top100_indices = torch.full((B, H, top100_q_len, top100_k), -1, dtype=torch.long, device=device)
    else:
      top100_values = None
      top100_indices = None

    max_kbi = min(num_k_blocks, qbi + 1)
    for kbi_start in range(0, max_kbi, key_chunk_blocks):
      kbi_end = min(kbi_start + key_chunk_blocks, max_kbi)
      k_start = kbi_start * block_size
      k_end = min(kbi_end * block_size, K)
      k_chunk = k_exp[:, k_start:k_end, :, :]       # [B, k_chunk_len, H, D]
      k_pos = torch.arange(k_start, k_end, device=device) # [k_chunk_len]

      q_mat = q_block.permute(0, 2, 1, 3)         # [B,H,q_blk_len,D]
      k_mat = k_chunk.permute(0, 2, 3, 1)         # [B,H,D,k_chunk_len]
      logits = (torch.matmul(q_mat, k_mat) * scale).to(torch.float32)

      causal_ok = k_pos[None, :] <= q_pos[:, None]    # [q_blk_len, k_chunk_len]
      logits = logits.masked_fill(~causal_ok[None, None, :, :], float("-inf"))

      block_m = logits.max(dim=-1).values         # [B,H,q_blk_len]
      m_new = torch.maximum(m_i, block_m)
      old_scale = torch.exp(m_i - m_new)
      old_scale = torch.where(torch.isfinite(m_i), old_scale, torch.zeros_like(old_scale))

      exp_logits = torch.exp(logits - m_new[..., None])
      exp_logits = torch.where(torch.isfinite(logits), exp_logits, torch.zeros_like(exp_logits))
      block_l = exp_logits.sum(dim=-1)

      if top100_q_len > 0:
        top100_logits = logits[:, :, top100_valid_query, :]
        block_topk_k = min(top100_k, top100_logits.shape[-1])
        block_top_values, block_top_local_indices = torch.topk(
          top100_logits, k=block_topk_k, dim=-1, largest=True, sorted=False
        )
        block_top_indices = block_top_local_indices + k_start
        merged_values = torch.cat((top100_values, block_top_values), dim=-1)
        merged_indices = torch.cat((top100_indices, block_top_indices), dim=-1)
        top100_values, top100_order = torch.topk(
          merged_values, k=top100_k, dim=-1, largest=True, sorted=False
        )
        top100_indices = merged_indices.gather(-1, top100_order)
        del top100_logits, block_top_values, block_top_local_indices
        del block_top_indices, merged_values, merged_indices, top100_order

      chunk_selected = selected_mask[:, :, qbi, kbi_start:kbi_end] # [B,H,key_chunk_blocks]
      if chunk_selected.any():
        token_block_offsets = (k_pos // block_size) - kbi_start
        selected_token_mask = chunk_selected[:, :, token_block_offsets]
        selected_block_l = (
          exp_logits * selected_token_mask[:, :, None, :].to(torch.float32)
        ).sum(dim=-1)
      else:
        selected_block_l = torch.zeros_like(block_l)

      l_i = l_i * old_scale + block_l
      selected_l_i = selected_l_i * old_scale + selected_block_l
      m_i = m_new

      del q_mat, k_mat, logits, exp_logits, block_l, selected_block_l

    captured_mass_per_query = selected_l_i / l_i.clamp_min(torch.finfo(torch.float32).tiny)
    captured_mass_sum += captured_mass_per_query.sum(dim=-1) # [B,H]

    if top100_q_len > 0:
      top100_k_block = (top100_indices.clamp(min=0) // block_size).clamp(max=num_k_blocks - 1)
      q_block_for_gather = torch.full_like(top100_k_block, qbi)
      top100_linear = q_block_for_gather * num_k_blocks + top100_k_block
      selected_top100 = selected_mask_flat.gather(-1, top100_linear.view(B, H, -1)).view_as(top100_indices)
      selected_top100 = selected_top100 & (top100_indices >= 0)
      top100_recall_per_query = selected_top100.to(torch.float32).sum(dim=-1) / float(top100_k)
      top100_recall_sum += top100_recall_per_query.sum(dim=-1)
      top100_recall_count += top100_q_len

    del q_block, m_i, l_i, selected_l_i, captured_mass_per_query
    del top100_q_pos
    if top100_values is not None:
      del top100_values, top100_indices

  head_captured_mass = captured_mass_sum.sum(dim=0) / (B * Q) # [H]
  if top100_recall_count > 0:
    head_top100_recall = top100_recall_sum.sum(dim=0) / (B * top100_recall_count)
  else:
    head_top100_recall = torch.zeros((H,), dtype=torch.float32, device=device)

  del k_exp, captured_mass_sum, top100_recall_sum, selected_mask_flat
  torch.cuda.empty_cache()


  # ...




  from pathlib import Path
  import json
  outpath = save_path


  block_cnt = torch.zeros((B, H), dtype=torch.long)
  for b in range(B):
    for h in range(H):
      idx = ragged[b][h]
      if isinstance(idx, torch.Tensor) and idx.numel() > 0:
        block_cnt[b, h] = torch.unique(idx.to("cpu")).numel()
      else:
        block_cnt[b, h] = 0

  block_num_per_head = block_cnt.float().mean(dim=0)


  def mask_to_head_type(block_sparse_mask: torch.Tensor, H: int) -> List[str]:
    m = block_sparse_mask
    if m.dtype != torch.bool:
      m = m.bool()
    if m.dim() == 1:
      qa = m
    elif m.dim() == 2:
      qa = m.any(dim=0)
    else:
      raise ValueError(f"block_sparse_mask shape expected [H] or [B,H], got {tuple(m.shape)}")
    return ["query_aware" if qa[h].item() else "vshead" for h in range(H)]

  head_type_list = mask_to_head_type(block_sparse_mask, H)


  head_captured_mass_list = [float(head_captured_mass[h].item()) for h in range(H)]
  head_top100_recall_list = [float(head_top100_recall[h].item()) for h in range(H)]
  head_type_list_full = head_type_list
  block_num_list = [float(block_num_per_head[h].item()) if B > 1 else int(block_cnt[0, h].item()) for h in range(H)]

  record = {
    "layer": layerid,
    "sample_id": sample_id,
    "num_heads": int(H),
    "head_captured_mass": head_captured_mass_list,
    "head_top100_recall": head_top100_recall_list,
    "top100_recall_query_start": 100,
    "head_type": head_type_list_full,
    "q_len": int(Q),
    "block_num": block_num_list,
  }
  with open(outpath, "a", encoding="utf-8") as f:
    json.dump(record, f, ensure_ascii=False)
    f.write("\n")




  del selected_mask
  torch.cuda.empty_cache()


@triton.jit
def block_wise_decode_attention_kernel(
  q_ptr, # shape: [batch_size, seq_len, num_heads, head_dim]
  k_ptr,
  v_ptr,
  o_ptr,
  block_idx_ptr, # shape: [batch_size, num_heads, num_activated_block]
  BATCH_SIZE,
  NUM_HEADS,
  NUM_KV_HEADS,
  NUM_SHARE_Q_HEADS,
  K_LEN,
  HEAD_DIM: tl.constexpr,
  NUM_BLOCK,
  softmax_scale,
  gqa_interleave: tl.constexpr,
  stride_qb,
  stride_qn,
  stride_qh,
  stride_qd,
  stride_kb,
  stride_kn,
  stride_kh,
  stride_kd,
  stride_vb,
  stride_vn,
  stride_vh,
  stride_vd,
  stride_ob,
  stride_on,
  stride_oh,
  stride_od,
  stride_bb,
  stride_bh,
  stride_bt,
  BLOCK_SIZE_Q: tl.constexpr, # q block size
  BLOCK_SIZE_K: tl.constexpr, # k block size
):
  pid_b = tl.program_id(0)
  pid_h = tl.program_id(1)
  if gqa_interleave:
    pid_kh = pid_h % NUM_KV_HEADS
  else:
    pid_kh = pid_h // NUM_SHARE_Q_HEADS
  block_idx_ptr = block_idx_ptr + pid_b * stride_bb + pid_h * stride_bh
  q_ptrs = tl.make_block_ptr(
    base=q_ptr + pid_b * stride_qb + pid_h * stride_qh,
    shape=(1, HEAD_DIM),
    strides=(stride_qn, stride_qd),
    offsets=(0, 0),
    block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
    order=(1, 0),
  )
  k_ptrs = tl.make_block_ptr(
    base=k_ptr + pid_b * stride_kb + pid_kh * stride_kh,
    shape=(HEAD_DIM, K_LEN),
    strides=(stride_kd, stride_kn),
    offsets=(0, 0),
    block_shape=(HEAD_DIM, BLOCK_SIZE_K),
    order=(0, 1),
  )
  v_ptrs = tl.make_block_ptr(
    base=v_ptr + pid_b * stride_vb + pid_kh * stride_vh,
    shape=(K_LEN, HEAD_DIM),
    strides=(stride_vn, stride_vd),
    offsets=(0, 0),
    block_shape=(BLOCK_SIZE_K, HEAD_DIM),
    order=(1, 0),
  )
  q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
  off_n = tl.arange(0, BLOCK_SIZE_K)
  m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
  lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
  acc_o = tl.full((BLOCK_SIZE_Q, HEAD_DIM), 0, dtype=tl.float32)
  for i in range(0, NUM_BLOCK):
    c = tl.load(block_idx_ptr).to(tl.int32) * BLOCK_SIZE_K
    block_idx_ptr = block_idx_ptr + stride_bt
    k = tl.load(
      tl.advance(k_ptrs, (0, c)), boundary_check=(1,), padding_option="zero"
    )
    qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
    qk += tl.where((off_n < K_LEN - c)[None, :], 0, float("-inf"))
    qk += tl.dot(q, k) * softmax_scale
    m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
    p = tl.math.exp2(qk - m_ij[:, None])
    l_ij = tl.sum(p, axis=1)
    acc_o_scale = tl.math.exp2(m_i - m_ij)
    acc_o = acc_o * acc_o_scale[:, None]
    v = tl.load(
      tl.advance(v_ptrs, (c, 0)), boundary_check=(0,), padding_option="zero"
    )
    p = p.to(v.dtype)
    acc_o += tl.dot(p, v)
    m_i = m_ij
    lse_i = m_ij + tl.math.log2(tl.math.exp2(lse_i - m_ij) + l_ij)
  acc_o = acc_o * tl.math.exp2(m_i - lse_i)[:, None]
  o_ptrs = tl.make_block_ptr(
    base=o_ptr + pid_b * stride_ob + pid_h * stride_oh,
    shape=(1, HEAD_DIM),
    strides=(stride_on, stride_od),
    offsets=(0, 0),
    block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
    order=(1, 0),
  )
  tl.store(o_ptrs, acc_o.to(tl.bfloat16), boundary_check=(0,))


def triton_block_wise_decode_attention(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  block_idx: torch.Tensor,
  block_size: int,
  softmax_scale: Optional[float] = None,
  gqa_interleave: bool = False,
) -> torch.Tensor:
  """Block wise sparse attention (causal attention) implemented by openai triton (ver 3.0.0).

  Args:
    q (torch.Tensor): Query states, shape [batch_size, 1, num_heads, head_dim]
    k (torch.Tensor): Key states, shape [batch_size, seq_len, num_heads, head_dim]
    v (torch.Tensor): Value states, same as key
    block_idx (torch.Tensor): Index of activated blocks, shape [batch_size, num_heads, activated_block_num]
    block_size (int): Block size, only support 16, 32, 64 and 128.
    softmax_scale (Optional[float], optional): Softmax scale. Defaults to 1/math.sqrt(head_dim)
    gqa_interleave (bool): use interleave mode of gqa, default to False.

  Returns:
    torch.Tensor: Attention output, shape [batch_size, 1, num_heads, head_dim]
  """
  batch_size, q_len, num_q_heads, head_dim = q.shape
  assert q_len == 1
  batch_size, k_len, num_kv_heads, head_dim = k.shape
  batch_size, num_q_heads, num_blocks = block_idx.shape
  assert q.dtype == torch.bfloat16
  assert head_dim in {16, 32, 64, 128}, "only support head_dim in {16, 32, 64, 128}"
  assert block_size in {
    16,
    32,
    64,
    128,
  }, "only support block size in {16, 32, 64, 128}"
  assert num_blocks <= triton.cdiv(k_len, block_size)
  assert num_q_heads % num_kv_heads == 0
  num_share_q_heads = num_q_heads // num_kv_heads
  if softmax_scale is None:
    softmax_scale = 1 / math.sqrt(head_dim) * math.log2(math.e)
  else:
    softmax_scale = softmax_scale * math.log2(math.e)
  block_idx = block_idx.sort(-1).values
  o = torch.empty_like(q)
  num_warps = 8
  BLOCK_SIZE_Q = 16
  BLOCK_SIZE_K = block_size
  block_wise_decode_attention_kernel[(batch_size, num_q_heads)](
    q,
    k,
    v,
    o,
    block_idx,
    batch_size,
    num_q_heads,
    num_q_heads,
    num_kv_heads,
    num_share_q_heads,
    k_len,
    head_dim,
    num_blocks,
    softmax_scale,
    gqa_interleave,
    q.stride(0),
    q.stride(1),
    q.stride(2),
    q.stride(3),
    k.stride(0),
    k.stride(1),
    k.stride(2),
    k.stride(3),
    v.stride(0),
    v.stride(1),
    v.stride(2),
    v.stride(3),
    o.stride(0),
    o.stride(1),
    o.stride(2),
    o.stride(3),
    block_idx.stride(0),
    block_idx.stride(1),
    block_idx.stride(2),
    BLOCK_SIZE_Q=BLOCK_SIZE_Q,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
    num_warps=num_warps,
    num_stages=3,
  )
  return o


@triton.jit
def count_kernel(
  x_ptr,
  y_ptr,
  k,
  r,
  stride_xb,
  stride_xh,
  stride_xk,
  stride_yb,
  stride_yh,
  stride_yr,
  BLOCK_SIZE_K: tl.constexpr,
  BLOCK_SIZE_R: tl.constexpr,
):
  pid_b = tl.program_id(0)
  pid_h = tl.program_id(1)
  x_ptr = x_ptr + pid_b * stride_xb + pid_h * stride_xh
  off_k = tl.arange(0, BLOCK_SIZE_K)
  x_ptrs = x_ptr + off_k * stride_xk
  y = tl.zeros((BLOCK_SIZE_R,), dtype=tl.int32)
  for i in range(0, k, BLOCK_SIZE_K):
    x = tl.load(x_ptrs, off_k < k - i, -1)
    x = x // r
    x = tl.where(off_k < k - i, x, -1)
    y += tl.histogram(x, BLOCK_SIZE_R)
    x_ptrs = x_ptrs + BLOCK_SIZE_K * stride_xk
  y = tl.cumsum(y, axis=0)
  y_ptr = y_ptr + pid_b * stride_yb + pid_h * stride_yh + stride_yr
  off_r = tl.arange(0, BLOCK_SIZE_R)
  tl.store(y_ptr + off_r * stride_yr, y, off_r < r)


def triton_column_count_cumsum(x: torch.Tensor, num_columns: int) -> torch.Tensor:
  """count columns of each row for a given index tensor, then do cumsum

  Args:
    x (torch.Tensor): block index in a flatten 2d grid, shape [batch_size, num_heads, activated_block_num]
    num_colums (int): number of columns in the grid

  Returns:
    torch.Tensor: cumsum of columns num in each row, shape [batch_size, num_heads, num_rows + 1 ]
      For example, in a 4x4 block grid, activated blocks have index [0, 5, 8, 9, 13, 14], number of blocks in each row is [1, 1, 2, 2],
      this function will return cumsum tensor [0, 1, 2, 4, 6]
  """
  x = x.to(torch.int32)
  b, h, k = x.shape
  r = num_columns
  block_size_k = min(triton.next_power_of_2(k), 4096)
  block_size_r = triton.next_power_of_2(r + 2)
  y = torch.zeros(b, h, r + 1, device=x.device, dtype=torch.int32)
  count_kernel[(b, h)](
    x,
    y,
    k,
    r,
    x.stride(0),
    x.stride(1),
    x.stride(2),
    y.stride(0),
    y.stride(1),
    y.stride(2),
    block_size_k,
    block_size_r,
  )
  return y


def torch_column_count_cumsum(x: torch.Tensor, num_columns: int) -> torch.Tensor:
  """count columns of each row for a given index tensor, then do cumsum

  Args:
    x (torch.Tensor): block index in a flatten 2d grid, shape [batch_size, num_heads, activated_block_num]
    num_colums (int): number of columns in the grid

  Returns:
    torch.Tensor: cumsum of columns num in each row, shape [batch_size, num_heads, num_rows + 1 ]
      For example, in a 4x4 block grid, activated blocks have index [0, 5, 8, 9, 13, 14], number of blocks in each row is [1, 1, 2, 2],
      this function will return cumsum tensor [0, 1, 2, 4, 6]
  """
  x = x.to(torch.int64)
  batch_size, num_heads, k = x.shape
  y = torch.zeros(
    batch_size, num_heads, num_columns + 1, dtype=torch.int32, device=x.device
  )
  mask = torch.zeros(
    (num_columns + 2) * num_columns, dtype=torch.bool, device=x.device
  )
  for b in range(batch_size):
    for h in range(num_heads):
      mask = mask.view(-1)
      mask.zero_()
      mask.index_fill_(dim=-1, index=x[b, h].view(-1), value=1)
      y[b, h, 1:] = (
        mask.view(num_columns + 2, num_columns)[:-2,].sum(-1).cumsum(-1)
      )
  return y


@triton.jit
def block_wise_prefill_attention_kernel(
  q_ptr, # shape: [batch_size, seq_len, num_heads, head_dim]
  k_ptr,
  v_ptr,
  o_ptr,
  block_idx_ptr, # shape: [batch_size, num_heads, num_all_block]
  idx_bin_ptr, # shape: [batch_size, num_heads, seq_len / block_size + 1]
  BATCH_SIZE,
  NUM_HEADS,
  NUM_KV_HEADS,
  NUM_SHARE_Q_HEADS,
  Q_LEN,
  K_LEN,
  HEAD_DIM,
  NUM_BLOCK,
  grid_offset,
  softmax_scale,
  gqa_interleave: tl.constexpr,
  stride_qb,
  stride_qn,
  stride_qh,
  stride_qd,
  stride_kb,
  stride_kn,
  stride_kh,
  stride_kd,
  stride_vb,
  stride_vn,
  stride_vh,
  stride_vd,
  stride_ob,
  stride_on,
  stride_oh,
  stride_od,
  stride_bb,
  stride_bh,
  stride_bt,
  stride_ib,
  stride_ih,
  stride_it,
  BLOCK_SIZE_Q: tl.constexpr, # q block size
  BLOCK_SIZE_K: tl.constexpr, # k block size
  BLOCK_SIZE_D: tl.constexpr, # d block size
):
  tl.static_assert(BLOCK_SIZE_Q == BLOCK_SIZE_K)
  pid_bh = tl.program_id(0)
  pid_b = pid_bh // NUM_HEADS
  pid_h = pid_bh % NUM_HEADS
  if gqa_interleave:
    pid_kh = pid_h % NUM_KV_HEADS
  else:
    pid_kh = pid_h // NUM_SHARE_Q_HEADS
  pid_q = tl.program_id(1)
  idx_bin_ptr = idx_bin_ptr + pid_b * stride_ib + pid_h * stride_ih
  bin_start = tl.load(idx_bin_ptr + pid_q * stride_it)
  bin_end = tl.load(idx_bin_ptr + (pid_q + 1) * stride_it)
  num_active_block = bin_end - bin_start
  block_idx_ptr = (
    block_idx_ptr + pid_b * stride_bb + pid_h * stride_bh + bin_start * stride_bt
  )
  q_ptrs = tl.make_block_ptr(
    base=q_ptr + pid_b * stride_qb + pid_h * stride_qh,
    shape=(Q_LEN, HEAD_DIM),
    strides=(stride_qn, stride_qd),
    offsets=(pid_q * BLOCK_SIZE_Q - grid_offset, 0),
    block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
    order=(1, 0),
  )
  k_ptrs = tl.make_block_ptr(
    base=k_ptr + pid_b * stride_kb + pid_kh * stride_kh,
    shape=(HEAD_DIM, K_LEN),
    strides=(stride_kd, stride_kn),
    offsets=(0, 0),
    block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
    order=(0, 1),
  )
  v_ptrs = tl.make_block_ptr(
    base=v_ptr + pid_b * stride_vb + pid_kh * stride_vh,
    shape=(K_LEN, HEAD_DIM),
    strides=(stride_vn, stride_vd),
    offsets=(0, 0),
    block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
    order=(1, 0),
  )
  q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
  off_m = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q - grid_offset
  off_n = tl.arange(0, BLOCK_SIZE_K)
  m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
  lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
  acc_o = tl.full((BLOCK_SIZE_Q, BLOCK_SIZE_D), 0, dtype=tl.float32)
  for i in range(0, num_active_block):
    c = tl.load(block_idx_ptr).to(tl.int32) % NUM_BLOCK * BLOCK_SIZE_K - grid_offset
    block_idx_ptr = block_idx_ptr + stride_bt
    k = tl.load(
      tl.advance(k_ptrs, (0, c)), boundary_check=(0, 1), padding_option="zero"
    )
    qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
    qk += tl.where((c + off_n)[None, :] >= 0, 0, float("-inf"))
    qk += tl.where(off_m[:, None] >= (c + off_n)[None, :], 0, float("-inf"))
    qk += tl.dot(q, k) * softmax_scale
    m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
    p = tl.math.exp2(qk - m_ij[:, None])
    l_ij = tl.sum(p, axis=1)
    acc_o_scale = tl.math.exp2(m_i - m_ij)
    acc_o = acc_o * acc_o_scale[:, None]
    v = tl.load(
      tl.advance(v_ptrs, (c, 0)), boundary_check=(0, 1), padding_option="zero"
    )
    p = p.to(v.dtype)
    acc_o += tl.dot(p, v)
    m_i = m_ij
    lse_i = m_ij + tl.math.log2(tl.math.exp2(lse_i - m_ij) + l_ij)
  acc_o = acc_o * tl.math.exp2(m_i - lse_i)[:, None]
  o_ptrs = tl.make_block_ptr(
    base=o_ptr + pid_b * stride_ob + pid_h * stride_oh,
    shape=(Q_LEN, HEAD_DIM),
    strides=(stride_on, stride_od),
    offsets=(pid_q * BLOCK_SIZE_Q - grid_offset, 0),
    block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
    order=(1, 0),
  )
  tl.store(o_ptrs, acc_o.to(tl.bfloat16), boundary_check=(0, 1))


def triton_block_wise_prefill_attention(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  block_idx: Union[torch.Tensor, List[List[torch.Tensor]]],
  block_size: int,
  grid_offset: int = 0,
  softmax_scale: Optional[float] = None,
  gqa_interleave: bool = False,
) -> torch.Tensor:
  """Block wise sparse attention (causal attention) implemented by openai triton (ver 3.0.0).

  Args:
    q (torch.Tensor): Query states, shape [batch_size, seq_lens, num_heads, head_dim]
    k (torch.Tensor): Key states, same as query
    v (torch.Tensor): Value states, same as query
    block_idx (torch.Tensor): Index of activated blocks, shape [batch_size, num_heads, activated_block_num], which is the index of the flattened block grid.
      For example, in a 4x4 block grid, if you want to activate 5 blocks: (0,0), (1,1), (2,0), (3,1), (3,2), the index will be: [0, 5, 8, 13, 14]
    block_size (int): Block size, only support 16, 32, 64 and 128.
    grid_offset (int): Move the grid that divides the block to the lower left corner by grid_offset, default to 0.
    softmax_scale (Optional[float], optional): Softmax scale. Defaults to 1/math.sqrt(head_dim)
    gqa_interleave (bool): use interleave mode of gqa, default to False.

  Returns:
    torch.Tensor: Attention output, shape [batch_size, seq_lens, num_heads, head_dim]
  """
  batch_size, q_len, num_q_heads, head_dim = q.shape
  batch_size, k_len, num_kv_heads, head_dim = k.shape
  assert q.dtype == torch.bfloat16
  assert q_len == k_len
  assert head_dim <= 256, "only support head_dim <= 256"
  if head_dim <= 128:
    assert block_size in {
      32,
      64,
      128,
    }, "only support block size in {32, 64, 128} if head_dim <= 128"
  else:
    assert block_size in {
      32,
      64,
    }, "only support block size in {32, 64} if 128 < head_dim <= 256"
  total_q_blocks = triton.cdiv(grid_offset, block_size) + triton.cdiv(
    q_len - grid_offset, block_size
  )
  total_k_blocks = triton.cdiv(grid_offset, block_size) + triton.cdiv(
    k_len - grid_offset, block_size
  )
  if not isinstance(block_idx, torch.Tensor):
    assert (
      isinstance(block_idx, list)
      and isinstance(block_idx[0], list)
      and isinstance(block_idx[0][0], torch.Tensor)
    )
    assert len(block_idx) == batch_size and len(block_idx[0]) == num_q_heads
    block_idx = [item.view(-1, 1) for sublist in block_idx for item in sublist]
    block_idx = torch.nn.utils.rnn.pad_sequence(
      block_idx,
      batch_first=True,
      padding_value=total_k_blocks * (total_k_blocks + 1),
    )
    block_idx = block_idx.view(batch_size, num_q_heads, -1)
  batch_size, num_q_heads, num_block = block_idx.shape
  assert q_len == k_len
  assert num_block <= total_q_blocks * (total_q_blocks + 1) // 2
  assert num_q_heads % num_kv_heads == 0
  num_share_q_heads = num_q_heads // num_kv_heads
  if softmax_scale is None:
    softmax_scale = 1 / math.sqrt(head_dim) * math.log2(math.e)
  else:
    softmax_scale = softmax_scale * math.log2(math.e)
  block_idx = block_idx.sort(-1).values
  if int(triton.__version__.split(".")[0]) >= 3:
    idx_bins = triton_column_count_cumsum(block_idx, total_k_blocks)
  else:
    warnings.warn(
      "triton version greater than 3.0.0 is required for faster attention"
    )
    idx_bins = torch_column_count_cumsum(block_idx, total_k_blocks)
  o = torch.empty_like(q)
  num_warps, num_stages = get_num_warps_stages(head_dim, block_size, GPU_NAME)
  BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
  block_wise_prefill_attention_kernel[(batch_size * num_q_heads, total_q_blocks)](
    q,
    k,
    v,
    o,
    block_idx,
    idx_bins,
    batch_size,
    num_q_heads,
    num_kv_heads,
    num_share_q_heads,
    q_len,
    k_len,
    head_dim,
    total_q_blocks,
    grid_offset,
    softmax_scale,
    gqa_interleave,
    q.stride(0),
    q.stride(1),
    q.stride(2),
    q.stride(3),
    k.stride(0),
    k.stride(1),
    k.stride(2),
    k.stride(3),
    v.stride(0),
    v.stride(1),
    v.stride(2),
    v.stride(3),
    o.stride(0),
    o.stride(1),
    o.stride(2),
    o.stride(3),
    block_idx.stride(0),
    block_idx.stride(1),
    block_idx.stride(2),
    idx_bins.stride(0),
    idx_bins.stride(1),
    idx_bins.stride(2),
    BLOCK_SIZE_Q=block_size,
    BLOCK_SIZE_K=block_size,
    BLOCK_SIZE_D=BLOCK_SIZE_D,
    num_warps=num_warps,
    num_stages=num_stages,
  )
  return o


def triton_block_wise_attention(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  block_idx: torch.Tensor,
  block_size: int,
  grid_offset: int = 0,
  softmax_scale: Optional[float] = None,
  gqa_interleave: bool = False,
) -> torch.Tensor:
  """Block wise sparse attention (causal attention) implemented by openai triton (ver 3.0.0).

  Args:
    q (torch.Tensor): Query states, shape [batch_size, seq_lens, num_heads, head_dim]
    k (torch.Tensor): Key states, same as query
    v (torch.Tensor): Value states, same as query
    block_idx (torch.Tensor): Index of activated blocks, shape [batch_size, num_heads, activated_block_num], which is the index of the flattened block grid.
      For example, in a 4x4 block grid, if you want to activate 5 blocks: (0,0), (1,1), (2,0), (3,1), (3,2), the index will be: [0, 5, 8, 13, 14]
    block_size (int): Block size, only support 16, 32, 64 and 128.
    grid_offset (int): Move the grid that divides the block to the lower left corner by grid_offset, default to 0.
    softmax_scale (Optional[float], optional): Softmax scale. Defaults to 1/math.sqrt(head_dim)
    gqa_interleave (bool): use interleave mode of gqa, default to False.

  Returns:
    torch.Tensor: Attention output, shape [batch_size, seq_lens, num_heads, head_dim]
  """
  if q.shape[1] > 1:
    return triton_block_wise_prefill_attention(
      q,
      k,
      v,
      block_idx,
      block_size,
      grid_offset,
      softmax_scale,
      gqa_interleave,
    )
  else:
    return triton_block_wise_decode_attention(
      q, k, v, block_idx, block_size, softmax_scale, gqa_interleave
    )


@triton.jit
def bnhd_pool_kernel(
  x_ptr,
  y_ptr,
  pool_type: tl.constexpr,
  batch_size,
  seq_len,
  num_heads,
  head_dim: tl.constexpr,
  stride_xb,
  stride_xn,
  stride_xh,
  stride_xd,
  stride_yb,
  stride_yn,
  stride_yh,
  stride_yd,
  BLOCK_SIZE_N: tl.constexpr,
  BLOCK_SIZE_H: tl.constexpr, # {16, 32, 64, 128, 256, 512}
  BLOCK_SIZE_D: tl.constexpr, # {16, 32, 64, 128, 256, 512}
):
  pid_b = tl.program_id(0)
  pid_n = tl.program_id(1)
  pid_h = tl.program_id(2)

  x_ptr = (
    x_ptr
    + pid_b * stride_xb
    + pid_n * BLOCK_SIZE_N * stride_xn
    + pid_h * BLOCK_SIZE_H * stride_xh
  )

  off_n = tl.arange(0, BLOCK_SIZE_N)
  off_h = tl.arange(0, BLOCK_SIZE_H)
  off_d = tl.arange(0, BLOCK_SIZE_D)

  cur_block_size_n = min(seq_len - pid_n * BLOCK_SIZE_N, BLOCK_SIZE_N)

  x_mask = (
    (off_n < seq_len - pid_n * BLOCK_SIZE_N)[:, None, None]
    & (off_h < num_heads - pid_h * BLOCK_SIZE_H)[None, :, None]
    & (off_d < head_dim)[None, None, :]
  )
  x = tl.load(
    x_ptr
    + off_n[:, None, None] * stride_xn
    + off_h[None, :, None] * stride_xh
    + off_d[None, None, :] * stride_xd,
    mask=x_mask,
    other=0,
  )
  if pool_type == 0:
    y = tl.sum(x, axis=0) / cur_block_size_n
  elif pool_type == 1:
    y = tl.max(x, axis=0)
  elif pool_type == 2:
    y = tl.min(x, axis=0)
  elif pool_type == 3:
    y = tl.max(tl.abs(x), axis=0)
  elif pool_type == 4:
    y = tl.sum(x, axis=0)
  else:
    y = tl.sum(x, axis=0) / cur_block_size_n
  y_ptr = (
    y_ptr + pid_b * stride_yb + pid_n * stride_yn + pid_h * BLOCK_SIZE_H * stride_yh
  )
  y_mask = (off_h < num_heads - pid_h * BLOCK_SIZE_H)[:, None] & (off_d < head_dim)[
    None, :
  ]
  tl.store(
    y_ptr + off_h[:, None] * stride_yh + off_d[None, :] * stride_yd, y, mask=y_mask
  )


def triton_bnhd_pool(x: torch.Tensor, kernel_size: int, pool_type: str = "avg"):
  b, n, h, d = x.shape
  assert d in {16, 32, 64, 128}
  assert kernel_size in {16, 32, 64, 128, 256, 512}
  m = triton.cdiv(n, kernel_size)
  y = torch.zeros(b, m, h, d, device=x.device, dtype=x.dtype)

  if pool_type == "last":
    if n % kernel_size == 0:
      return x[:, kernel_size - 1 :: kernel_size, ...]
    else:
      return torch.cat(
        (x[:, kernel_size - 1 :: kernel_size, ...], x[:, -1:, ...]), dim=1
      )

  block_size_h = triton.next_power_of_2(h)
  while kernel_size * block_size_h * d > 128 * 128 * 128:
    block_size_h = block_size_h // 2

  block_size_d = triton.next_power_of_2(d)
  pool_str_to_type = {"avg": 0, "max": 1, "min": 2, "maxabs": 3, "sum": 4}
  pool_type = pool_str_to_type[pool_type]

  grid = lambda META: (
    b,
    triton.cdiv(n, META["BLOCK_SIZE_N"]),
    triton.cdiv(h, META["BLOCK_SIZE_H"]),
  )
  bnhd_pool_kernel[grid](
    x,
    y,
    pool_type,
    b,
    n,
    h,
    d,
    x.stride(0),
    x.stride(1),
    x.stride(2),
    x.stride(3),
    y.stride(0),
    y.stride(1),
    y.stride(2),
    y.stride(3),
    BLOCK_SIZE_N=kernel_size,
    BLOCK_SIZE_H=block_size_h,
    BLOCK_SIZE_D=block_size_d,
  )
  return y


@triton.jit
def bhn_sumpool_kernel(
  x_ptr,
  y_ptr,
  batch_size,
  num_heads,
  seq_len,
  stride_xb,
  stride_xh,
  stride_xn,
  stride_yb,
  stride_yh,
  stride_yn,
  BLOCK_SIZE_N: tl.constexpr,
  BLOCK_SIZE_H: tl.constexpr, # {16, 32, 64, 128, 256, 512}
):
  pid_b = tl.program_id(0)
  pid_h = tl.program_id(1)
  pid_n = tl.program_id(2)
  x_ptr = (
    x_ptr
    + pid_b * stride_xb
    + pid_h * BLOCK_SIZE_H * stride_xh
    + pid_n * BLOCK_SIZE_N * stride_xn
  )
  off_h = tl.arange(0, BLOCK_SIZE_H)
  off_n = tl.arange(0, BLOCK_SIZE_N)
  x_mask = (off_n < seq_len - pid_n * BLOCK_SIZE_N)[None, :] & (
    off_h < num_heads - pid_h * BLOCK_SIZE_H
  )[:, None]
  x = tl.load(
    x_ptr + off_h[:, None] * stride_xh + off_n[None, :] * stride_xn,
    mask=x_mask,
    other=0,
  )
  y = tl.sum(x, axis=1)
  y_ptr = (
    y_ptr + pid_b * stride_yb + pid_h * BLOCK_SIZE_H * stride_yh + pid_n * stride_yn
  )
  y_mask = off_h < num_heads - pid_h * BLOCK_SIZE_H
  tl.store(y_ptr + off_h * stride_yh, y, mask=y_mask)


def triton_bhn_sumpool(x: torch.Tensor, kernel_size: int):
  b, h, n = x.shape
  assert kernel_size in {16, 32, 64, 128, 256, 512}
  m = triton.cdiv(n, kernel_size)
  y = torch.empty(b, h, m, device=x.device, dtype=x.dtype)
  block_size_h = triton.next_power_of_2(h)
  grid = lambda META: (
    b,
    triton.cdiv(h, META["BLOCK_SIZE_H"]),
    triton.cdiv(n, META["BLOCK_SIZE_N"]),
  )
  bhn_sumpool_kernel[grid](
    x,
    y,
    b,
    h,
    n,
    x.stride(0),
    x.stride(1),
    x.stride(2),
    y.stride(0),
    y.stride(1),
    y.stride(2),
    BLOCK_SIZE_N=kernel_size,
    BLOCK_SIZE_H=block_size_h,
  )
  return y


def torch_bhn_sumpool(x: torch.Tensor, kernel_size: int):
  b, h, n = x.shape
  x = torch.nn.functional.pad(
    x,
    (
      0,
      math.ceil(n / kernel_size) * kernel_size - n,
    ),
    value=0,
  )
  x = x.view(b, h, -1, kernel_size).sum(-1)
  return x


def score_cover_topk(x: torch.Tensor, score: float):
  cumsum_x = torch.cumsum(torch.sort(x, dim=-1, descending=True).values, dim=-1)
  topk = torch.sum(cumsum_x <= score, dim=-1) + 1
  return topk


def score_cover_idx(x: torch.Tensor, score: float, padding_value=0):
  x, idx = torch.sort(x, dim=-1, descending=True)
  cumsum_x = torch.cumsum(x, dim=-1)
  idx[cumsum_x > score] = padding_value
  return idx


def sum_all_diagonal_matrix(mat: torch.tensor):
  b, h, n, m = mat.shape
  mat_padded = torch.nn.functional.pad(mat, (n - 1, 0), value=0)
  mat_strided = mat_padded.as_strided(
    (b, h, m, n), (h * n * (n + m - 1), n * (n + m - 1), 1, n + m)
  )
  sum_diags = torch.sum(mat_strided, -1)
  return sum_diags


def transform_veritcal_slash_idx(v_idx, s_idx, num_blocks):
  batch_size, num_heads, _ = v_idx.shape


  range_blocks = torch.arange(num_blocks, device=s_idx.device)[None, None, :, None]





  #
  v_idx = (
    torch.arange(0, num_blocks, device=v_idx.device)[None, None, :, None]
    * num_blocks
    + v_idx[:, :, None, :]
  ).view(batch_size, num_heads, -1)
  v_idx[v_idx // num_blocks < v_idx % num_blocks] = 0
  s_idx = (
    range_blocks * num_blocks + range_blocks + s_idx[:, :, None, :] * num_blocks
  ).view(batch_size, num_heads, -1)
  s_idx[s_idx >= num_blocks * num_blocks] = 0
  vs_idx = torch.cat((s_idx, v_idx), dim=-1)
  block_idx = [
    [torch.unique(vs_idx[b, h]) for h in range(num_heads)]
    for b in range(batch_size)
  ]
  return block_idx


causal_mask = None


def get_block_vertical_slash_from_qk(
  qk: torch.Tensor,
  block_size: int,
):
  batch_size, num_heads, last_q_len, seq_len = qk.shape
  slash = sum_all_diagonal_matrix(qk)
  slash = torch_bhn_sumpool(slash, block_size)
  slash = slash / last_q_len
  vertical = qk.sum(-2)
  vertical = torch_bhn_sumpool(vertical, block_size)
  vertical = vertical / last_q_len
  return vertical, slash


def square_root_js_divergence(p: torch.Tensor, q: torch.Tensor):
  m = (p + q) / 2
  return torch.sqrt(
    0.5 * (p * torch.log(p / m)).sum(-1) + 0.5 * (q * torch.log(q / m)).sum(-1)
  )


def get_active_blocks(
  q,
  k,
  v,
  block_size,
  gamma,
  min_budget,
  max_budget,
  tau=0,
  gqa_interleave=False,
):

  batch_size, seq_len, num_heads, head_dim = q.shape
  num_share_q_heads = num_heads // k.shape[2]
  num_blocks = math.ceil(seq_len / block_size)
  max_budget = min(max_budget, num_blocks)

  last_q = q[:, -block_size:, :, :] / math.sqrt(head_dim)



  if not gqa_interleave:




    qk = torch.einsum(
      "bihgd, bjhgd -> bhgij",  # [b, block_size, num_kv_head, num_share_q_heads, head_dim]
      last_q.view(
        last_q.shape[0], last_q.shape[1], -1, num_share_q_heads, head_dim
      ),
      k.view(k.shape[0], k.shape[1], -1, 1, head_dim),
    )
  else:
    qk = torch.einsum(
      "bihgd, bjhgd -> bhgij",
      last_q.view(
        last_q.shape[0], last_q.shape[1], num_share_q_heads, -1, head_dim
      ),
      k.view(k.shape[0], k.shape[1], 1, -1, head_dim),
    )
  global causal_mask


  if causal_mask is None:
    causal_mask = torch.arange(0, block_size, device=last_q.device)
    causal_mask = causal_mask[:, None] >= causal_mask[None, :]
    causal_mask = causal_mask[None, None, None, ...]
  qk[..., -block_size:].masked_fill_(
    ~causal_mask[..., :block_size, :block_size], float("-inf")
  )

  qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)  # L: blocksize T: seqlen
  qk = rearrange(qk, "b h g i j -> b (h g) i j")






  slash = sum_all_diagonal_matrix(qk) / qk.shape[-2]


  vertical = qk.mean(-2)





  num_vertical_blocks = score_cover_topk(vertical, gamma) // 128 + 1
  num_slash_blocks = score_cover_topk(slash, gamma) // 128 + 1


  num_vertical_blocks[num_vertical_blocks < min_budget] = min_budget
  num_vertical_blocks[num_vertical_blocks > max_budget] = max_budget
  num_slash_blocks[num_slash_blocks < min_budget] = min_budget
  num_slash_blocks[num_slash_blocks > max_budget] = max_budget
  







  vertical = torch_bhn_sumpool(vertical, block_size)
  slash = torch_bhn_sumpool(slash, block_size)

  with _breakdown_measure("pattern"):
    if not gqa_interleave:






      avg_k = triton_bnhd_pool(k, block_size).repeat_interleave(num_share_q_heads, 2)
    else:
      avg_k = triton_bnhd_pool(k, block_size).repeat(1, 1, num_share_q_heads, 1)
    



    avg_qk = torch.einsum(   #lastq [B, block_size, Hq, D]
      "bihd, bjhd -> bhij", last_q.mean(1, keepdim=True), avg_k
    ).squeeze(2)

    avg_qk = torch.softmax(avg_qk, dim=-1, dtype=torch.float32)





    kl_div = square_root_js_divergence(avg_qk, vertical)
    block_sparse_mask = kl_div < tau
    
    return_mask=block_sparse_mask
    num_vertical_blocks[block_sparse_mask] = min_budget
    num_slash_blocks[block_sparse_mask] = min_budget

  vertical[..., :1] = torch.inf

  slash[..., -1:] = torch.inf






  num_slash_blocks = num_slash_blocks.view(batch_size * num_heads)
  slash = slash.view(batch_size * num_heads, -1)





  slash_topk = (num_blocks - 1) - slash.topk(
    min(num_slash_blocks.max().item(), num_blocks), -1
  ).indices


  slash_topk[
    torch.arange(slash_topk.shape[-1], device=num_slash_blocks.device)[None, :]
    >= num_slash_blocks[:, None]
  ] = 0

  slash_topk = slash_topk.view(batch_size, num_heads, -1)

  num_vertical_blocks = num_vertical_blocks.view(batch_size * num_heads)
  vertical = vertical.view(batch_size * num_heads, -1)
  vertical_topk = vertical.topk(
    min(num_vertical_blocks.max().item(), num_blocks), -1
  ).indices
  vertical_topk[
    torch.arange(vertical_topk.shape[-1], device=num_vertical_blocks.device)[
      None, :
    ]
    >= num_vertical_blocks[:, None]
  ] = 0
  vertical_topk = vertical_topk.view(batch_size, num_heads, -1)






  block_idx = transform_veritcal_slash_idx(vertical_topk, slash_topk, num_blocks)


  block_causal_mask = None

  for b, h in block_sparse_mask.nonzero():
    if block_causal_mask is None:
      block_causal_mask = torch.tril(
        torch.ones(num_blocks, num_blocks, device=q.device, dtype=torch.bool)
      )

    pad_q = math.ceil(seq_len / block_size) * block_size - seq_len





    avg_q = (
      torch.nn.functional.pad(q[b, :, h, :], (0, 0, 0, pad_q), value=0)
      .view(num_blocks, block_size, head_dim)
      .mean(1)
    )

    avg_q[-1, :] = avg_q[-1, :] * block_size / (block_size - pad_q)



    attn = torch.einsum(
      "id, jd -> ij", avg_q / math.sqrt(head_dim), avg_k[b, :, h, :]
    ).masked_fill_(~block_causal_mask, float("-inf"))



    attn = torch.softmax(attn, dim=-1, dtype=torch.float32).view(-1)


    block_topk = score_cover_idx(attn, gamma * num_blocks)

    block_idx[b][h] = torch.unique(torch.cat((block_idx[b][h], block_topk), dim=-1))
  return block_idx,return_mask



def record_blocks(task,model_name,seq_len,num_blocks):
  avg=sum(blocks_num)/len(blocks_num)
  

  blocks = num_blocks * (num_blocks + 1) // 2
  from pathlib import Path
  import json

  outdir = Path("efficiency/blocks")
  outdir.mkdir(parents=True, exist_ok=True)

  outpath_pre_head = outdir / f"{model_name}_{task}_block_num.jsonl"
  outpath_pre_head.parent.mkdir(parents=True, exist_ok=True)


  with open(outpath_pre_head, "a", encoding="utf-8") as f:

    record = {
      "avg_blocks_nums": float(avg),
      "blocks": int(blocks),
      "seq_len": int(seq_len),
    }
    json.dump(record, f, ensure_ascii=False)
    f.write("\n")



from transformers.utils import is_flash_attn_2_available

if is_flash_attn_2_available():
  from flash_attn import flash_attn_func
else:
  flash_attn_func = triton_flash_attention


@torch.no_grad()
def flex_prefill_attention(
  model_name:str,
  layerid:int,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  gamma: float,
  tau: float = 0,
  task:str=" ",
  min_budget: int = None,
  max_budget: int = None,
  gqa_interleave: bool = False,
  softmax_scale: Optional[float] = None,
  block_size: int = 128,
  return_computational_ratio: bool = False,
  type:str="",
  save_path:str="",
  sample_id: str = "",
) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
  """Flex Prefill sparse attention function. If query length is 1, will use flash decoding attention.

  Args:
    q (torch.Tensor): query tensor, shape [batch_size, q_len, num_q_heads, head_dim]
    k (torch.Tensor): key tensor, shape [batch_size, kv_len, num_kv_heads, head_dim]
    v (torch.Tensor): value tensor, shape [batch_size, kv_len, num_kv_heads, head_dim]
    gamma (float): attention coverage ratio, (0, 1).
    tau (float, optional): query aware head threshold, [0, 1]. Defaults to 0.
    min_budget (int, optional): minimum number of calculated tokens. Defaults to None.
    max_budget (int, optional): maximum number of calculated tokens. Defaults to None.
    gqa_interleave (bool, optional): GQA pattern. Defaults to False.
    softmax_scale (Optional[float], optional): softmax scale. Defaults to None, which means sqrt(head_dim).
    block_size (int, optional): block size. Defaults to 128.
    return_computational_ratio (bool, optional): whether to return computation ratio. Defaults to False.

  Returns:
    Union[torch.Tensor, Tuple[torch.Tensor, float]]: if return_computational_ratio is True, return attention output, else return attention output and computation ratio.
  """
  batch_size, q_len, num_q_heads, head_dim = q.shape
  batch_size, k_len, num_kv_heads, head_dim = k.shape
  assert batch_size == 1, "only support batch size 1 for now"
  if q_len == 1:
    if gqa_interleave:
      attn_out = triton_flash_attention(
        q, k, v, softmax_scale=softmax_scale, gqa_interleave=True
      )
    else:
      attn_out = flash_attn_func(q, k, v, softmax_scale=softmax_scale)
    if return_computational_ratio:
      return attn_out, 1
    else:
      return attn_out
  assert q.shape[1] == k.shape[1]
  assert head_dim in {16, 32, 64, 128}
  assert block_size in {16, 32, 64, 128}
  num_blocks = math.ceil(q_len / block_size)
  min_budget = 1 if min_budget is None else min_budget
  max_budget = 2147483647 if max_budget is None else max_budget



  if q_len <= max(2 * block_size, math.ceil(min_budget / block_size) * block_size):
    if gqa_interleave:
      attn_out = triton_flash_attention(
        q, k, v, softmax_scale=softmax_scale, causal=True, gqa_interleave=True
      )
    else:
      attn_out = flash_attn_func(
        q, k, v, softmax_scale=softmax_scale, causal=True
      )
    if return_computational_ratio:
      return attn_out, 1
    else:
      return attn_out







  block_idx, block_sparse_mask= get_active_blocks(
    q,
    k,
    v,
    block_size,
    gamma,
    math.ceil(min_budget / block_size),
    math.ceil(max_budget / block_size),
    tau,
    gqa_interleave,
  )

  global blocks_num


  mask = block_sparse_mask.to(torch.bool)     # (B,H,q_blk,k_blk)
  selected_per_qblk = mask.sum(dim=-1)

  total = sum(t.numel() for group in block_idx for t in group)

  if model_name=="Llama":
    total = total / 32
  elif model_name=="Qwen":
    total=total / 28
  
  if task!=" ":
    blocks_num.append(total)
  
  if len(blocks_num)==28 and model_name=="Qwen" and task!=" ":
    record_blocks(task,model_name,q_len,num_blocks)
    blocks_num=[]
  elif len(blocks_num)==32 and model_name=="Llama" and task!=" ":
    record_blocks(task,model_name,q_len,num_blocks)
    blocks_num=[]





  


  if type=="recall":
    calculate_flexprefill_layer_captured_mass(save_path, model_name, layerid, q, k, block_size, block_idx, block_sparse_mask, min_budget, gamma, tau, sample_id=sample_id, gqa_interleave=gqa_interleave)

  


  if return_computational_ratio:

    #
    activated_block_num = sum(
      [
        block_idx[b][h].shape[-1]
        for b in range(len(block_idx))
        for h in range(len(block_idx[0]))
      ]
    )

    total_block_num = num_blocks * num_blocks * len(block_idx) * len(block_idx[0])

    computational_ratio = activated_block_num / total_block_num

  

  attn_out = triton_block_wise_attention(
    q,
    k,
    v,
    block_idx,
    block_size,
    softmax_scale=softmax_scale,
    gqa_interleave=gqa_interleave,
  )
  if return_computational_ratio:
    return attn_out, computational_ratio
  else:
    return attn_out


if __name__ == "__main__":
  torch.manual_seed(0)
  B, N, H, D = 1, 64000, 32, 128
  gamma = 0.9
  tau = 0.1

  q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16) / 0.5
  k = torch.randn(B, N, H // 4, D, device="cuda", dtype=torch.bfloat16) / 0.5
  v = torch.randn(B, N, H // 4, D, device="cuda", dtype=torch.bfloat16)

  flex_prefill_output, computational_ratio = flex_prefill_attention(
    q,
    k,
    v,
    gamma,
    tau,
    min_budget=1024,
    max_budget=None,
    gqa_interleave=False,
    block_size=128,
    return_computational_ratio=True,
  )
  print("attention output norm:", flex_prefill_output.norm())
  print(f"computational ratio: {computational_ratio*100:.2f}%")
