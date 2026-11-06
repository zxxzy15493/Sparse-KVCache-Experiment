"""CUDA regressions for kernel launches after Accelerate moves between GPUs."""

import pytest
import torch

from xattn.src.Xattention import xattn_estimate
from xattn.src.kernels import (
  flat_group_gemm,
  flat_group_gemm_fuse_reshape,
  softmax_fuse_block_sum,
)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs")
@pytest.mark.parametrize("kernel", ["gemm", "fused_gemm", "causal_softmax", "softmax"])
def test_kernel_launch_uses_input_device_and_stream(kernel):
  # Peer access makes a launch on the wrong GPU legal, but does not order it
  # after PyTorch operations on the tensor's own stream.
  with torch.cuda.device(0):
    peer = torch.ones(1, device="cuda:0").to("cuda:1").to("cuda:0")
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    stream = torch.cuda.Stream(device=1)
    q = torch.zeros(1, 2, 1024, 64, device="cuda:1", dtype=torch.bfloat16)
    k = torch.ones_like(q)
    logits = torch.zeros(1, 2, 64, 64, device="cuda:1", dtype=torch.bfloat16)
    torch.cuda.synchronize(1)

    def launch():
      if kernel == "gemm":
        return flat_group_gemm(q[:, :, :64], k[:, :, :64], 0, 64)
      if kernel == "fused_gemm":
        return flat_group_gemm_fuse_reshape(q, k, 16, 0, 64)
      return softmax_fuse_block_sum(
        logits, 8, 8, 0, 64, 64, 1.4426950408889634,
        is_causal=kernel == "causal_softmax",
      )

    # Compile before delaying the producer, so JIT latency cannot hide the race.
    with torch.cuda.device(1):
      launch()
    torch.cuda.synchronize(1)
    with torch.cuda.stream(stream):
      torch.cuda._sleep(100_000_000)
      q.fill_(1)
      logits[..., 0] = 4
      # Simulate a model layer on GPU 1 while the current device is still 0.
      with torch.cuda.device(0):
        actual = launch()
        assert torch.cuda.current_device() == 0
    torch.cuda.synchronize(0)
    stream.synchronize()

    if kernel in {"gemm", "fused_gemm"}:
      expected = torch.full_like(actual, 64 if kernel == "gemm" else 64 * 16)
    else:
      scores = logits.float()
      if kernel == "causal_softmax":
        scores = scores.masked_fill(torch.ones(64, 64, device=logits.device, dtype=torch.bool).triu(1), -float("inf"))
      expected = scores.softmax(-1).reshape(1, 2, 8, 8, 8, 8).sum(-1).sum(-2).to(actual.dtype)
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs")
@pytest.mark.parametrize("length", [1031, 131072])
def test_estimator_matches_when_current_device_differs(length):
  torch.manual_seed(17)
  with torch.cuda.device(0):
    query = torch.randn(1, 4, length, 128, device="cuda:0", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    other_query, other_key = query.to("cuda:1"), key.to("cuda:1")
    for repeat in range(3):
      expected_sum, expected_mask = xattn_estimate(query, key, 128, 16, threshold=1)
      actual_sum, actual_mask = xattn_estimate(other_query, other_key, 128, 16, threshold=1)
      assert torch.cuda.current_device() == 0
      assert torch.isfinite(actual_sum).all()
      torch.testing.assert_close(actual_sum.cpu(), expected_sum.cpu())
      torch.testing.assert_close(actual_mask.cpu(), expected_mask.cpu())
