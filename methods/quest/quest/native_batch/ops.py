"""Thin wrappers around the original-style batched CUDA operator extension."""

from functools import lru_cache
import importlib.util

import torch


@lru_cache(maxsize=1)
def load_kernels():
    from .build import build
    path = build()
    spec = importlib.util.spec_from_file_location('_kernels_native_batch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rms_norm(x, weight, epsilon):
    x = x.contiguous()
    output = torch.empty_like(x)
    load_kernels().rms_norm_forward(x, weight, output, epsilon)
    return output
