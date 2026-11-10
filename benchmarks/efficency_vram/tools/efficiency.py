"""Generation and token-step timing, independent of models and prediction code."""

from contextlib import contextmanager
import time

import torch


class EfficiencyMeter:
    def __init__(self, devices, batch_size, output_length):
        self.devices = devices
        self.batch_size = batch_size
        self.output_length = output_length
        self.step_latencies_s = []

    def synchronize(self):
        for device in self.devices:
            torch.cuda.synchronize(device)

    def __enter__(self):
        self.synchronize()
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.synchronize()
        self.elapsed = time.perf_counter() - self.started

    @contextmanager
    def step(self):
        """Wrap one forward and argmax; cache/mask updates remain outside."""
        self.synchronize()
        started = time.perf_counter()
        yield
        self.synchronize()
        self.step_latencies_s.append(time.perf_counter() - started)

    def metrics(self):
        steps = self.step_latencies_s
        decode_s = sum(steps[1:]) if steps else None
        decode_steps = len(steps) - 1 if steps else None
        return {
            "generation_latency_s": self.elapsed,
            "ttft_s": steps[0] if steps else None,
            "ttft_ms": steps[0] * 1000 if steps else None,
            "tpot_ms": decode_s * 1000 / decode_steps if decode_steps else None,
            "decode_latency_s": decode_s,
            "decode_steps": decode_steps,
            "decode_throughput_tokens_per_s": self.batch_size * decode_steps / decode_s if decode_steps else None,
            "step_latencies_s": steps or None,
            "step_timing_scope": "forward_and_argmax" if steps else None,
            "step_timing_unavailable_reason": None if steps else "Native generate API has no uniform step observer",
        }
