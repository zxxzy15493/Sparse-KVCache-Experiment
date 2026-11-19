"""时间与显存测量；计算和同步边界保持原实现。"""

import time
from contextlib import contextmanager

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


class MemoryMonitor:
    def __init__(self, devices):
        self.devices = devices
        self.decode_baseline = None
        self.decode_peak = None
        self.prefill_peak = None

    def snapshot(self, peak=False):
        allocated = torch.cuda.max_memory_allocated if peak else torch.cuda.memory_allocated
        reserved = torch.cuda.max_memory_reserved if peak else torch.cuda.memory_reserved
        return {str(device): {"allocated_mib": allocated(device) / 1024**2,
                              "reserved_mib": reserved(device) / 1024**2}
                for device in self.devices}

    def synchronize(self):
        for device in self.devices:
            torch.cuda.synchronize(device)

    def begin_decode_phase(self):
        """Restart peak counters after prefill, immediately before decode."""
        self.synchronize()
        self.prefill_peak = self.snapshot(peak=True)
        self.decode_baseline = self.snapshot()
        for device in self.devices:
            torch.cuda.reset_peak_memory_stats(device)

    def __enter__(self):
        self.synchronize()
        self.baseline = self.snapshot()
        for device in self.devices:
            torch.cuda.reset_peak_memory_stats(device)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.synchronize()
        measured_peak = self.snapshot(peak=True)
        if self.decode_baseline is not None:
            self.decode_peak = measured_peak
            self.peak = {
                device: {
                    field: max(self.prefill_peak[device][field], values[field])
                    for field in ("allocated_mib", "reserved_mib")
                }
                for device, values in measured_peak.items()
            }
        else:
            self.peak = measured_peak

    def metrics(self):
        by_device = {
            device: {"baseline_allocated_mib": self.baseline[device]["allocated_mib"],
                     "baseline_reserved_mib": self.baseline[device]["reserved_mib"],
                     "peak_allocated_mib": values["allocated_mib"],
                     "peak_reserved_mib": values["reserved_mib"],
                     "incremental_peak_allocated_mib": values["allocated_mib"] - self.baseline[device]["allocated_mib"]}
            for device, values in self.peak.items()
        }
        return {
            "memory_by_device": by_device,
            "peak_allocated_mib": max(values["peak_allocated_mib"] for values in by_device.values()),
            "peak_reserved_mib": max(values["peak_reserved_mib"] for values in by_device.values()),
            "sum_device_peak_allocated_mib": sum(values["peak_allocated_mib"] for values in by_device.values()),
            "sum_device_peak_reserved_mib": sum(values["peak_reserved_mib"] for values in by_device.values()),
            "decode_memory_by_device": self._decode_memory_by_device(),
            "decode_peak_allocated_mib": self._decode_peak("allocated_mib"),
            "decode_peak_reserved_mib": self._decode_peak("reserved_mib"),
        }

    def _decode_memory_by_device(self):
        if self.decode_peak is None or self.decode_baseline is None:
            return None

        return {
            device: {
                "baseline_allocated_mib": self.decode_baseline[device]["allocated_mib"],
                "baseline_reserved_mib": self.decode_baseline[device]["reserved_mib"],
                "peak_allocated_mib": peak["allocated_mib"],
                "peak_reserved_mib": peak["reserved_mib"],
                "incremental_peak_allocated_mib": (
                    peak["allocated_mib"] - self.decode_baseline[device]["allocated_mib"]
                ),
            }
            for device, peak in self.decode_peak.items()
        }

    def _decode_peak(self, field):
        if self.decode_peak is None:
            return None

        return max(values[field] for values in self.decode_peak.values())


