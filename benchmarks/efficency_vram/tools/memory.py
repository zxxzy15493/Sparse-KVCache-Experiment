"""PyTorch allocator memory monitoring for an explicit set of CUDA devices."""

import torch


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
