"""Decode-only CUDA memory measurements for the dedicated VRAM experiment.

AdaKV (HeadKV) has no separate ``decode_forward`` entry point: the patched
language-model module (``model.model`` for LLaMA/Qwen, ``model.transformer.encoder``
for ChatGLM) is called once for the prefill and then once per decoding step. This
monitor therefore wraps that module's ``forward`` and detects decoding steps from
the module's own ``_timing['decode_steps']`` counter when it is available
(LLaMA), falling back to the call order otherwise (Qwen/ChatGLM).
"""


class DecodeMemoryMonitor:
    """Observe the language-model forward without changing the model adapter.

    Average allocated memory is sampled after each synchronized decode forward.
    Decode peaks use allocator counters reset at the first decode call. Prefill
    peaks are saved so resetting those counters does not lose batch-wide peaks.
    """

    def __init__(self, module):
        self.module = module
        self.num_steps = 0
        self._started = False
        self._call_index = 0

    def __enter__(self):
        import torch

        self._cuda = torch.cuda
        self._devices = list(range(self._cuda.device_count()))
        self._original = self.module.forward
        self._had_override = "forward" in vars(self.module)
        self._allocated_sum = {f"cuda:{device}": 0.0 for device in self._devices}
        self._prefill_alloc = {}
        self._prefill_reserved = {}
        self._decode_peak = {}
        self.module.forward = self._forward
        return self

    def _synchronize(self):
        for device in self._devices:
            self._cuda.synchronize(device)

    @staticmethod
    def _decode_steps(module):
        timing = getattr(module, "_timing", None)
        if timing is None:
            return None
        return timing.get("decode_steps")

    def _forward(self, *args, **kwargs):
        steps_before = self._decode_steps(self.module)
        result = self._original(*args, **kwargs)
        steps_after = self._decode_steps(self.module)
        if steps_before is not None and steps_after is not None:
            is_decode = steps_after > steps_before
        else:
            # Qwen/ChatGLM do not populate ``_timing``: the first call of every
            # generation is the prefill, the rest are decoding steps.
            is_decode = self._call_index > 0
        self._call_index += 1
        if not is_decode:
            return result

        if not self._started:
            self._synchronize()
            for device in self._devices:
                key = f"cuda:{device}"
                self._prefill_alloc[key] = self._cuda.max_memory_allocated(device) / 1024**2
                self._prefill_reserved[key] = self._cuda.max_memory_reserved(device) / 1024**2
                self._cuda.reset_peak_memory_stats(device)
            self._started = True
        self._synchronize()
        for device in self._devices:
            self._allocated_sum[f"cuda:{device}"] += self._cuda.memory_allocated(device) / 1024**2
        self.num_steps += 1
        return result

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc is None and self.num_steps:
                self._synchronize()
                self._decode_peak = {
                    f"cuda:{device}": self._cuda.max_memory_allocated(device) / 1024**2
                    for device in self._devices
                }
        finally:
            if self._had_override:
                self.module.forward = self._original
            else:
                del self.module.forward
        return False

    def metrics(self):
        averages = {device: total / self.num_steps if self.num_steps else None
                    for device, total in self._allocated_sum.items()}
        peaks = {device: self._decode_peak.get(device) for device in self._allocated_sum}
        return {
            "decode_num_steps": self.num_steps,
            "decode_avg_alloc": max(averages.values(), default=0.0) if self.num_steps else None,
            "decode_peak_alloc": max(peaks.values(), default=0.0) if self.num_steps else None,
            "decode_avg_alloc_by_device": averages,
            "decode_peak_alloc_by_device": peaks,
        }

    def merge_batch_peaks(self, metrics):
        """Keep the original whole-batch allocated and reserved peak metrics."""
        result = dict(metrics)
        for field, prefill in (("peak_alloc", self._prefill_alloc),
                               ("peak_reserved", self._prefill_reserved)):
            per_device = dict(result[f"{field}_by_device"])
            for device, peak in prefill.items():
                per_device[device] = max(per_device.get(device, 0.0), peak)
            result[f"{field}_by_device"] = per_device
            result[field] = max(per_device.values(), default=0.0)
        return result
