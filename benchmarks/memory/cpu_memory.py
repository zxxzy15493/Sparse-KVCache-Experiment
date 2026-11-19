"""采样 Linux 推理进程的 CPU 常驻内存。"""

from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread


STATUS_PATH = Path("/proc/self/status")
KIB_PER_MIB = 1024


def read_process_memory():
    memory = {}
    for line in STATUS_PATH.read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"VmRSS", "VmHWM"}:
            memory[key] = int(value.split()[0])
    return memory["VmRSS"], memory["VmHWM"]


class CPUMemoryMonitor:
    def __init__(self, interval_s=0.10):
        self.interval_s = interval_s
        self.phases = {}

    @contextmanager
    def measure(self, phase):
        start, _ = read_process_memory()
        peak = start
        stop = Event()

        def sample_loop():
            nonlocal peak
            while not stop.wait(self.interval_s):
                rss, _ = read_process_memory()
                peak = max(peak, rss)

        sampler = Thread(target=sample_loop, daemon=True)
        sampler.start()
        try:
            yield
        finally:
            stop.set()
            sampler.join()
            end, _ = read_process_memory()
            peak = max(peak, end)
            self.phases[phase] = {
                "rss_start_mib": start / KIB_PER_MIB,
                "rss_end_mib": end / KIB_PER_MIB,
                "rss_peak_mib": peak / KIB_PER_MIB,
                "rss_peak_increase_mib": (peak - start) / KIB_PER_MIB,
            }

    @staticmethod
    def process_peak_mib():
        _, peak = read_process_memory()
        return peak / KIB_PER_MIB
