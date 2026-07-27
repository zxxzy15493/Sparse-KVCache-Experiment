import importlib.util
import sys
import time
import types
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
  sys.path.insert(0, str(METHOD_ROOT))


def _load_breakdown_time_common():
  module_name = f"_breakdown_time_common_{METHOD_ROOT.name.replace('-', '_')}"
  module_path = METHOD_ROOT / "breakdown_time_common.py"
  spec = importlib.util.spec_from_file_location(module_name, module_path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load breakdown_time_common from {module_path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module


_time_common = _load_breakdown_time_common()
MinferenceStyleTimeManager = _time_common.MinferenceStyleTimeManager
add_minference_time_fields = _time_common.add_minference_time_fields
patch_decoder_layers_minference_style = _time_common.patch_decoder_layers_minference_style
write_important_breakdown_csv = _time_common.write_important_breakdown_csv


@dataclass
class TimerStat:
  total: float = 0.0
  count: int = 0

  def add(self, elapsed: float) -> None:
    self.total += elapsed
    self.count += 1

  @property
  def avg(self) -> float:
    return self.total / self.count if self.count else 0.0


class XattentionBreakdownProfiler:
  """Runtime component timer for the existing x-attention implementation."""

  def __init__(self) -> None:
    self.stats: Dict[str, TimerStat] = defaultdict(TimerStat)
    self.phase = "idle"
    self._active_depth = 0
    self._patched = False
    self._orig: List[tuple[Any, str, Any]] = []
    self.time_manager = MinferenceStyleTimeManager()

  def reset_run(self) -> None:
    self.stats = defaultdict(TimerStat)
    self.time_manager.reset()

  @contextmanager
  def _active(self, phase: str):
    old_phase = self.phase
    self.phase = phase
    self._active_depth += 1
    try:
      yield
    finally:
      self._active_depth -= 1
      self.phase = old_phase

  def _sync(self) -> None:
    return

  def measure(self, component: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if self._active_depth <= 0:
      return fn(*args, **kwargs)
    key = f"{self.phase}_{component}"
    self._sync()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    self._sync()
    self.stats[key].add(time.perf_counter() - start)
    return result

  def measure_forward(self, phase: str, model: Any, **kwargs: Any) -> Any:
    with self._active(phase):
      with self.time_manager.measure("total", phase):
        return model(**kwargs)

  def _patch_attr(self, obj: Any, name: str, value: Any) -> None:
    self._orig.append((obj, name, getattr(obj, name)))
    setattr(obj, name, value)

  def _wrap_callable(
    self,
    original: Callable[..., Any],
    component: str,
    time_component: Optional[str] = None,
  ) -> Callable[..., Any]:
    profiler = self

    def wrapped(*args: Any, **kwargs: Any) -> Any:
      if time_component is None:
        return original(*args, **kwargs)
      with profiler.time_manager.measure(time_component):
        return original(*args, **kwargs)

    return wrapped

  def patch_xattn_ops(self, loader_module: Any) -> None:
    if self._patched:
      return

    import xattn.src.Xattention as xattn_mod

    self._patch_attr(
      xattn_mod,
      "Xattention_prefill",
      self._wrap_callable(
        xattn_mod.Xattention_prefill,
        "xattention_prefill",
        "xattention_prefill",
      ),
    )
    if hasattr(loader_module, "Xattention_prefill"):
      self._patch_attr(
        loader_module,
        "Xattention_prefill",
        self._wrap_callable(
          loader_module.Xattention_prefill,
          "xattention_prefill",
          "xattention_prefill",
        ),
      )

    self._patch_attr(
      xattn_mod,
      "xattn_estimate",
      self._wrap_callable(xattn_mod.xattn_estimate, "xattn_estimate"),
    )
    self._patch_attr(
      xattn_mod,
      "find_blocks_chunked",
      self._wrap_callable(
        xattn_mod.find_blocks_chunked,
        "find_blocks_chunked",
        "retrieve",
      ),
    )
    self._patch_attr(
      xattn_mod,
      "block_sparse_attn_func",
      self._wrap_callable(
        xattn_mod.block_sparse_attn_func,
        "block_sparse_attention",
        "attn",
      ),
    )
    loader_flash_attn = getattr(loader_module, "flash_attn_func", None)
    if loader_flash_attn is not None:
      self._patch_attr(
        loader_module,
        "flash_attn_func",
        self._wrap_callable(
          loader_flash_attn,
          "flash_attn_decode",
          "attn",
        ),
      )
    self._patched = True

  def patch_model_modules(self, model: Any) -> None:
    patch_decoder_layers_minference_style(
      model,
      self.time_manager,
      measure_self_attn=False,
    )

  def unpatch(self) -> None:
    while self._orig:
      obj, name, value = self._orig.pop()
      setattr(obj, name, value)
    self._patched = False

  def summary_row(
    self,
    *,
    run_idx: int,
    model_family: str,
    input_len: int,
    generated_tokens: int,
    stride: int,
    threshold: Optional[float],
  ) -> Dict[str, Any]:
    row: Dict[str, Any] = {
      "run_idx": run_idx,
      "model_family": model_family,
      "input_len": input_len,
      "generated_tokens": generated_tokens,
      "stride": stride,
      "threshold": threshold if threshold is not None else "table_threshold",
    }
    row = add_minference_time_fields(row, self.time_manager)
    summary = self.time_manager.summarize()
    for phase in ("prefill", "decode"):
      prefix = f"{phase}_xattention_prefill"
      row[f"{prefix}_time"] = summary.get(f"{prefix}_time", 0.0)
      row[f"{prefix}_event_time"] = summary.get(f"{prefix}_event_time", 0.0)
      row[f"{prefix}_calls"] = int(summary.get(f"{prefix}_calls", 0))
      divisor = max(1, row[f"{prefix}_calls"])
      row[f"{prefix}_avg_time"] = row[f"{prefix}_time"] / divisor
      row[f"{prefix}_avg_event_time"] = row[f"{prefix}_event_time"] / divisor
    return row


def write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
  write_important_breakdown_csv(path, rows)
