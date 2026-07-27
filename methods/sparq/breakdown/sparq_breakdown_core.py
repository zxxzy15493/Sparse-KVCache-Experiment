import sys
import time
import importlib.util
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

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


class SparqBreakdownProfiler:
  """Runtime component timer for the existing SparQ implementation."""

  def __init__(self) -> None:
    self.stats: Dict[str, TimerStat] = defaultdict(TimerStat)
    self.phase = "idle"
    self._patched = False
    self._active_depth = 0
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

  def measure(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if self._active_depth <= 0:
      return fn(*args, **kwargs)
    key = f"{self.phase}_{name}"
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

  def patch_sparq(self) -> None:
    if self._patched:
      return

    from llminference.methods import ann_attention_copy as ann
    from llminference.models import llama_attention02, qwen_attention

    original_forward = ann.AnnAttention.forward
    original_sparseq = ann.SparseQ.forward
    original_lowrank = ann.LowRank.forward
    original_softmax = torch.softmax

    profiler = self

    def forward_wrapper(module, *args, **kwargs):
      return original_forward(module, *args, **kwargs)

    def sparseq_wrapper(module, *args, **kwargs):
      return original_sparseq(module, *args, **kwargs)

    def lowrank_wrapper(module, *args, **kwargs):
      return original_lowrank(module, *args, **kwargs)

    def softmax_wrapper(*args, **kwargs):
      return original_softmax(*args, **kwargs)

    def wrap_flash_attn(original):
      if original is None:
        return None

      def wrapped(*args, **kwargs):
        with self.time_manager.measure("attn"):
          return original(*args, **kwargs)

      return wrapped

    self._patch_attr(ann, "_SPARQ_BREAKDOWN_TIME_MANAGER", self.time_manager)
    self._patch_attr(ann.AnnAttention, "forward", forward_wrapper)
    self._patch_attr(ann.SparseQ, "forward", sparseq_wrapper)
    self._patch_attr(ann.LowRank, "forward", lowrank_wrapper)
    self._patch_attr(torch, "softmax", softmax_wrapper)
    if hasattr(llama_attention02, "flash_attn_func"):
      self._patch_attr(
        llama_attention02,
        "flash_attn_func",
        wrap_flash_attn(getattr(llama_attention02, "flash_attn_func", None)),
      )
    if hasattr(qwen_attention, "flash_attn_func"):
      self._patch_attr(
        qwen_attention,
        "flash_attn_func",
        wrap_flash_attn(getattr(qwen_attention, "flash_attn_func", None)),
      )
    self._patched = True

  def patch_model_modules(self, model: Any) -> None:
    patch_decoder_layers_minference_style(
      model,
      self.time_manager,
      measure_self_attn=False,
    )
    return

  def unpatch(self) -> None:
    while self._orig:
      obj, name, value = self._orig.pop()
      setattr(obj, name, value)
    self._patched = False

  def summary_row(
    self,
    *,
    run_idx: int,
    input_len: int,
    generated_tokens: int,
    model_type: str,
    k: int,
    local_k: int,
    rank: int,
    score: str,
  ) -> Dict[str, Any]:
    row: Dict[str, Any] = {
      "run_idx": run_idx,
      "model_type": model_type,
      "input_len": input_len,
      "generated_tokens": generated_tokens,
      "k": k,
      "local_k": local_k,
      "rank": rank,
      "score": score,
    }
    return add_minference_time_fields(row, self.time_manager)


def write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
  write_important_breakdown_csv(path, rows)
