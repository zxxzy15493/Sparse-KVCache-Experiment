import importlib.util
import sys
import time
import types
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
class ComponentStat:
  cpu_time: float = 0.0
  event_time: float = 0.0
  calls: int = 0

  def add(self, cpu_time: float, event_time: float) -> None:
    self.cpu_time += cpu_time
    self.event_time += event_time
    self.calls += 1


@dataclass
class PhaseStats:
  components: Dict[str, ComponentStat] = field(
    default_factory=lambda: defaultdict(ComponentStat)
  )


class FlexprefillBreakdownProfiler:
  """Runtime profiler for FlexPrefill-specific components.

  The profiler intentionally patches functions at runtime instead of editing the
  package implementation. It measures the components introduced by FlexPrefill:
  active-block selection, block-wise sparse attention, dense fallback/decode,
  optional metric calculation, plus layer-level attention and MLP time.
  """

  def __init__(self) -> None:
    self.phase_stats: Dict[str, PhaseStats] = defaultdict(PhaseStats)
    self.current_phase = "idle"
    self.prefill_total_cpu = 0.0
    self.prefill_total_event = 0.0
    self.decode_total_cpu = 0.0
    self.decode_total_event = 0.0
    self.decode_steps = 0
    self._patched = False
    self._orig_ops = {}
    self._orig_module_flex = {}
    self.time_manager = MinferenceStyleTimeManager()

  @contextmanager
  def phase(self, phase: str):
    old_phase = self.current_phase
    self.current_phase = phase
    try:
      yield
    finally:
      self.current_phase = old_phase

  @contextmanager
  def measure(self, component: str, phase: Optional[str] = None):
    phase_name = phase or self.current_phase
    start_cpu = time.perf_counter()
    try:
      yield
    finally:
      cpu_time = time.perf_counter() - start_cpu
      event_time = 0.0
      self.phase_stats[phase_name].components[component].add(
        cpu_time, event_time
      )

  def reset_run(self) -> None:
    self.phase_stats = defaultdict(PhaseStats)
    self.current_phase = "idle"
    self.prefill_total_cpu = 0.0
    self.prefill_total_event = 0.0
    self.decode_total_cpu = 0.0
    self.decode_total_event = 0.0
    self.decode_steps = 0
    self.time_manager.reset()

  def patch_flex_ops(self) -> None:
    if self._patched:
      return

    import flex_prefill.ops.flex_prefill_attention as ops
    setattr(ops, "_FLEXPREFILL_BREAKDOWN_TIME_MANAGER", self.time_manager)

    def wrap_op(name: str, component: str):
      original = getattr(ops, name)
      self._orig_ops[name] = original

      def wrapped(*args, **kwargs):
        time_component = None
        if name == "get_active_blocks":
          time_component = "retrieve"
        elif name in {
          "triton_block_wise_attention",
          "triton_flash_attention",
          "flash_attn_func",
        }:
          time_component = "attn"

        if time_component is None:
          return original(*args, **kwargs)
        with self.time_manager.measure(time_component):
          return original(*args, **kwargs)

      setattr(ops, name, wrapped)

    wrap_op("get_active_blocks", "active_blocks")
    wrap_op("triton_block_wise_attention", "sparse_attention")
    wrap_op("triton_flash_attention", "dense_attention")
    wrap_op("flash_attn_func", "dense_attention")
    wrap_op("calculate_flexprefill_layer_captured_mass", "metric_captured_mass")

    original_flex = ops.flex_prefill_attention
    self._orig_ops["flex_prefill_attention"] = original_flex

    def timed_flex_prefill_attention(*args, **kwargs):
      return original_flex(*args, **kwargs)

    ops.flex_prefill_attention = timed_flex_prefill_attention

    for module_name in (
      "flex_prefill.modules.qwen2.flex_prefill_attention",
      "flex_prefill.modules.llama.flex_prefill_attention",
      "flex_prefill.modules.glm.flex_prefill_attention",
    ):
      try:
        module = __import__(module_name, fromlist=["flex_prefill_attention"])
      except Exception:
        continue
      if hasattr(module, "flex_prefill_attention"):
        self._orig_module_flex[module_name] = module.flex_prefill_attention
        module.flex_prefill_attention = timed_flex_prefill_attention

    self._patched = True

  def patch_model_modules(self, model) -> None:
    patch_decoder_layers_minference_style(
      model,
      self.time_manager,
      measure_self_attn=False,
    )

  def measure_forward(self, phase: str, fn, *args, **kwargs):
    start_cpu = time.perf_counter()
    with self.phase(phase):
      with self.time_manager.measure("total", phase):
        result = fn(*args, **kwargs)
    cpu_time = time.perf_counter() - start_cpu
    event_time = 0.0

    if phase == "prefill":
      self.prefill_total_cpu += cpu_time
      self.prefill_total_event += event_time
    elif phase == "decode":
      self.decode_total_cpu += cpu_time
      self.decode_total_event += event_time
      self.decode_steps += 1
    return result

  def _subtract_nested_pattern_from_retrieve(self, row: dict) -> None:
    """Keep FlexPrefill mode assignment out of retrieve after nested timing."""

    decode_divisor = max(1, int(row.get("decode_steps", 0) or 0))
    for phase in ("prefill", "decode"):
      retrieve_time_key = f"{phase}_retrieve_time"
      retrieve_event_key = f"{phase}_retrieve_event_time"
      pattern_time = float(row.get(f"{phase}_pattern_time", 0.0) or 0.0)
      pattern_event = float(row.get(f"{phase}_pattern_event_time", 0.0) or 0.0)
      retrieve_time = float(row.get(retrieve_time_key, 0.0) or 0.0)
      retrieve_event = float(row.get(retrieve_event_key, 0.0) or 0.0)

      if pattern_time > 0.0 and retrieve_time > 0.0:
        row[retrieve_time_key] = max(0.0, retrieve_time - pattern_time)
      if pattern_event > 0.0 and retrieve_event > 0.0:
        row[retrieve_event_key] = max(0.0, retrieve_event - pattern_event)

      divisor = 1 if phase == "prefill" else decode_divisor
      row[f"{phase}_retrieve_avg_time"] = row[retrieve_time_key] / divisor
      row[f"{phase}_retrieve_avg_event_time"] = row[retrieve_event_key] / divisor

      selection_time = row[f"{phase}_retrieve_time"] + row[f"{phase}_index_build_time"]
      selection_event_time = (
        row[f"{phase}_retrieve_event_time"]
        + row[f"{phase}_index_build_event_time"]
      )
      row[f"{phase}_selection_time"] = selection_time
      row[f"{phase}_selection_event_time"] = selection_event_time
      row[f"{phase}_selection_avg_time"] = selection_time / divisor
      row[f"{phase}_selection_avg_event_time"] = selection_event_time / divisor

    layer_divisor = max(1, int(getattr(self.time_manager, "num_layers", 0) or 0))
    row["prefill_retrieve_layer_avg_time"] = (
      row["prefill_retrieve_time"] / layer_divisor
    )
    row["prefill_retrieve_layer_avg_event_time"] = (
      row["prefill_retrieve_event_time"] / layer_divisor
    )
    for component in ("selection", "retrieve"):
      row[f"prefill_{component}_breakdown_time"] = row[
        f"prefill_{component}_time"
      ]
      row[f"prefill_{component}_breakdown_event_time"] = row[
        f"prefill_{component}_event_time"
      ]
      row[f"decode_{component}_breakdown_time"] = (
        row[f"decode_{component}_time"] / decode_divisor
      )
      row[f"decode_{component}_breakdown_event_time"] = (
        row[f"decode_{component}_event_time"] / decode_divisor
      )

  def summary_row(
    self,
    run_idx: int,
    input_len: int,
    generated_tokens: int,
    block_size: Optional[int] = None,
    gamma: Optional[float] = None,
    tau: Optional[float] = None,
    min_budget: Optional[int] = None,
    max_budget: Optional[int] = None,
  ) -> dict:
    row = {
      "run_idx": run_idx,
      "input_len": input_len,
      "generated_tokens": generated_tokens,
      "block_size": block_size if block_size is not None else "",
      "gamma": gamma if gamma is not None else "",
      "tau": tau if tau is not None else "",
      "min_budget": min_budget if min_budget is not None else "",
      "max_budget": max_budget if max_budget is not None else "",
    }
    row = add_minference_time_fields(row, self.time_manager)
    self._subtract_nested_pattern_from_retrieve(row)
    return row


def write_csv(path: str, rows: Iterable[dict]) -> None:
  write_important_breakdown_csv(path, rows)
