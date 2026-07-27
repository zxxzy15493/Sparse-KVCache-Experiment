import csv
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


COMPONENTS = (
  "prepare_metadata",
  "begin_forward",
  "end_forward",
  "attn_total",
  "qkv_proj",
  "rope",
  "append_kv_prefill",
  "append_kv_decode",
  "prefill_attention",
  "decode_estimate",
  "decode_topk",
  "decode_full_attn",
  "decode_approx_attn",
  "o_proj",
  "rms_norm",
  "mlp",
)


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


class QuestKernelBreakdownProfiler:
  """Runtime profiler for Quest kernel components.

  This profiler does not reimplement Quest. It wraps the original kernel-path
  Python functions and model modules, then calls the original implementation
  unchanged while recording component-level timing.
  """

  def __init__(self) -> None:
    self.phase_stats: Dict[str, PhaseStats] = defaultdict(PhaseStats)
    self.current_phase = "idle"
    self.prefill_total_cpu = 0.0
    self.prefill_total_event = 0.0
    self.decode_total_cpu = 0.0
    self.decode_total_event = 0.0
    self.decode_steps = 0
    self._patched_ops = False
    self._patched_controller = False
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

  def patch_quest_ops(self) -> None:
    if self._patched_ops:
      return

    import quest.utils as quest_utils

    def wrap_op(name: str, component: str):
      original = getattr(quest_utils, name)
      if getattr(original, "_quest_kernel_breakdown_patched", False):
        return

      def wrapped(*args, **kwargs):
        time_component = None
        if component == "decode_topk":
          time_component = "index_build"
        elif component == "prefill_attention":
          time_component = "attn"

        if time_component is None:
          return original(*args, **kwargs)
        with self.time_manager.measure(time_component):
          return original(*args, **kwargs)

      wrapped._quest_kernel_breakdown_patched = True
      setattr(quest_utils, name, wrapped)

    def wrap_append_kv():
      original = quest_utils.append_kv
      if getattr(original, "_quest_kernel_breakdown_patched", False):
        return

      def wrapped(k, *args, **kwargs):
        with self.time_manager.measure("write_cache"):
          return original(k, *args, **kwargs)

      wrapped._quest_kernel_breakdown_patched = True
      quest_utils.append_kv = wrapped

    def wrap_decode_sparse_attn():
      original = quest_utils.decode_sparse_attn
      if getattr(original, "_quest_kernel_breakdown_patched", False):
        return

      def wrapped(q, iController, layer_idx, topk_indices, *args, **kwargs):
        component = "decode_full_attn"
        if topk_indices is getattr(iController, "topk_dindices_buffer", None):
          component = "decode_approx_attn"
        elif topk_indices is getattr(iController, "kv_indices_without_last", None):
          component = "decode_full_attn"
        with self.time_manager.measure("attn"):
          return original(q, iController, layer_idx, topk_indices, *args, **kwargs)

      wrapped._quest_kernel_breakdown_patched = True
      quest_utils.decode_sparse_attn = wrapped

    wrap_op("apply_rope_in_place", "rope")
    wrap_append_kv()
    wrap_op("prefill_forward", "prefill_attention")
    wrap_op("decode_estimate", "decode_estimate")
    wrap_op("decode_topk", "decode_topk")
    wrap_decode_sparse_attn()
    self._patched_ops = True

  def patch_controller(self) -> None:
    if self._patched_controller:
      return

    from quest.utils.controller import InferenceController

    def wrap_method(name: str, component: str):
      original = getattr(InferenceController, name)
      if getattr(original, "_quest_kernel_breakdown_patched", False):
        return

      def wrapped(this, *args, **kwargs):
        time_component = "index_build" if component in {"prepare_metadata", "begin_forward"} else None
        if time_component is None:
          return original(this, *args, **kwargs)
        with self.time_manager.measure(time_component):
          return original(this, *args, **kwargs)

      wrapped._quest_kernel_breakdown_patched = True
      setattr(InferenceController, name, wrapped)

    wrap_method("prepare_metadata", "prepare_metadata")
    wrap_method("begin_forward", "begin_forward")
    wrap_method("end_forward", "end_forward")
    self._patched_controller = True

  def patch_model_modules(self, model) -> None:
    patch_decoder_layers_minference_style(
      model,
      self.time_manager,
      measure_self_attn=False,
    )
    return
    for name, module in model.named_modules():
      if module.__class__.__name__ == "QuestAttention":
        self._patch_attention_forward(module)

      if name.endswith(("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")):
        self._patch_module_forward(module, "qkv_proj", "_quest_kernel_qkv_patched")

      if name.endswith("self_attn.o_proj"):
        self._patch_module_forward(module, "o_proj", "_quest_kernel_oproj_patched")

      if name.endswith(("input_layernorm", "post_attention_layernorm", ".norm")):
        self._patch_module_forward(module, "rms_norm", "_quest_kernel_norm_patched")

      if name.endswith("mlp") and hasattr(module, "forward"):
        self._patch_module_forward(module, "mlp", "_quest_kernel_mlp_patched")

  def _patch_attention_forward(self, module) -> None:
    if getattr(module, "_quest_kernel_attn_patched", False):
      return
    original_forward = module.forward

    def attn_forward(this, *args, _orig=original_forward, **kwargs):
      hidden_states = args[0] if args else kwargs.get("hidden_states")
      q_len = int(hidden_states.shape[1]) if hidden_states is not None else 0
      phase = "prefill" if q_len > 1 else "decode"

      with self.phase(phase), self.measure("attn_total", phase):
        return _orig(*args, **kwargs)

    module.forward = types.MethodType(attn_forward, module)
    module._quest_kernel_attn_patched = True

  def _patch_module_forward(self, module, component: str, flag: str) -> None:
    if getattr(module, flag, False):
      return
    original_forward = module.forward

    def wrapped_forward(this, *args, _orig=original_forward, **kwargs):
      return _orig(*args, **kwargs)

    module.forward = types.MethodType(wrapped_forward, module)
    setattr(module, flag, True)

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

  def summary_row(self, run_idx: int, input_len: int, generated_tokens: int) -> dict:
    row = {
      "run_idx": run_idx,
      "input_len": input_len,
      "generated_tokens": generated_tokens,
      "prefill_total_time": self.prefill_total_cpu,
      "prefill_total_event_time": self.prefill_total_event,
      "decode_total_avg_time": self.decode_total_cpu / max(1, self.decode_steps),
      "decode_total_avg_event_time": self.decode_total_event / max(1, self.decode_steps),
    }
    for phase in ("prefill", "decode"):
      for component in COMPONENTS:
        stat = self.phase_stats[phase].components[component]
        prefix = f"{phase}_{component}"
        value_cpu = stat.cpu_time
        value_event = stat.event_time
        if phase == "decode":
          value_cpu /= max(1, self.decode_steps)
          value_event /= max(1, self.decode_steps)
        row[f"{prefix}_time"] = value_cpu
        row[f"{prefix}_event_time"] = value_event
        row[f"{prefix}_calls"] = stat.calls
    return add_minference_time_fields(row, self.time_manager)


def write_csv(path: str, rows: Iterable[dict]) -> None:
  rows = list(rows)
  if not rows:
    return
  fieldnames: List[str] = list(rows[0].keys())
  with open(path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
