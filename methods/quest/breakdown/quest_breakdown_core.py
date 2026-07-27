import inspect
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

# )


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


class QuestBreakdownProfiler:
  """Runtime profiler for LLaMA/Qwen evaluation Quest attention.

  The evaluation implementation monkey-patches Hugging Face attention.forward.
  This profiler follows the same standard: it wraps the original functions and
  modules at runtime, records timing, and does not copy or reimplement the
  Quest attention forward logic.
  """

  def __init__(self) -> None:
    self.phase_stats: Dict[str, PhaseStats] = defaultdict(PhaseStats)
    self.current_phase = "idle"
    self.prefill_total_cpu = 0.0
    self.prefill_total_event = 0.0
    self.decode_total_cpu = 0.0
    self.decode_total_event = 0.0
    self.decode_steps = 0
    self._patched_eval_modules = set()
    self._patched_external_kernels = []
    self.time_manager = MinferenceStyleTimeManager()

  def _wrap_external_kernel(self, module, name: str, marker: str) -> None:
    if module is None or not hasattr(module, name):
      return
    original = getattr(module, name)
    if getattr(original, marker, False):
      return

    def timed_kernel(*args, **kwargs):
      with self.time_manager.measure("attn"):
        return original(*args, **kwargs)

    setattr(timed_kernel, marker, True)
    setattr(module, name, timed_kernel)
    self._patched_external_kernels.append((module, name, original))

  def _patch_llama_flash_kernel(self) -> None:
    llama_module = sys.modules.get("evaluation.llama")
    if llama_module is not None and hasattr(llama_module, "_QUEST_BREAKDOWN_TIME_MANAGER"):
      setattr(llama_module, "_QUEST_BREAKDOWN_TIME_MANAGER", self.time_manager)
    self._wrap_external_kernel(
      llama_module,
      "flash_attn_func",
      "_quest_kernel_breakdown_patched",
    )

  def _patch_qwen_flash_kernel(self) -> None:
    try:
      from transformers.models.qwen2 import modeling_qwen2
    except Exception:
      return
    self._wrap_external_kernel(
      modeling_qwen2,
      "_flash_attention_forward",
      "_quest_kernel_breakdown_patched",
    )

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

  def patch_eval_module(self, eval_module) -> None:
    module_key = id(eval_module)
    if module_key in self._patched_eval_modules:
      return

    def wrap_function(name: str, component: str):
      if not hasattr(eval_module, name):
        return
      original = getattr(eval_module, name)
      if getattr(original, "_quest_eval_breakdown_patched", False):
        return

      def wrapped(*args, **kwargs):
        time_component = None
        if component == "local_heavy_hitter_mask":
          time_component = "retrieve"

        if time_component is None:
          return original(*args, **kwargs)
        with self.time_manager.measure(time_component):
          return original(*args, **kwargs)

      wrapped._quest_eval_breakdown_patched = True
      setattr(eval_module, name, wrapped)

    wrap_function("apply_rotary_pos_emb", "rope")
    wrap_function("repeat_kv", "repeat_kv")
    wrap_function("quest_select_token_indices_from_chunks", "chunk_score_select")
    wrap_function("local_heavy_hitter_mask", "local_heavy_hitter_mask")
    wrap_function("calculate_recall_from_mask", "recall_metric")
    wrap_function("calculate_topkrate", "topkrate_metric")
    if hasattr(eval_module, "_QUEST_BREAKDOWN_TIME_MANAGER"):
      setattr(eval_module, "_QUEST_BREAKDOWN_TIME_MANAGER", self.time_manager)
    self._patch_llama_flash_kernel()
    self._patch_qwen_flash_kernel()
    self._patched_eval_modules.add(module_key)

  def patch_model_modules(self, model) -> None:
    self._patch_llama_flash_kernel()
    self._patch_qwen_flash_kernel()
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

  def summary_row(
    self,
    run_idx: int,
    input_len: int,
    generated_tokens: int,
    quest_module: str = "",
    token_budget: Optional[int] = None,
    chunk_size: Optional[int] = None,
  ) -> dict:
    row = {
      "run_idx": run_idx,
      "input_len": input_len,
      "generated_tokens": generated_tokens,
      "quest_module": quest_module,
      "token_budget": token_budget if token_budget is not None else "",
      "chunk_size": chunk_size if chunk_size is not None else "",
    }
    return add_minference_time_fields(row, self.time_manager)


def write_csv(path: str, rows: Iterable[dict]) -> None:
  write_important_breakdown_csv(path, rows)
