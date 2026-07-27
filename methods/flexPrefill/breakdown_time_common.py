import csv
import os
import time
import types
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


_MISSING = object()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


IMPORTANT_BREAKDOWN_FIELDNAMES = [
    "run_idx",
    "model_type",
    "model_family",
    "quest_module",
    "input_len",
    "generated_tokens",
    "decode_steps",
    "k",
    "local_k",
    "rank",
    "score",
    "token_budget",
    "chunk_size",
    "stride",
    "threshold",
    "no_repeatkv",
    "block_size",
    "min_budget",
    "max_budget",
    "gamma",
    "tau",
    "prefill_time",
    "prefill_event_time",
    "decode_avg_time",
    "decode_avg_event_time",
    "prefill_attn_time",
    "prefill_attn_event_time",
    "decode_attn_avg_time",
    "decode_attn_avg_event_time",
    "prefill_write_cache_time",
    "prefill_write_cache_event_time",
    "decode_write_cache_avg_time",
    "decode_write_cache_avg_event_time",
    "prefill_retrieve_time",
    "prefill_retrieve_event_time",
    "decode_retrieve_avg_time",
    "decode_retrieve_avg_event_time",
    "prefill_pattern_time",
    "prefill_pattern_event_time",
    "decode_pattern_avg_time",
    "decode_pattern_avg_event_time",
    "prefill_other_time",
    "prefill_other_event_time",
    "decode_other_avg_time",
    "decode_other_avg_event_time",
    "prefill_index_build_time",
    "prefill_index_build_event_time",
    "decode_index_build_avg_time",
    "decode_index_build_avg_event_time",
]


def _first_present(row: Dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name, _MISSING)
        if value is not _MISSING and value is not None and value != "":
            return value
    return ""


def important_breakdown_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return the compact CSV row used by breakdown experiments."""

    compact = {name: row.get(name, "") for name in IMPORTANT_BREAKDOWN_FIELDNAMES}
    compact.update(
        {
            "prefill_time": _first_present(
                row, "prefill_time", "prefill_total_time", "prefill_latency"
            ),
            "prefill_event_time": _first_present(
                row,
                "prefill_event_time",
                "prefill_total_event_time",
                "prefill_latency_event_time",
            ),
            "decode_avg_time": _first_present(
                row,
                "decode_time",
                "decode_total_avg_time",
                "decode_latency_avg_time",
            ),
            "decode_avg_event_time": _first_present(
                row,
                "decode_event_time",
                "decode_total_avg_event_time",
                "decode_latency_avg_event_time",
            ),
            "prefill_attn_time": _first_present(
                row, "prefill_attn_breakdown_time", "prefill_attn_time"
            ),
            "prefill_attn_event_time": _first_present(
                row,
                "prefill_attn_breakdown_event_time",
                "prefill_attn_event_time",
            ),
            "decode_attn_avg_time": _first_present(
                row,
                "decode_attn_breakdown_time",
                "decode_attn_avg_time",
                "decode_attn_time",
            ),
            "decode_attn_avg_event_time": _first_present(
                row,
                "decode_attn_breakdown_event_time",
                "decode_attn_avg_event_time",
                "decode_attn_event_time",
            ),
            "prefill_write_cache_time": _first_present(
                row,
                "prefill_write_cache_breakdown_time",
                "prefill_write_cache_time",
            ),
            "prefill_write_cache_event_time": _first_present(
                row,
                "prefill_write_cache_breakdown_event_time",
                "prefill_write_cache_event_time",
            ),
            "decode_write_cache_avg_time": _first_present(
                row,
                "decode_write_cache_breakdown_time",
                "decode_write_cache_avg_time",
                "decode_write_cache_time",
            ),
            "decode_write_cache_avg_event_time": _first_present(
                row,
                "decode_write_cache_breakdown_event_time",
                "decode_write_cache_avg_event_time",
                "decode_write_cache_event_time",
            ),
            "prefill_retrieve_time": _first_present(
                row,
                "prefill_retrieve_breakdown_time",
                "prefill_retrieve_time",
            ),
            "prefill_retrieve_event_time": _first_present(
                row,
                "prefill_retrieve_breakdown_event_time",
                "prefill_retrieve_event_time",
            ),
            "decode_retrieve_avg_time": _first_present(
                row,
                "decode_retrieve_breakdown_time",
                "decode_retrieve_avg_time",
                "decode_retrieve_time",
            ),
            "decode_retrieve_avg_event_time": _first_present(
                row,
                "decode_retrieve_breakdown_event_time",
                "decode_retrieve_avg_event_time",
                "decode_retrieve_event_time",
            ),
            "prefill_pattern_time": _first_present(
                row,
                "prefill_pattern_breakdown_time",
                "prefill_pattern_time",
            ),
            "prefill_pattern_event_time": _first_present(
                row,
                "prefill_pattern_breakdown_event_time",
                "prefill_pattern_event_time",
            ),
            "decode_pattern_avg_time": _first_present(
                row,
                "decode_pattern_breakdown_time",
                "decode_pattern_avg_time",
                "decode_pattern_time",
            ),
            "decode_pattern_avg_event_time": _first_present(
                row,
                "decode_pattern_breakdown_event_time",
                "decode_pattern_avg_event_time",
                "decode_pattern_event_time",
            ),
            "prefill_other_time": _first_present(
                row,
                "prefill_other_breakdown_time",
                "prefill_other_time",
            ),
            "prefill_other_event_time": _first_present(
                row,
                "prefill_other_breakdown_event_time",
                "prefill_other_event_time",
            ),
            "decode_other_avg_time": _first_present(
                row,
                "decode_other_breakdown_time",
                "decode_other_avg_time",
                "decode_other_time",
            ),
            "decode_other_avg_event_time": _first_present(
                row,
                "decode_other_breakdown_event_time",
                "decode_other_avg_event_time",
                "decode_other_event_time",
            ),
            "prefill_index_build_time": _first_present(
                row, "prefill_index_build_time"
            ),
            "prefill_index_build_event_time": _first_present(
                row, "prefill_index_build_event_time"
            ),
            "decode_index_build_avg_time": _first_present(
                row, "decode_index_build_avg_time", "decode_index_build_time"
            ),
            "decode_index_build_avg_event_time": _first_present(
                row,
                "decode_index_build_avg_event_time",
                "decode_index_build_event_time",
            ),
        }
    )
    return compact


def write_important_breakdown_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    compact_rows = [important_breakdown_row(row) for row in rows]
    if not compact_rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=IMPORTANT_BREAKDOWN_FIELDNAMES)
        writer.writeheader()
        writer.writerows(compact_rows)


@dataclass
class EventRecord:
    phase: str
    component: str
    start_event: Optional[torch.cuda.Event]
    end_event: Optional[torch.cuda.Event]
    start_cpu: float
    end_cpu: float = 0.0
    nvtx_pushed: bool = False


class MinferenceStyleTimeManager:
    """CUDA-event timing with the same layer split used by MInference.

    Step 0 is treated as prefill. Steps >= 1 are decode steps, matching
    MInference's `decode_step * num_layers + layer_idx` event indexing logic.
    """

    def __init__(self) -> None:
        self.decode_step = 0
        self.num_layers = 0
        self.records: List[EventRecord] = []
        self.enable_nvtx = _env_flag("BREAKDOWN_NVTX")
        self.sync_timing = _env_flag("BREAKDOWN_SYNC_TIMING")
        self._active_stack: List[Tuple[str, str]] = []
        self._other_context_depth = 0
        self._other_pause_depth = 0
        self._other_phase_stack: List[str] = []
        self._other_component_stack: List[str] = []
        self._active_other: Optional[EventRecord] = None

    def reset(self) -> None:
        self.decode_step = 0
        self.records = []
        self._active_stack = []
        self._other_context_depth = 0
        self._other_pause_depth = 0
        self._other_phase_stack = []
        self._other_component_stack = []
        self._active_other = None

    @property
    def current_phase(self) -> str:
        return "prefill" if self.decode_step == 0 else "decode"

    def _new_event(self) -> Optional[torch.cuda.Event]:
        if not torch.cuda.is_available():
            return None
        return torch.cuda.Event(enable_timing=True)

    def _sync_if_needed(self) -> None:
        if self.sync_timing and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _begin_record(self, phase: str, component: str) -> EventRecord:
        self._sync_if_needed()
        if self.sync_timing:
            start_event = None
            end_event = None
        else:
            start_event = self._new_event()
            end_event = self._new_event()
        nvtx_pushed = False
        if self.enable_nvtx and torch.cuda.is_available():
            torch.cuda.nvtx.range_push(f"{phase}:{component}")
            nvtx_pushed = True
        if start_event is not None:
            start_event.record()
        return EventRecord(
            phase=phase,
            component=component,
            start_event=start_event,
            end_event=end_event,
            start_cpu=time.perf_counter(),
            nvtx_pushed=nvtx_pushed,
        )

    def _finish_record(self, record: EventRecord) -> None:
        if record.end_event is not None:
            record.end_event.record()
        self._sync_if_needed()
        record.end_cpu = time.perf_counter()
        if record.nvtx_pushed:
            torch.cuda.nvtx.range_pop()
        self.records.append(record)

    def _current_other_component(self) -> str:
        if not self._other_component_stack:
            return "other"
        return self._other_component_stack[-1]

    def _start_other_span(self, phase: str) -> None:
        if self._active_other is None:
            self._active_other = self._begin_record(
                phase,
                self._current_other_component(),
            )

    def _stop_other_span(self) -> None:
        if self._active_other is not None:
            record = self._active_other
            self._active_other = None
            self._finish_record(record)

    def _pauses_other(self, component: str) -> bool:
        return component in {"attn", "retrieve", "index_build", "pattern", "write_cache"}

    def _pause_other_for_component(self, component: str) -> bool:
        if self._other_context_depth <= 0 or not self._pauses_other(component):
            return False
        if self._other_pause_depth == 0:
            self._stop_other_span()
        self._other_pause_depth += 1
        return True

    def _resume_other_after_component(self, component: str, paused: bool) -> None:
        if not paused:
            return
        self._other_pause_depth -= 1
        if self._other_pause_depth == 0 and self._other_context_depth > 0:
            if component == "attn":
                self._other_component_stack[-1] = "postAttn_ffn"
                self._start_other_span(self._other_phase_stack[-1])
            elif self._current_other_component() == "postAttn_ffn":
                self._start_other_span(self._other_phase_stack[-1])

    def _resume_same_other_after_component(self, paused: bool) -> None:
        if not paused:
            return
        self._other_pause_depth -= 1
        if self._other_pause_depth == 0 and self._other_context_depth > 0:
            self._start_other_span(self._other_phase_stack[-1])

    @contextmanager
    def measure(self, component: str, phase: Optional[str] = None):
        phase_name = phase or self.current_phase
        paused_other = self._pause_other_for_component(component)
        record = self._begin_record(phase_name, component)
        self._active_stack.append((phase_name, component))
        try:
            yield
        finally:
            self._finish_record(record)
            self._active_stack.pop()
            self._resume_other_after_component(component, paused_other)

    @contextmanager
    def measure_pre_attn_component(self, component: str, phase: Optional[str] = None):
        phase_name = phase or self.current_phase
        paused_other = self._pause_other_for_component(component)
        record = self._begin_record(phase_name, component)
        self._active_stack.append((phase_name, component))
        try:
            yield
        finally:
            self._finish_record(record)
            self._active_stack.pop()
            self._resume_same_other_after_component(paused_other)

    def ensure_post_attn_other(self, phase: Optional[str] = None) -> None:
        """Start post-attention other timing if no attention kernel opened it."""

        if self._other_context_depth <= 0 or self._other_pause_depth > 0:
            return
        phase_name = phase or self.current_phase
        if self._active_other is not None:
            if self._active_other.component == "postAttn_ffn":
                return
            self._stop_other_span()
        self._other_component_stack[-1] = "postAttn_ffn"
        self._start_other_span(phase_name)

    @contextmanager
    def measure_other_exclusive(self, phase: Optional[str] = None):
        """Measure PQCache-style pre/post other spans around core stages.

        preAttn_ffn starts before input_layernorm and continues through the
        attention module's qkv/RoPE work until the first measured core stage
        (write_cache/retrieve/index_build/attn). postAttn_ffn starts after the
        measured attention kernel and covers o_proj plus the decoder-layer FFN.
        """

        phase_name = phase or self.current_phase
        self._other_context_depth += 1
        self._other_phase_stack.append(phase_name)
        self._other_component_stack.append("preAttn_ffn")
        if self._other_context_depth == 1 and self._other_pause_depth == 0:
            self._start_other_span(phase_name)
        try:
            yield
        finally:
            if self._other_context_depth == 1:
                self._stop_other_span()
            self._other_phase_stack.pop()
            self._other_component_stack.pop()
            self._other_context_depth -= 1

    def summarize(self) -> Dict[str, float]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        totals: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)

        for record in self.records:
            prefix = f"{record.phase}_{record.component}"
            cpu_time = record.end_cpu - record.start_cpu
            if self.sync_timing:
                event_time = cpu_time
            elif record.start_event is not None and record.end_event is not None:
                event_time = record.start_event.elapsed_time(record.end_event) / 1000.0
            else:
                event_time = 0.0
            totals[f"{prefix}_time"] += cpu_time
            totals[f"{prefix}_event_time"] += event_time
            counts[f"{prefix}_calls"] += 1

        out: Dict[str, float] = {}
        for key, value in totals.items():
            out[key] = value
        for key, value in counts.items():
            out[key] = value
        return out


@contextmanager
def measure_cache_update_calls(
    manager: MinferenceStyleTimeManager,
    *cache_objects: Optional[Any],
):
    """Measure KV-cache writes without changing the attention implementation."""

    restores = []
    seen = set()

    def patch_cache(cache: Any) -> None:
        if cache is None or id(cache) in seen or not hasattr(cache, "update"):
            return
        seen.add(id(cache))

        original_update = getattr(cache, "update")
        if getattr(original_update, "_breakdown_write_cache_patched", False):
            return

        def timed_update(*args: Any, **kwargs: Any):
            with manager.measure("write_cache"):
                return original_update(*args, **kwargs)

        timed_update._breakdown_write_cache_patched = True
        cache_dict = getattr(cache, "__dict__", None)
        old_instance_update = (
            cache_dict.get("update", _MISSING)
            if isinstance(cache_dict, dict)
            else _MISSING
        )

        try:
            setattr(cache, "update", timed_update)
        except (AttributeError, TypeError):
            cache_cls = cache.__class__
            class_update = getattr(cache_cls, "update", None)
            if class_update is None or getattr(
                class_update, "_breakdown_write_cache_patched", False
            ):
                return

            def timed_class_update(this: Any, *args: Any, **kwargs: Any):
                with manager.measure("write_cache"):
                    return class_update(this, *args, **kwargs)

            timed_class_update._breakdown_write_cache_patched = True
            setattr(cache_cls, "update", timed_class_update)
            restores.append((cache_cls, "update", class_update, True))
            return

        restores.append((cache, "update", old_instance_update, False))

    for cache_object in cache_objects:
        patch_cache(cache_object)

    try:
        yield
    finally:
        for obj, name, old_value, is_class_patch in reversed(restores):
            if is_class_patch:
                setattr(obj, name, old_value)
            elif old_value is _MISSING:
                try:
                    delattr(obj, name)
                except (AttributeError, TypeError):
                    pass
            else:
                setattr(obj, name, old_value)


def patch_decoder_layers_minference_style(
    model: Any,
    manager: MinferenceStyleTimeManager,
    *,
    measure_self_attn: bool = True,
) -> None:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return

    manager.num_layers = len(layers)
    for layer_idx, layer in enumerate(layers):
        if getattr(layer, "_breakdown_time_decoder_patched", False):
            continue
        if not all(hasattr(layer, attr) for attr in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")):
            continue

        layer.layer_idx = layer_idx
        layer.num_layers = len(layers)
        original_cls = layer.__class__

        def decoder_forward(
            this,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Any] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs: Any,
        ):
            with manager.measure_other_exclusive():
                with manager.measure("pre_attn_norm"):
                    residual = hidden_states
                    hidden_states = this.input_layernorm(hidden_states)

                attn_kwargs = {
                    "hidden_states": hidden_states,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "past_key_value": past_key_value,
                    "output_attentions": output_attentions,
                    "use_cache": use_cache,
                    **kwargs,
                }
                if cache_position is not None:
                    attn_kwargs["cache_position"] = cache_position
                if position_embeddings is not None:
                    attn_kwargs["position_embeddings"] = position_embeddings

                module_cache = getattr(this.self_attn, "past_key_value", None)
                with measure_cache_update_calls(manager, past_key_value, module_cache):
                    if measure_self_attn:
                        with manager.measure("attn"):
                            attn_outputs = this.self_attn(**attn_kwargs)
                    else:
                        attn_outputs = this.self_attn(**attn_kwargs)
                manager.ensure_post_attn_other()

                hidden_states = attn_outputs[0]
                self_attn_weights = attn_outputs[1] if len(attn_outputs) > 1 else None
                present_key_value = attn_outputs[2] if len(attn_outputs) > 2 else None

                with manager.measure("ffn"):
                    hidden_states = residual + hidden_states
                    residual = hidden_states
                    hidden_states = this.post_attention_layernorm(hidden_states)
                    hidden_states = this.mlp(hidden_states)
                    hidden_states = residual + hidden_states

            outputs = (hidden_states,)
            if output_attentions:
                outputs += (self_attn_weights,)
            if use_cache:
                outputs += (present_key_value,)

            if this.layer_idx == this.num_layers - 1:
                manager.decode_step += 1
            return outputs

        layer.forward = types.MethodType(decoder_forward, layer)
        layer._breakdown_time_decoder_patched = True
        layer._breakdown_time_original_cls = original_cls


def add_minference_time_fields(row: Dict[str, Any], manager: MinferenceStyleTimeManager) -> Dict[str, Any]:
    summary = manager.summarize()
    row.setdefault("prefill_latency", row.get("prefill_total_time", 0.0))
    row.setdefault("prefill_latency_event_time", row.get("prefill_total_event_time", 0.0))
    row.setdefault("decode_latency_avg_time", row.get("decode_total_avg_time", 0.0))
    row.setdefault("decode_latency_avg_event_time", row.get("decode_total_avg_event_time", 0.0))
    components = (
        "total",
        "pre_attn_norm",
        "attn",
        "ffn",
        "index_build",
        "pattern",
        "retrieve",
        "write_cache",
        "preAttn_ffn",
        "postAttn_ffn",
    )
    row["decode_steps"] = max(0, manager.decode_step - 1)
    decode_divisor = max(1, row["decode_steps"])
    for phase in ("prefill", "decode"):
        for component in components:
            prefix = f"{phase}_{component}"
            row[f"{prefix}_time"] = summary.get(f"{prefix}_time", 0.0)
            row[f"{prefix}_event_time"] = summary.get(f"{prefix}_event_time", 0.0)
            row[f"{prefix}_calls"] = int(summary.get(f"{prefix}_calls", 0))
            if component in {"index_build", "pattern", "retrieve", "other"}:
                # These components are reported per generation phase rather
                # than per individual record span.
                divisor = 1 if phase == "prefill" else decode_divisor
            else:
                divisor = max(1, row[f"{prefix}_calls"])
            row[f"{prefix}_avg_time"] = row[f"{prefix}_time"] / divisor
            row[f"{prefix}_avg_event_time"] = row[f"{prefix}_event_time"] / divisor
        row[f"{phase}_other_time"] = (
            summary.get(f"{phase}_other_time", 0.0)
            + row[f"{phase}_preAttn_ffn_time"]
            + row[f"{phase}_postAttn_ffn_time"]
        )
        row[f"{phase}_other_event_time"] = (
            summary.get(f"{phase}_other_event_time", 0.0)
            + row[f"{phase}_preAttn_ffn_event_time"]
            + row[f"{phase}_postAttn_ffn_event_time"]
        )
        row[f"{phase}_other_calls"] = (
            int(summary.get(f"{phase}_other_calls", 0))
            + row[f"{phase}_preAttn_ffn_calls"]
            + row[f"{phase}_postAttn_ffn_calls"]
        )
        other_divisor = 1 if phase == "prefill" else decode_divisor
        row[f"{phase}_other_avg_time"] = (
            row[f"{phase}_other_time"] / other_divisor
        )
        row[f"{phase}_other_avg_event_time"] = (
            row[f"{phase}_other_event_time"] / other_divisor
        )
        selection_time = (
            row[f"{phase}_retrieve_time"] + row[f"{phase}_index_build_time"]
        )
        selection_event_time = (
            row[f"{phase}_retrieve_event_time"]
            + row[f"{phase}_index_build_event_time"]
        )
        row[f"{phase}_selection_time"] = selection_time
        row[f"{phase}_selection_event_time"] = selection_event_time
        row[f"{phase}_selection_calls"] = (
            row[f"{phase}_retrieve_calls"] + row[f"{phase}_index_build_calls"]
        )
        selection_divisor = 1 if phase == "prefill" else decode_divisor
        row[f"{phase}_selection_avg_time"] = selection_time / selection_divisor
        row[f"{phase}_selection_avg_event_time"] = (
            selection_event_time / selection_divisor
        )

    layer_divisor = max(1, int(getattr(manager, "num_layers", 0) or 0))
    row["prefill_retrieve_layer_avg_time"] = row["prefill_retrieve_time"] / layer_divisor
    row["prefill_retrieve_layer_avg_event_time"] = (
        row["prefill_retrieve_event_time"] / layer_divisor
    )
    row["prefill_total_time"] = row.get("prefill_total_time") or row["prefill_total_time"]
    row["prefill_total_event_time"] = row.get("prefill_total_event_time") or row["prefill_total_event_time"]
    row["decode_total_time"] = row.get("decode_total_time") or row["decode_total_time"]
    row["decode_total_event_time"] = row.get("decode_total_event_time") or row["decode_total_event_time"]
    row["decode_total_avg_time"] = row["decode_total_time"] / decode_divisor
    row["decode_total_avg_event_time"] = row["decode_total_event_time"] / decode_divisor
    row["prefill_latency"] = row["prefill_total_time"]
    row["prefill_latency_event_time"] = row["prefill_total_event_time"]
    row["decode_latency_avg_time"] = row["decode_total_avg_time"]
    row["decode_latency_avg_event_time"] = row["decode_total_avg_event_time"]
    row["prefill_time"] = row["prefill_total_time"]
    row["prefill_event_time"] = row["prefill_total_event_time"]
    row["decode_time"] = row["decode_total_avg_time"]
    row["decode_event_time"] = row["decode_total_avg_event_time"]
    for component in ("selection", "pattern", "retrieve", "attn", "other", "write_cache"):
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
    return row
