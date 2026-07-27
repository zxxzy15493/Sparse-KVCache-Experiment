import torch
from contextlib import contextmanager
from collections import defaultdict
import json
import time


class Timer:
  def __init__(self, layer_cnt = 32):
    self.pq_compute_time = 0

    self.transfer_time = 0

    self.compute_time = 0

    self.layer_cnt = layer_cnt

    self.decode_pq_start = []
    self.decode_pq_end = []

    self.can_recording = False

    self.transfer_time_tuples = []

  def append_compute_event(self, event_s, event_e):
    self.decode_pq_start.append(event_s)
    assert len(self.decode_pq_start) <= self.layer_cnt
    
    self.decode_pq_end.append(event_e)
    assert len(self.decode_pq_end) <= self.layer_cnt
  
  def set_start_end_event(self, s, e):
    self.start_event = s
    self.end_event = e

  def get_decode_time_parts(self,):
    pq = 0
    non_pq = 0
    torch.cuda.synchronize() 
    for i in range(self.layer_cnt):
      pq += self.decode_pq_start[i].elapsed_time(self.decode_pq_end[i])

    for i in range(1, self.layer_cnt):
      non_pq += self.decode_pq_end[i-1].elapsed_time(self.decode_pq_start[i])
    
    non_pq += self.start_event.elapsed_time(self.decode_pq_start[0])
    non_pq += self.decode_pq_end[self.layer_cnt-1].elapsed_time(self.end_event)
    
    transfer_time = 0

    for t in self.transfer_time_tuples:
      time = t[0].elapsed_time(t[1])
      transfer_time += time

    self.transfer_time_tuples = []

    return pq, non_pq, transfer_time, self.start_event.elapsed_time(self.end_event)

  def append_transfer_time_tuples(self, a, b):
    self.transfer_time_tuples.append((a,b))

  def set_recording_state(self, can_recording):
    self.can_recording = can_recording

  def can_record(self):
    return self.can_recording

global_timer = Timer()


class PrefillComponentTimer:
  """CUDA-event timer for named prefill components.

  The timer records component spans lazily and computes elapsed times only
  when summarized. This keeps CUDA synchronization out of the hot path.
  """

  def __init__(self):
    self.can_recording = False
    self.reset()

  def reset(self):
    self._records = []
    self._active = {}
    self._total_start = None
    self._total_end = None
    self._meta = {}

  def set_recording_state(self, can_recording):
    self.can_recording = can_recording

  def can_record(self):
    return self.can_recording

  def set_meta(self, **kwargs):
    self._meta.update(kwargs)

  def _make_stamp(self):
    if torch.cuda.is_available():
      event = torch.cuda.Event(enable_timing=True)
      event.record()
      return ("cuda", event)
    return ("cpu", time.perf_counter())

  @staticmethod
  def _elapsed_ms(start, end):
    if start[0] == "cuda":
      return start[1].elapsed_time(end[1])
    return (end[1] - start[1]) * 1000.0

  def start_total(self):
    if not self.can_recording:
      return
    self._total_start = self._make_stamp()

  def end_total(self):
    if not self.can_recording:
      return
    self._total_end = self._make_stamp()

  def start(self, component, layer_id=None, phase="prefill", **kwargs):
    if not self.can_recording:
      return
    key = (phase, layer_id, component)
    if key in self._active:
      raise RuntimeError(f"timer component already active: {key}")
    self._active[key] = (self._make_stamp(), kwargs)

  def end(self, component, layer_id=None, phase="prefill", **kwargs):
    if not self.can_recording:
      return
    key = (phase, layer_id, component)
    if key not in self._active:
      raise RuntimeError(f"timer component was not started: {key}")
    start, start_kwargs = self._active.pop(key)
    meta = {}
    meta.update(start_kwargs)
    meta.update(kwargs)
    self._records.append(
      {
        "phase": phase,
        "layer": layer_id,
        "component": component,
        "start": start,
        "end": self._make_stamp(),
        "meta": meta,
      }
    )

  @contextmanager
  def record(self, component, layer_id=None, phase="prefill", **kwargs):
    self.start(component, layer_id=layer_id, phase=phase, **kwargs)
    try:
      yield
    finally:
      self.end(component, layer_id=layer_id, phase=phase)

  def append_event_pair(self, component, start_event, end_event, layer_id=None, phase="prefill", **kwargs):
    if not self.can_recording:
      return
    self._records.append(
      {
        "phase": phase,
        "layer": layer_id,
        "component": component,
        "start": ("cuda", start_event),
        "end": ("cuda", end_event),
        "meta": kwargs,
      }
    )

  def records(self, synchronize=True):
    if synchronize and torch.cuda.is_available():
      torch.cuda.synchronize()

    out = []
    for record in self._records:
      item = dict(self._meta)
      item.update(record["meta"])
      item.update(
        {
          "phase": record["phase"],
          "layer": record["layer"],
          "component": record["component"],
          "time_ms": self._elapsed_ms(record["start"], record["end"]),
        }
      )
      out.append(item)
    return out

  def summary(self, synchronize=True):
    records = self.records(synchronize=synchronize)
    by_component = defaultdict(float)
    by_layer_component = defaultdict(float)

    for item in records:
      by_component[item["component"]] += item["time_ms"]
      by_layer_component[(item["layer"], item["component"])] += item["time_ms"]

    total_ms = None
    accounted_ms = sum(by_component.values())
    unaccounted_ms = None
    if self._total_start is not None and self._total_end is not None:
      if synchronize and torch.cuda.is_available():
        torch.cuda.synchronize()
      total_ms = self._elapsed_ms(self._total_start, self._total_end)
      unaccounted_ms = total_ms - accounted_ms

    return {
      "meta": dict(self._meta),
      "total_ms": total_ms,
      "accounted_ms": accounted_ms,
      "unaccounted_ms": unaccounted_ms,
      "by_component_ms": dict(by_component),
      "by_layer_component_ms": {
        f"{layer}:{component}": value
        for (layer, component), value in by_layer_component.items()
      },
      "records": records,
    }

  def write_jsonl(self, path, synchronize=True):
    records = self.records(synchronize=synchronize)
    with open(path, "a", encoding="utf-8") as f:
      for item in records:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")


prefill_timer = PrefillComponentTimer()

