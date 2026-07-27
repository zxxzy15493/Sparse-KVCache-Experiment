import os
from typing import Dict, List, Optional

import torch


SYNC_TEST_TIME = bool(int(os.environ.get("SYNC_TEST_TIME", "0")))

DEFAULT_STAGES = (
  "prefill_pre_ffn",
  "prefill_build_index",
  "prefill_attn",
  "prefill_post_ffn",
  "decode_pre_ffn",
  "decode_write_cache",
  "decode_retrieve",
  "decode_attn",
  "decode_post_ffn",
)

_layer_count = 32
_can_recording = False
_stages_storage: Dict[str, List[List[dict]]] = {}


def _is_decode_stage(stage_name: str) -> bool:
  return stage_name.startswith("decode_")


def _empty_stage_storage() -> List[List[dict]]:
  return [[] for _ in range(_layer_count)]


def _new_record() -> dict:
  return {
    "start_event": torch.cuda.Event(enable_timing=True),
    "end_event": torch.cuda.Event(enable_timing=True),
    "ended": False,
  }


def init_timer(layer_count: int = 32) -> None:
  global _layer_count, _stages_storage
  _layer_count = int(layer_count)
  _stages_storage = {}
  init_all_stages()


def init_stage(stage_name: str) -> None:
  _stages_storage[stage_name] = _empty_stage_storage()


def init_all_stages() -> None:
  for stage_name in DEFAULT_STAGES:
    init_stage(stage_name)


def set_recording_state(can_recording: bool) -> None:
  global _can_recording
  _can_recording = bool(can_recording)


def can_record() -> bool:
  return _can_recording


def reset_timer() -> None:
  for stage_name in list(_stages_storage.keys()):
    _stages_storage[stage_name] = _empty_stage_storage()


def stage_begin(stage_name: str, layer_idx: int) -> None:
  if not _can_recording:
    return
  if layer_idx < 0 or layer_idx >= _layer_count:
    raise IndexError(f"layer_idx={layer_idx} out of range [0, {_layer_count - 1}]")
  if stage_name not in _stages_storage:
    init_stage(stage_name)
  if SYNC_TEST_TIME:
    torch.cuda.synchronize()
  record = _new_record()
  record["start_event"].record()
  _stages_storage[stage_name][layer_idx].append(record)


def stage_end(stage_name: str, layer_idx: int) -> None:
  if not _can_recording:
    return
  if stage_name not in _stages_storage:
    return
  if layer_idx < 0 or layer_idx >= _layer_count:
    raise IndexError(f"layer_idx={layer_idx} out of range [0, {_layer_count - 1}]")
  layer_records = _stages_storage[stage_name][layer_idx]
  if not layer_records:
    return
  record = layer_records[-1]
  if record["ended"]:
    return
  record["end_event"].record()
  if SYNC_TEST_TIME:
    torch.cuda.synchronize()
  record["ended"] = True


def _finished_records(stage_name: str) -> List[dict]:
  records = []
  for layer_records in _stages_storage.get(stage_name, []):
    records.extend(record for record in layer_records if record["ended"])
  return records


def get_decode_token_count(stage_name: Optional[str] = None) -> int:
  if stage_name is not None:
    if not _is_decode_stage(stage_name):
      return 0
    return max((len(records) for records in _stages_storage.get(stage_name, [])), default=0)

  count = 0
  for name, layer_storage in _stages_storage.items():
    if _is_decode_stage(name):
      count = max(count, max((len(records) for records in layer_storage), default=0))
  return count


def get_stage_total_time_ms(stage_name: str) -> float:
  torch.cuda.synchronize()
  total_ms = 0.0
  for record in _finished_records(stage_name):
    total_ms += record["start_event"].elapsed_time(record["end_event"])
  return total_ms


def get_stage_time_ms(stage_name: str, average_decode_by_token: bool = True) -> float:
  total_ms = get_stage_total_time_ms(stage_name)
  if average_decode_by_token and _is_decode_stage(stage_name):
    token_count = get_decode_token_count(stage_name)
    if token_count > 0:
      return total_ms / token_count
  return total_ms


def get_all_stage_times_ms(average_decode_by_token: bool = True) -> Dict[str, float]:
  return {
    stage_name: get_stage_time_ms(stage_name, average_decode_by_token)
    for stage_name in _stages_storage
  }


def get_summary_times_ms(average_decode_by_token: bool = True) -> Dict[str, float]:
  times = get_all_stage_times_ms(average_decode_by_token)
  times["prefill_others"] = times.get("prefill_pre_ffn", 0.0) + times.get("prefill_post_ffn", 0.0)
  times["decode_others"] = times.get("decode_pre_ffn", 0.0) + times.get("decode_post_ffn", 0.0)
  return times


def print_summary(average_decode_by_token: bool = True) -> Dict[str, float]:
  times = get_summary_times_ms(average_decode_by_token)
  decode_count = get_decode_token_count()
  print("Stage breakdown (ms):", flush=True)
  if average_decode_by_token:
    print(f" prefill_*: total, decode_*: per-token average, decode_token_count={decode_count}", flush=True)
  for stage_name in sorted(times):
    print(f" {stage_name:24s}: {times[stage_name]:10.4f} ms", flush=True)
  return times
