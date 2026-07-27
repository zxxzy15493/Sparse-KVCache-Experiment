import time

import torch

_layer_cnt = 32
_can_recording = False
_stages_storage = {}


def _is_decode_stage(stage_name):
    return stage_name.startswith("decode_")


def _check_layer_idx(layer_idx):
    if layer_idx < 0 or layer_idx >= _layer_cnt:
        raise IndexError(
            f"layer_idx={layer_idx} is out of range, expected [0, {_layer_cnt - 1}]"
        )


def _new_record():
    return {
        "start_event": torch.cuda.Event(enable_timing=True),
        "end_event": torch.cuda.Event(enable_timing=True),
        "cpu_start": 0.0,
        "cpu_end": 0.0,
        "ended": False,
    }


def _synchronize_for_statistics():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def set_layer(layer_cnt):
    global _layer_cnt
    _layer_cnt = int(layer_cnt)


def init_timer(layer_cnt=32):
    global _stages_storage
    set_layer(layer_cnt)
    _stages_storage = {}


def init_stage(stage_name):
    _stages_storage[stage_name] = [[] for _ in range(_layer_cnt)]


def init_all_stages():
    for prefix in ("prefill", "decode"):
        init_stage(f"{prefix}_pre_ffn")
        init_stage(f"{prefix}_attn")
        init_stage(f"{prefix}_post_ffn")
        init_stage(f"{prefix}_index_build")
        init_stage(f"{prefix}_retrieve")
        init_stage(f"{prefix}_load")
        init_stage(f"{prefix}_unload")


def set_recording_state(can_recording):
    global _can_recording
    _can_recording = bool(can_recording)


def can_record():
    return _can_recording


def reset_timer():
    for stage_name in _stages_storage:
        _stages_storage[stage_name] = [[] for _ in range(_layer_cnt)]


def stage_begin(stage_name, layer_idx):
    if not _can_recording:
        return
    _check_layer_idx(layer_idx)
    if stage_name not in _stages_storage:
        init_stage(stage_name)
    record = _new_record()
    record["cpu_start"] = time.perf_counter()
    record["start_event"].record()
    _stages_storage[stage_name][layer_idx].append(record)


def stage_end(stage_name, layer_idx):
    if not _can_recording:
        return
    _check_layer_idx(layer_idx)
    if stage_name not in _stages_storage:
        raise RuntimeError(f"stage_end({stage_name}) called before stage_begin")
    layer_records = _stages_storage[stage_name][layer_idx]
    if len(layer_records) == 0:
        raise RuntimeError(
            f"stage_end({stage_name}, layer_idx={layer_idx}) called before stage_begin"
        )
    record = layer_records[-1]
    if record["ended"]:
        raise RuntimeError(
            f"stage_end({stage_name}, layer_idx={layer_idx}) called twice"
        )
    record["end_event"].record()
    record["cpu_end"] = time.perf_counter()
    record["ended"] = True


def _get_finished_records(stage_name):
    if stage_name not in _stages_storage:
        return []
    records = []
    for layer_records in _stages_storage[stage_name]:
        records.extend(record for record in layer_records if record["ended"])
    return records


def get_decode_token_count(stage_name=None):
    if stage_name is not None:
        if stage_name not in _stages_storage or not _is_decode_stage(stage_name):
            return 0
        return max(
            (len(layer_records) for layer_records in _stages_storage[stage_name]),
            default=0,
        )

    max_token_count = 0
    for name, layer_storage in _stages_storage.items():
        if not _is_decode_stage(name):
            continue
        stage_token_count = max((len(records) for records in layer_storage), default=0)
        max_token_count = max(max_token_count, stage_token_count)
    return max_token_count


def _get_stage_total_time_ms(stage_name):
    records = _get_finished_records(stage_name)
    if len(records) == 0:
        return 0.0

    gpu_sum_ms = 0.0
    cpu_sum_ms = 0.0
    for record in records:
        gpu_sum_ms += record["start_event"].elapsed_time(record["end_event"])
        cpu_sum_ms += (record["cpu_end"] - record["cpu_start"]) * 1000.0
    return max(cpu_sum_ms, gpu_sum_ms)


def _get_stage_exclusive_total_time_ms(stage_name):
    total_ms = _get_stage_total_time_ms(stage_name)
    if not stage_name.endswith("_pre_ffn"):
        return total_ms

    prefix = stage_name[: -len("_pre_ffn")]
    for nested_name in (
        f"{prefix}_index_build",
        f"{prefix}_retrieve",
        f"{prefix}_load",
    ):
        total_ms -= _get_stage_total_time_ms(nested_name)
    return max(0.0, total_ms)


def _get_stage_time_unsynced(stage_name, average_decode_by_token=True):
    total_ms = _get_stage_exclusive_total_time_ms(stage_name)
    if average_decode_by_token and _is_decode_stage(stage_name):
        token_count = get_decode_token_count(stage_name)
        if token_count > 0:
            return total_ms / token_count
    return total_ms


def get_stage_time(stage_name, average_decode_by_token=True):
    _synchronize_for_statistics()
    return _get_stage_time_unsynced(stage_name, average_decode_by_token)


def get_all_stages_time(average_decode_by_token=True):
    _synchronize_for_statistics()
    return {
        stage_name: _get_stage_time_unsynced(stage_name, average_decode_by_token)
        for stage_name in _stages_storage
    }


def get_all_stages_total_time():
    _synchronize_for_statistics()
    return {
        stage_name: _get_stage_total_time_ms(stage_name)
        for stage_name in _stages_storage
    }


def print_all_stages_time(average_decode_by_token=True):
    times = get_all_stages_time(average_decode_by_token=average_decode_by_token)
    decode_token_count = get_decode_token_count()
    print("\n" + "=" * 70)
    if average_decode_by_token:
        print("Stage Breakdown Times (ms)")
        print("prefill_* : layer-summed total time")
        print("decode_*  : layer-summed per-token average time")
    else:
        print("Stage Breakdown Times (ms)")
        print("all stages: layer-summed total time")
    print(f"decode token count: {decode_token_count}")
    print("=" * 70)
    for stage_name, t in sorted(times.items()):
        print(f"  {stage_name:25s}: {t:10.4f} ms")
    print("=" * 70)
    return times


class Timer:
    def __init__(self, layer_cnt=32):
        self.layer_cnt = layer_cnt
        self.decode_pq_start = []
        self.decode_pq_end = []
        self.transfer_time_tuples = []
        self.start_event = None
        self.end_event = None

    def set_recording_state(self, can_recording):
        set_recording_state(can_recording)

    def can_record(self):
        return can_record()

    def append_compute_event(self, event_s, event_e):
        self.decode_pq_start.append(event_s)
        self.decode_pq_end.append(event_e)

    def append_transfer_time_tuples(self, event_s, event_e):
        self.transfer_time_tuples.append((event_s, event_e))

    def set_start_end_event(self, event_s, event_e):
        self.start_event = event_s
        self.end_event = event_e


global_timer = Timer()
