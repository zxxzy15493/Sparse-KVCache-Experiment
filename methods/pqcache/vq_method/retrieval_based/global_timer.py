import os
import time
import torch


# ============== Module-level functions for stage-based timing ==============

SYNC_TEST_TIME = bool(int(os.environ.get("SYNC_TEST_TIME", "0")))

_layer_cnt = 32
_can_recording = False

# stage_name -> List[layer_idx -> List[record]]
# record = {
#   "start_event": torch.cuda.Event,
#   "end_event": torch.cuda.Event,
#   "cpu_start": float,
#   "cpu_end": float,
#   "ended": bool,
# }
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


def set_layer(layer_cnt):
    """Set the number of layers. Called during initialization only."""
    global _layer_cnt
    _layer_cnt = int(layer_cnt)


def init_timer(layer_cnt=32):
    global _stages_storage
    set_layer(layer_cnt)
    _stages_storage = {}


def init_stage(stage_name):
    _stages_storage[stage_name] = [[] for _ in range(_layer_cnt)]


def init_all_stages():
    init_stage("prefill_preAttn_ffn")
    init_stage("prefill_write_cache")
    init_stage("prefill_attn")
    init_stage("prefill_postAttn_ffn")

    init_stage("decode_preAttn_ffn")
    init_stage("decode_write_cache")
    init_stage("decode_attn")
    init_stage("decode_postAttn_ffn")


def set_recording_state(can_recording):
    global _can_recording
    _can_recording = bool(can_recording)


def can_record():
    return _can_recording


def reset_timer():
    """
    Clear all recorded events.

    Call this before every measured generate() if you want each generate()
    to produce an independent stage breakdown.
    """
    for stage_name in _stages_storage:
        _stages_storage[stage_name] = [[] for _ in range(_layer_cnt)]


def stage_begin(stage_name, layer_idx):
    if not _can_recording:
        return

    _check_layer_idx(layer_idx)

    if stage_name not in _stages_storage:
        init_stage(stage_name)
    
    record = _new_record()
    # record["cpu_start"] = time.perf_counter()
    # if SYNC_TEST_TIME:
    #     torch.cuda.synchronize()
    
    record["start_event"].record()

    _stages_storage[stage_name][layer_idx].append(record)


def stage_end(stage_name, layer_idx):
    if not _can_recording:
        return

    # _check_layer_idx(layer_idx)

    # if stage_name not in _stages_storage:
    #     raise RuntimeError(f"stage_end({stage_name}) called before init_stage/stage_begin")

    layer_records = _stages_storage[stage_name][layer_idx]

    # if len(layer_records) == 0:
    #     raise RuntimeError(
    #         f"stage_end({stage_name}, layer_idx={layer_idx}) called before stage_begin"
    #     )

    record = layer_records[-1]

    # if record["ended"]:
    #     raise RuntimeError(
    #         f"stage_end({stage_name}, layer_idx={layer_idx}) called twice for the same record"
    #     )
    # record["cpu_end"] = time.perf_counter()
    record["end_event"].record()
    # if SYNC_TEST_TIME:
    #     torch.cuda.synchronize()

    
    record["ended"] = True


def _get_finished_records(stage_name):
    if stage_name not in _stages_storage:
        return []

    records = []
    for layer_records in _stages_storage[stage_name]:
        for record in layer_records:
            if record["ended"]:
                records.append(record)

    return records


def get_decode_token_count(stage_name=None):
    """
    Infer decode token count from recorded decode stages.

    If stage_name is provided, count tokens from that specific decode stage.
    Otherwise, use the maximum count among all decode stages.

    Assumption:
        For a normal decode stage, each layer records once per generated token.
        Therefore, token count = max number of records among layers.
    """
    if stage_name is not None:
        if stage_name not in _stages_storage:
            return 0

        if not _is_decode_stage(stage_name):
            return 0

        return max((len(layer_records) for layer_records in _stages_storage[stage_name]), default=0)

    max_token_count = 0

    for name, layer_storage in _stages_storage.items():
        if not _is_decode_stage(name):
            continue

        stage_token_count = max((len(layer_records) for layer_records in layer_storage), default=0)
        max_token_count = max(max_token_count, stage_token_count)

    return max_token_count


def _get_stage_total_time_ms(stage_name):
    """
    Return total stage time in ms.

    This is always layer-summed and call-summed.
    For decode stages, this is total time over all generated tokens.
    """
    torch.cuda.synchronize()

    records = _get_finished_records(stage_name)

    if len(records) == 0:
        return 0.0

    gpu_sum_ms = 0.0
    cpu_sum_ms = 0.0

    for record in records:
        gpu_sum_ms += record["start_event"].elapsed_time(record["end_event"])
        # cpu_sum_ms += (record["cpu_end"] - record["cpu_start"]) * 1000.0

    return max(cpu_sum_ms, gpu_sum_ms)


def get_stage_time(stage_name, average_decode_by_token=True):
    """
    Return stage time in ms.

    prefill_*:
        layer-summed total time.

    decode_*:
        if average_decode_by_token=True:
            layer-summed per-token average time.
        else:
            layer-summed total time over all decode tokens.
    """
    total_ms = _get_stage_total_time_ms(stage_name)

    if average_decode_by_token and _is_decode_stage(stage_name):
        token_count = get_decode_token_count(stage_name)

        if token_count > 0:
            return total_ms / token_count

    return total_ms


def get_all_stages_time(average_decode_by_token=True):
    """
    Return all stage times in ms.

    By default:
        prefill stages are total layer-summed time.
        decode stages are layer-summed per-token average time.
    """
    result = {}

    for stage_name in _stages_storage:
        result[stage_name] = get_stage_time(
            stage_name,
            average_decode_by_token=average_decode_by_token,
        )

    return result


def get_all_stages_total_time():
    """
    Return raw total times in ms.

    decode_* stages are NOT divided by token count here.
    """
    result = {}

    for stage_name in _stages_storage:
        result[stage_name] = _get_stage_total_time_ms(stage_name)

    return result


def print_all_stages_time(average_decode_by_token=True):
    times = get_all_stages_time(average_decode_by_token=average_decode_by_token)

    decode_token_count = get_decode_token_count()

    print("\n" + "=" * 70)

    if average_decode_by_token:
        print("Stage Breakdown Times (ms)")
        print("prefill_* : layer-summed total time")
        print("decode_*  : layer-summed per-token average time")
        print(f"decode token count: {decode_token_count}")
    else:
        print("Stage Breakdown Times (ms)")
        print("all stages: layer-summed total time")
        print(f"decode token count: {decode_token_count}")

    print("=" * 70)

    for stage_name, t in sorted(times.items()):
        print(f"  {stage_name:25s}: {t:10.4f} ms")

    print("=" * 70)

    return times


# ============== Legacy Timer class for backward compatibility ==============

class Timer:
    def __init__(self, layer_cnt=32):
        self.pq_compute_time = 0.0
        self.transfer_time = 0.0
        self.compute_time = 0.0

        self.layer_cnt = layer_cnt

        self.decode_pq_start = []
        self.decode_pq_end = []

        self.can_recording = False

        self.transfer_time_tuples = []

        self.start_event = None
        self.end_event = None

    def append_compute_event(self, event_s, event_e):
        self.decode_pq_start.append(event_s)
        self.decode_pq_end.append(event_e)

        if len(self.decode_pq_start) > self.layer_cnt:
            raise RuntimeError(
                f"decode_pq_start has {len(self.decode_pq_start)} events, "
                f"but layer_cnt={self.layer_cnt}"
            )

    def set_start_end_event(self, s, e):
        self.start_event = s
        self.end_event = e

    def get_decode_time_parts(self):
        """
        Legacy interface.

        Returns:
            pq, non_pq, transfer_time, total_time

        Unit:
            milliseconds.
        """
        torch.cuda.synchronize()

        if self.start_event is None or self.end_event is None:
            return 0.0, 0.0, 0.0, 0.0

        n = min(len(self.decode_pq_start), len(self.decode_pq_end))

        if n == 0:
            total_time = self.start_event.elapsed_time(self.end_event)
            return 0.0, total_time, 0.0, total_time

        pq = 0.0
        non_pq = 0.0

        for i in range(n):
            pq += self.decode_pq_start[i].elapsed_time(self.decode_pq_end[i])

        for i in range(1, n):
            non_pq += self.decode_pq_end[i - 1].elapsed_time(self.decode_pq_start[i])

        non_pq += self.start_event.elapsed_time(self.decode_pq_start[0])
        non_pq += self.decode_pq_end[n - 1].elapsed_time(self.end_event)

        transfer_time = 0.0

        for event_s, event_e in self.transfer_time_tuples:
            transfer_time += event_s.elapsed_time(event_e)

        self.transfer_time_tuples = []

        total_time = self.start_event.elapsed_time(self.end_event)

        return pq, non_pq, transfer_time, total_time

    def append_transfer_time_tuples(self, a, b):
        self.transfer_time_tuples.append((a, b))

    def set_recording_state(self, can_recording):
        global _can_recording
        self.can_recording = bool(can_recording)
        _can_recording = bool(can_recording)

    def can_record(self):
        return _can_recording


global_timer = Timer()
