"""
Unified CUDA Event TimeManager for measurement.

Measures:
  Prefill: lat(s) / pattern(s) / attn(s) / ffn(s) / index(s) / write_cache(s)
  Decode: lat(ms) / attn(ms) / ffn(ms) / retrieve(ms) / write_cache(ms)  (32-step average)
"""

import torch


class TimeManager:
    def __init__(self, num_layers: int = 32, max_decode_steps: int = 512):
        self.num_layers = num_layers
        self.max_steps = max_decode_steps
        self.decode_step = 0
        self._alloc_events()
        self.prefill_latency = 0.0
        self.decode_latency = 0.0
        self.decode_steps = 0
        self.clear_rounds()

    def _alloc_events(self):
        L = self.num_layers
        M = self.max_steps + 1

        def evs(count):
            return [torch.cuda.Event(enable_timing=True) for _ in range(count)]

        self._pre_ffn_start = evs(L * M)
        self._pre_ffn_end = evs(L * M)
        self._attn_start = evs(L * M)
        self._attn_end = evs(L * M)
        self._post_ffn_start = evs(L * M)
        self._post_ffn_end = evs(L * M)
        self._qkv_proj_start = evs(L * M)
        self._qkv_proj_end = evs(L * M)
        self._o_proj_start = evs(L * M)
        self._o_proj_end = evs(L * M)
        self._write_cache_start = evs(L * M)
        self._write_cache_end = evs(L * M)
        self._retrieve_start = evs(L * M)
        self._retrieve_end = evs(L * M)
        self._pref_pattern_start = evs(L)
        self._pref_pattern_end = evs(L)
        self._pref_idx_start = evs(L)
        self._pref_idx_end = evs(L)
        self._pure_attn_start  = evs(L * M)
        self._pure_attn_end    = evs(L * M)

    def _idx(self, layer_idx: int) -> int:
        return self.decode_step * self.num_layers + layer_idx

    def record_pre_ffn_start(self, layer_idx: int): self._pre_ffn_start[self._idx(layer_idx)].record()
    def record_pre_ffn_end(self, layer_idx: int): self._pre_ffn_end[self._idx(layer_idx)].record()
    def record_attn_start(self, layer_idx: int): self._attn_start[self._idx(layer_idx)].record()
    def record_attn_end(self, layer_idx: int): self._attn_end[self._idx(layer_idx)].record()
    def record_post_ffn_start(self, layer_idx: int): self._post_ffn_start[self._idx(layer_idx)].record()
    def record_post_ffn_end(self, layer_idx: int): self._post_ffn_end[self._idx(layer_idx)].record()
    def record_qkv_proj_start(self, layer_idx: int): self._qkv_proj_start[self._idx(layer_idx)].record()
    def record_qkv_proj_end(self, layer_idx: int): self._qkv_proj_end[self._idx(layer_idx)].record()
    def record_o_proj_start(self, layer_idx: int): self._o_proj_start[self._idx(layer_idx)].record()
    def record_o_proj_end(self, layer_idx: int): self._o_proj_end[self._idx(layer_idx)].record()
    def record_write_cache_start(self, layer_idx: int): self._write_cache_start[self._idx(layer_idx)].record()
    def record_write_cache_end(self, layer_idx: int): self._write_cache_end[self._idx(layer_idx)].record()
    def record_retrieve_start(self, layer_idx: int): self._retrieve_start[self._idx(layer_idx)].record()
    def record_retrieve_end(self, layer_idx: int): self._retrieve_end[self._idx(layer_idx)].record()
    def record_pure_attn_start(self, layer_idx: int): self._pure_attn_start[self._idx(layer_idx)].record()
    def record_pure_attn_end(self, layer_idx: int): self._pure_attn_end[self._idx(layer_idx)].record()

    def finalize_decode_step(self):
        self.decode_step += 1

    @property
    def pref_pattern_start(self): return self._pref_pattern_start
    @property
    def pref_pattern_end(self): return self._pref_pattern_end
    @property
    def pref_idx_start(self): return self._pref_idx_start
    @property
    def pref_idx_end(self): return self._pref_idx_end

    def clear_rounds(self):
        self._rounds = []

    def finish_round(self):
        torch.cuda.synchronize()
        L = self.num_layers
        S = self.decode_step
        prefill_attn = prefill_ffn = prefill_pattern = prefill_idx = prefill_write_cache = 0.0
        for layer_idx in range(L):
            idx = layer_idx
            qkv_proj = self._qkv_proj_start[idx].elapsed_time(self._qkv_proj_end[idx]) / 1000.0
            o_proj = self._o_proj_start[idx].elapsed_time(self._o_proj_end[idx]) / 1000.0
            pure_attn = self._pure_attn_start[idx].elapsed_time(self._pure_attn_end[idx]) / 1000.0
            pref_pattern = self._pref_pattern_start[layer_idx].elapsed_time(self._pref_pattern_end[layer_idx]) / 1000.0
            pref_idx_t = self._pref_idx_start[layer_idx].elapsed_time(self._pref_idx_end[layer_idx]) / 1000.0
            pref_wc = self._write_cache_start[idx].elapsed_time(self._write_cache_end[idx]) / 1000.0
            rope_expand = self._attn_start[idx].elapsed_time(self._attn_end[idx]) / 1000.0 - pure_attn - qkv_proj - o_proj - pref_pattern - pref_idx_t - pref_wc
            prefill_attn += pure_attn
            prefill_ffn += self._pre_ffn_start[idx].elapsed_time(self._pre_ffn_end[idx]) / 1000.0
            prefill_ffn += self._post_ffn_start[idx].elapsed_time(self._post_ffn_end[idx]) / 1000.0
            prefill_ffn += qkv_proj + o_proj + rope_expand
            prefill_pattern += pref_pattern
            prefill_idx += pref_idx_t
            prefill_write_cache += pref_wc

        # Decode: accumulate in ms (elapsed_time returns ms), then average per step
        decode_attn = decode_ffn = decode_retrieve = decode_write_cache = 0.0
        decode_steps = min(max(S - 1, 0), 32)
        for step in range(1, decode_steps + 1):
            for layer_idx in range(L):
                idx = step * L + layer_idx
                qkv_proj_ms = self._qkv_proj_start[idx].elapsed_time(self._qkv_proj_end[idx])
                o_proj_ms = self._o_proj_start[idx].elapsed_time(self._o_proj_end[idx])
                pure_attn_ms = self._pure_attn_start[idx].elapsed_time(self._pure_attn_end[idx])
                retrieve_ms = self._retrieve_start[idx].elapsed_time(self._retrieve_end[idx])
                wc_ms = self._write_cache_start[idx].elapsed_time(self._write_cache_end[idx])
                rope_expand_ms = self._attn_start[idx].elapsed_time(self._attn_end[idx]) - pure_attn_ms - qkv_proj_ms - o_proj_ms - retrieve_ms - wc_ms
                decode_attn += pure_attn_ms
                decode_ffn += self._pre_ffn_start[idx].elapsed_time(self._pre_ffn_end[idx])
                decode_ffn += self._post_ffn_start[idx].elapsed_time(self._post_ffn_end[idx])
                decode_ffn += qkv_proj_ms + o_proj_ms + rope_expand_ms
                decode_retrieve += retrieve_ms
                decode_write_cache += wc_ms

        if decode_steps > 0:
            decode_attn /= decode_steps
            decode_ffn /= decode_steps
            decode_retrieve /= decode_steps
            decode_write_cache /= decode_steps

        self._rounds.append({
            "prefill_latency": self.prefill_latency,
            "prefill_pattern": prefill_pattern,
            "prefill_attn": prefill_attn,
            "prefill_ffn": prefill_ffn,
            "prefill_idx": prefill_idx,
            "prefill_write_cache": prefill_write_cache,
            "decode_latency": (self.decode_latency / decode_steps * 1000.0) if decode_steps > 0 else 0.0,
            "decode_attn": decode_attn,
            "decode_ffn": decode_ffn,
            "decode_retrieve": decode_retrieve,
            "decode_write_cache": decode_write_cache,
            "decode_steps": decode_steps,
        })
        self.decode_step = 0
        self.prefill_latency = 0.0
        self.decode_latency = 0.0
        self.decode_steps = 0

    def get_last_round(self) -> dict:
        return self._rounds[-1] if self._rounds else {}

    def get_averaged_report(self) -> dict:
        if not self._rounds:
            return {}
        keys = self._rounds[0].keys()
        return {key: sum(round_[key] for round_ in self._rounds) / len(self._rounds) for key in keys}


time_manager = TimeManager()