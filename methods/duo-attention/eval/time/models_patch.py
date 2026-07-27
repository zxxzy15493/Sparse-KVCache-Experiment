"""Patch utilities for DuoAttention timing."""

from typing import Any
from forward import timed_decoder_layer_forward
from TimeManager import time_manager


def _build_timing_events():
    return {
        "pref_pattern_start":   time_manager.pref_pattern_start,
        "pref_pattern_end":     time_manager.pref_pattern_end,
        "pref_idx_start":       time_manager.pref_idx_start,
        "pref_idx_end":         time_manager.pref_idx_end,
        "pref_pure_start":      time_manager.pref_pure_start,
        "pref_pure_end":        time_manager.pref_pure_end,
        "dec_retrieve_start":   time_manager.dec_retrieve_start,
        "dec_retrieve_end":     time_manager.dec_retrieve_end,
        "dec_pure_start":       time_manager.dec_pure_start,
        "dec_pure_end":         time_manager.dec_pure_end,
        # new: qkv_proj / o_proj / write_cache
        "qkv_proj_start":       time_manager.qkv_proj_start,
        "qkv_proj_end":         time_manager.qkv_proj_end,
        "o_proj_start":         time_manager.o_proj_start,
        "o_proj_end":           time_manager.o_proj_end,
        "write_cache_start":    time_manager.write_cache_start,
        "write_cache_end":      time_manager.write_cache_end,
    }


def patch_model_with_timing(model: Any) -> Any:
    layers = model.model.layers
    num_layers = len(layers)
    time_manager.num_layers = num_layers
    time_manager._alloc_events()
    timing_ref = _build_timing_events()
    for layer_idx, layer in enumerate(layers):
        layer.layer_idx = layer_idx
        layer.num_layers = num_layers
        layer.self_attn.config._timing_events = timing_ref
        layer.self_attn.config._time_manager = time_manager
        DecoderLayer = layer.__class__
        layer.forward = timed_decoder_layer_forward.__get__(layer, DecoderLayer)
    return model