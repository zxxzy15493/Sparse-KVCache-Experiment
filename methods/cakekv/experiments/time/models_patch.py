"""
Patch utilities: replace decoder layer forward with timed version,
and inject attention-internal timing references into each self_attn.config.
"""

from typing import Any
from forward import timed_decoder_layer_forward
from TimeManager import time_manager


def patch_model_with_timing(model: Any) -> Any:
    layers = model.model.layers
    num_layers = len(layers)

    time_manager.num_layers = num_layers
    time_manager._alloc_events()

    for layer_idx, layer in enumerate(layers):
        layer.layer_idx = layer_idx
        layer.num_layers = num_layers
        layer.self_attn.config._time_manager = time_manager
        DecoderLayer = layer.__class__
        layer.forward = timed_decoder_layer_forward.__get__(layer, DecoderLayer)

    return model
